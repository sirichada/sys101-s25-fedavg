use candle_core::{DType, Result as CandleResult, Tensor, D, Module, Device};
use candle_nn::{loss, ops, Optimizer, VarBuilder, VarMap, SGD};
use candle_datasets::vision::Dataset;
use candle_app::{LinearModel, Model, ModelStatus};
use tokio::net::{TcpStream, TcpListener};
use tokio::io::{AsyncReadExt, AsyncWriteExt, AsyncBufReadExt, BufReader};
use base64::Engine;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::Mutex;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;
use std::io::{self, Write};
use tokio::time::sleep;
use tokio::time::Duration;
use clap::{Parser, ArgAction};

// CLI arguments
#[derive(Parser)]
struct Args {
    #[arg(short, long, default_value = "127.0.0.1:50051")]
    server: String,
    
    #[arg(short, long, default_value = "127.0.0.1:0")]
    listen: String,
    
    #[arg(short, long, action = ArgAction::SetTrue)]
    interactive: bool,
}

struct Client {
    server_addr: String,
    models: HashMap<String, (LinearModel, VarMap, ModelStatus)>,
    dataset: Arc<Dataset>,
    local_addr: String,
}

impl Client {
    fn new(server_addr: &str) -> Self {
        // Load full MNIST dataset
        let full_dataset = candle_datasets::vision::mnist::load().expect("Failed to load MNIST dataset");
        
        // Create a random subset of the training data for this client
        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..full_dataset.train_images.dim(0).unwrap()).collect();
        indices.shuffle(&mut rng);
        
        // Select 10,000 random samples (or fewer if the dataset is smaller)
        let subset_size = std::cmp::min(10_000, indices.len());
        let selected_indices = &indices[..subset_size];

        // Create an index tensor to select the samples
        let index_tensor = Tensor::from_vec(
            selected_indices.iter().map(|&i| i as i64).collect::<Vec<i64>>(),
            (subset_size,),
            &Device::Cpu,
        ).unwrap();

        // Select random samples using the index tensor
        let train_images = full_dataset.train_images.index_select(&index_tensor, 0).unwrap();
        let train_labels = full_dataset.train_labels.index_select(&index_tensor, 0).unwrap();

        // Create the dataset with the selected subset
        let dataset = Dataset {
            train_images,
            train_labels,
            test_images: full_dataset.test_images.clone(),
            test_labels: full_dataset.test_labels.clone(),
            labels: full_dataset.labels,
        };

        Client {
            server_addr: server_addr.to_string(),
            models: HashMap::new(),
            dataset: Arc::new(dataset),
            local_addr: String::new(),
        }
    }

    // Join a federated learning session for a specific model
    async fn join(&mut self, server_ip: &str, model_name: &str) -> Result<(TcpStream, TcpListener)> {
        println!("Connecting to server at {}", server_ip);
        let mut stream = TcpStream::connect(server_ip).await?;
        
        // Start listening for incoming connections from the server
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let local_addr = listener.local_addr()?.to_string();
        self.local_addr = local_addr.clone();

        // Register with the server
        let message = format!("REGISTER|{}|{}", local_addr, model_name);
        stream.write_all(message.as_bytes()).await?;
        stream.flush().await?;

        // Wait for server acknowledgment
        let mut buffer = [0; 1024];
        let n = stream.read(&mut buffer).await?;
        let response = String::from_utf8_lossy(&buffer[..n]);
        println!("Server response: {}", response);

        // Notify server that we're ready for training
        stream.write_all(b"READY").await?;
        stream.flush().await?;

        Ok((stream, listener))
    }

    // Train the local model
    async fn train(&mut self, model_name: &str, epochs: usize) -> CandleResult<()> {
        if let Some((model, varmap, status)) = &mut self.models.get_mut(model_name) {
            if *status != ModelStatus::Initialized && *status != ModelStatus::Ready {
                println!("Client model {} is already training or in invalid state", model_name);
                return Ok(());
            }
            *status = ModelStatus::Training;

            let dev = Device::Cpu;
            let train_images = self.dataset.train_images.to_device(&dev)?;
            let train_labels = self.dataset.train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
            let mut sgd = SGD::new(varmap.all_vars(), 0.1)?;

            let test_images = self.dataset.test_images.to_device(&dev)?;
            let test_labels = self.dataset.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;

            for epoch in 1..=epochs {
                // Forward pass
                let logits = Module::forward(model, &train_images)?;
                let log_sm = ops::log_softmax(&logits, D::Minus1)?;
                let loss = loss::nll(&log_sm, &train_labels)?;
                
                // Backward pass and optimization
                sgd.backward_step(&loss)?;

                // Evaluate on test set
                let test_logits = Module::forward(model, &test_images)?;
                let sum_ok = test_logits
                    .argmax(D::Minus1)?
                    .eq(&test_labels)?
                    .to_dtype(DType::F32)?
                    .sum_all()?
                    .to_scalar::<f32>()?;
                let accuracy = sum_ok / test_labels.dims1()? as f32;

                println!(
                    "Client trained epoch {}/{} for model {}, accuracy: {:.2}%",
                    epoch,
                    epochs,
                    model_name,
                    accuracy * 100.0
                );
            }

            *status = ModelStatus::Ready;
            println!("Client completed training for {}", model_name);
        } else {
            return Err(candle_core::Error::Msg(format!("Model {} not found", model_name)));
        }
        Ok(())
    }

    // Get current model parameters and status
    fn get(&self, model_name: &str) -> Option<(Vec<f32>, Vec<f32>, ModelStatus)> {
        if let Some((model, _, status)) = &self.models.get(model_name) {
            // Extract model weights and biases
            let weights_data = model.weight().ok()?.to_vec2::<f32>().ok()?
                .into_iter().flatten().collect::<Vec<f32>>();
            let bias_data = model.bias().ok()?.to_vec1::<f32>().ok()?;
            Some((weights_data, bias_data, status.clone()))
        } else {
            None
        }
    }

    // Test model accuracy on local test dataset
    fn test(&self, model_name: &str) -> CandleResult<f32> {
        if let Some((model, _, _)) = &self.models.get(model_name) {
            let dev = Device::Cpu;
            let test_images = self.dataset.test_images.to_device(&dev)?;
            let test_labels = self.dataset.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;
            
            let logits = Module::forward(model, &test_images)?;
            let sum_ok = logits
                .argmax(D::Minus1)?
                .eq(&test_labels)?
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()?;
                
            let accuracy = sum_ok / test_labels.dims1()? as f32;
            Ok(accuracy)
        } else {
            Err(candle_core::Error::Msg(format!("Model {} not found", model_name)))
        }
    }

    // Main client processing loop
    async fn run_inner(listener: TcpListener, client: Arc<Mutex<Self>>) -> Result<()> {
        println!("Client listening on {}", listener.local_addr()?);

        loop {
            let (mut stream, _) = listener.accept().await?;
            let mut buffer = [0; 65536];
            
            match stream.read(&mut buffer).await {
                Ok(n) => {
                    let message = String::from_utf8_lossy(&buffer[..n]);
                    let parts: Vec<&str> = message.split('|').collect();

                    let mut client_guard = client.lock().await;
                    match parts[0] {
                        "TRAIN" if parts.len() >= 5 => {
                            let model_name = parts[1];
                            println!("Received TRAIN request for {} with {} epochs", model_name, parts[4]);
                            
                            // Deserialize model parameters
                            let weights_data: Vec<f32> = bincode::deserialize(
                                &base64::engine::general_purpose::STANDARD.decode(parts[2])?,
                            )?;
                            let bias_data: Vec<f32> = bincode::deserialize(
                                &base64::engine::general_purpose::STANDARD.decode(parts[3])?,
                            )?;
                            let epochs: usize = parts[4].parse().map_err(|e| anyhow::anyhow!("Invalid epochs: {}", e))?;

                            // Create tensors from received data
                            let weights = Tensor::from_vec(weights_data, &[10, 784], &Device::Cpu)?;
                            let bias = Tensor::from_vec(bias_data, &[10], &Device::Cpu)?;
                            
                            // Create or update model
                            let varmap = VarMap::new();
                            let vs = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
                            let model = LinearModel::new(vs)?;
                            
                            // Set model parameters
                            {
                                let mut data = varmap.data().lock().unwrap();
                                data.get_mut("linear.weight").unwrap().set(&weights)?;
                                data.get_mut("linear.bias").unwrap().set(&bias)?;
                            }

                            // Store the model
                            client_guard.models.insert(model_name.to_string(), (model, varmap, ModelStatus::Initialized));
                            
                            // Train the model locally
                            client_guard.train(model_name, epochs).await?;

                            // Send updated parameters back to server
                            if let Some((model, _, _)) = client_guard.models.get(model_name) {
                                let weights_data = model.weight()?.to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
                                let bias_data = model.bias()?.to_vec1::<f32>()?;
                                
                                let response = format!(
                                    "UPDATE|{}|{}",
                                    base64::engine::general_purpose::STANDARD.encode(&bincode::serialize(&weights_data)?),
                                    base64::engine::general_purpose::STANDARD.encode(&bincode::serialize(&bias_data)?)
                                );
                                
                                stream.write_all(response.as_bytes()).await?;
                                stream.flush().await?;
                            }
                        }
                        "GET" if parts.len() >= 2 => {
                            let model_name = parts[1];
                            println!("Received GET request for {}", model_name);
                            
                            if let Some((weights_data, bias_data, status)) = client_guard.get(model_name) {
                                let weights = bincode::serialize(&weights_data)?;
                                let bias = bincode::serialize(&bias_data)?;
                                
                                let response = format!(
                                    "MODEL|{}|{}|{}",
                                    base64::engine::general_purpose::STANDARD.encode(&weights),
                                    base64::engine::general_purpose::STANDARD.encode(&bias),
                                    status.to_string()
                                );
                                
                                stream.write_all(response.as_bytes()).await?;
                            } else {
                                stream.write_all(b"No model available").await?;
                            }
                            
                            stream.flush().await?;
                        }
                        "TEST" if parts.len() >= 2 => {
                            let model_name = parts[1];
                            println!("Received TEST request for {}", model_name);
                            
                            match client_guard.test(model_name) {
                                Ok(accuracy) => {
                                    let response = format!("ACCURACY|{}", accuracy);
                                    stream.write_all(response.as_bytes()).await?;
                                }
                                Err(e) => {
                                    stream.write_all(format!("Error: {}", e).as_bytes()).await?;
                                }
                            }
                            
                            stream.flush().await?;
                        }
                        "COMPLETE" => {
                            println!("Received COMPLETE notification from server: Training round completed");
                        }
                        _ => {
                            println!("Received unrecognized message: {}", message);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error reading from server: {}", e);
                    break;
                }
            }
        }
        
        Ok(())
    }

    // Handle interactive CLI commands
    async fn handle_commands(client: Arc<Mutex<Client>>) -> Result<()> {
        println!("Client CLI");
        println!("Available commands:");
        println!("  JOIN <server_ip> <model> - Join a federated learning session");
        println!("  GET <model> - Get local model parameters and status");
        println!("  TEST <model> - Test local model accuracy");
        println!("  exit - Exit the client");
        
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::new(stdin);
        let mut input = String::new();

        loop {
            print!("> ");
            io::stdout().flush()?;
            input.clear();
            reader.read_line(&mut input).await?;
            let input = input.trim();

            if input.eq_ignore_ascii_case("exit") {
                println!("Shutting down client...");
                break;
            }

            let parts: Vec<&str> = input.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            match parts[0].to_uppercase().as_str() {
                "JOIN" => {
                    if parts.len() < 3 {
                        println!("Usage: JOIN <server_ip> <model>");
                        continue;
                    }
                    
                    let server_ip = parts[1];
                    let model_name = parts[2];
                    
                    let mut client_guard = client.lock().await;
                    match client_guard.join(server_ip, model_name).await {
                        Ok(_) => println!("Successfully joined model {} on server {}", model_name, server_ip),
                        Err(e) => eprintln!("Failed to join: {}", e),
                    }
                }
                "GET" => {
                    if parts.len() < 2 {
                        println!("Usage: GET <model>");
                        continue;
                    }
                    
                    let model_name = parts[1];
                    let client_guard = client.lock().await;
                    
                    if let Some((weights, bias, status)) = client_guard.get(model_name) {
                        println!("Model: {}", model_name);
                        println!("Status: {}", status.to_string());
                        println!("Weights size: {}, Bias size: {}", weights.len(), bias.len());
                    } else {
                        println!("Model {} not found locally", model_name);
                    }
                }
                "TEST" => {
                    if parts.len() < 2 {
                        println!("Usage: TEST <model>");
                        continue;
                    }
                    
                    let model_name = parts[1];
                    let client_guard = client.lock().await;
                    
                    match client_guard.test(model_name) {
                        Ok(accuracy) => println!("Local model {} accuracy: {:.2}%", model_name, accuracy * 100.0),
                        Err(e) => eprintln!("Test error: {}", e),
                    }
                }
                _ => {
                    println!("Invalid command. Available commands: JOIN, GET, TEST, exit");
                }
            }
        }
        
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    let client = Arc::new(Mutex::new(Client::new(&args.server)));
    println!("Client initialized, connecting to server at {}", args.server);
    
    // If in interactive mode, only set up listener but wait for explicit JOIN command
    if args.interactive {
        let client_clone = Arc::clone(&client);
        
        let listener = TcpListener::bind(args.listen).await?;
        println!("Client listening on {}", listener.local_addr()?);
        
        // Start listener thread
        let client_listener = Arc::clone(&client_clone);
        tokio::spawn(async move {
            if let Err(e) = Client::run_inner(listener, client_listener).await {
                eprintln!("Client listener error: {}", e);
            }
        });
        
        // Start interactive CLI
        Client::handle_commands(client_clone).await?;
    } else {
        // Auto-join the server for the mnist model
        let (_stream, listener) = {
            let mut client_guard = client.lock().await;
            client_guard.join(&args.server, "mnist").await?
        };
        
        println!("Client setup complete, listening on {}", client.lock().await.local_addr);
        
        // Run in non-interactive mode
        let client_clone = Arc::clone(&client);
        let handle = tokio::spawn(async move {
            if let Err(e) = Client::run_inner(listener, client_clone).await {
                eprintln!("Client run error: {}", e);
            }
        });
        
        // Wait for Ctrl+C to terminate
        tokio::signal::ctrl_c().await?;
        println!("Client terminating...");
    }
    
    Ok(())
}