use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use candle_core::{DType, Result as CandleResult, Tensor, Device, D};
use candle_nn::{VarBuilder, VarMap};
use candle_app::{LinearModel, Model, ModelStatus};
use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt, AsyncBufReadExt, BufReader};
use base64::Engine;
use anyhow::Result;
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};
use std::io::{self, Write};
use rand::seq::SliceRandom;
use rand::thread_rng;

struct Server {
    clients: HashMap<String, String>, // client_ip -> model_name
    ready_clients: HashMap<String, bool>, // client_ip -> is_ready
    models: HashMap<String, (LinearModel, VarMap, ModelStatus)>, // model_name -> (model, varmap, status)
    test_datasets: HashMap<String, candle_datasets::vision::Dataset>, // model_name -> test_dataset
}

impl Server {
    fn new() -> Self {
        Server {
            clients: HashMap::new(),
            ready_clients: HashMap::new(),
            models: HashMap::new(),
            test_datasets: HashMap::new(),
        }
    }

    // Register a client for a specific model
    fn register(&mut self, client_ip: String, model_name: String) -> Result<()> {
        println!("Registering client {} for model {}", client_ip, model_name);
        self.clients.insert(client_ip.clone(), model_name);
        self.ready_clients.insert(client_ip, false);
        Ok(())
    }

    // Mark a client as ready for training
    fn mark_ready(&mut self, client_ip: &str) {
        if let Some(ready) = self.ready_clients.get_mut(client_ip) {
            *ready = true;
            println!("Client {} marked as ready", client_ip);
        }
    }

    // Remove a client from tracking
    fn remove_client(&mut self, client_ip: &str) {
        println!("Removing client {} from tracking", client_ip);
        self.clients.remove(client_ip);
        self.ready_clients.remove(client_ip);
    }

    // Initialize or reset a model
    fn init(&mut self, model_name: &str) -> CandleResult<()> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let model = LinearModel::new(vs)?;
        println!("Initializing model {}", model_name);
        
        // Load or create test dataset if needed
        if !self.test_datasets.contains_key(model_name) {
            if model_name == "mnist" {
                self.test_datasets.insert(model_name.to_string(), candle_datasets::vision::mnist::load()?);
            } else {
                return Err(candle_core::Error::Msg(format!("Unknown model type: {}", model_name)));
            }
        }
        
        self.models.insert(model_name.to_string(), (model, varmap, ModelStatus::Initialized));
        Ok(())
    }

    // Get a specific model
    fn get_model(&self, model_name: &str) -> Option<&(LinearModel, VarMap, ModelStatus)> {
        self.models.get(model_name)
    }

    // Aggregate model updates from clients using Federated Averaging
    async fn aggregate_updates(&mut self, model_name: &str, updates: Vec<(Vec<f32>, Vec<f32>)>) -> CandleResult<()> {
        let (model, varmap, status) = self.models.get_mut(model_name).ok_or_else(|| {
            candle_core::Error::Msg(format!("Model {} not initialized", model_name))
        })?;

        let mut weights_sum: Vec<f32> = vec![0.0; 10 * 784];
        let mut bias_sum: Vec<f32> = vec![0.0; 10];
        let num_clients = updates.len() as f32;

        // Sum all client updates
        for (weights_data, bias_data) in &updates {
            for (i, &w) in weights_data.iter().enumerate() {
                weights_sum[i] += w;
            }
            for (i, &b) in bias_data.iter().enumerate() {
                bias_sum[i] += b;
            }
        }

        // Compute averages
        let weights_avg: Vec<f32> = weights_sum.into_iter().map(|w| w / num_clients).collect();
        let bias_avg: Vec<f32> = bias_sum.into_iter().map(|b| b / num_clients).collect();

        // Update the global model
        let weights_tensor = Tensor::from_vec(weights_avg, &[10, 784], &Device::Cpu)?;
        let bias_tensor = Tensor::from_vec(bias_avg, &[10], &Device::Cpu)?;

        let mut data = varmap.data().lock().unwrap();
        data.get_mut("linear.weight")
            .ok_or_else(|| candle_core::Error::Msg("linear.weight missing".to_string()))?
            .set(&weights_tensor)?;
        data.get_mut("linear.bias")
            .ok_or_else(|| candle_core::Error::Msg("linear.bias missing".to_string()))?
            .set(&bias_tensor)?;

        *status = ModelStatus::Ready;
        println!("Global model {} updated with {} client updates", model_name, updates.len());
        Ok(())
    }

    // Start a training process for a model
    async fn train(&self, model_name: &str, clients_to_use: usize, rounds: usize, epochs: usize, server: Arc<Mutex<Self>>) -> Result<()> {
        let (tx, mut rx) = mpsc::channel(32);
        
        // Update model status to training
        {
            let mut server_guard = server.lock().await;
            if let Some((_, _, status)) = server_guard.models.get_mut(model_name) {
                *status = ModelStatus::Training;
            } else {
                return Err(anyhow::anyhow!("Model {} not found", model_name));
            }
        }

        // Run training rounds
        for round in 1..=rounds {
            println!("Starting training round {} for model {}", round, model_name);

            // Get ready clients for this model
            let ready_clients: Vec<String> = {
                let server_guard = server.lock().await;
                let mut clients = server_guard.ready_clients
                    .iter()
                    .filter(|&(client_ip, &ready)| {
                        ready && server_guard.clients.get(client_ip).map_or(false, |m| m == model_name)
                    })
                    .map(|(ip, _)| ip.clone())
                    .collect::<Vec<String>>();
                
                // Select clients based on the specified count
                if clients.len() > clients_to_use {
                    let mut rng = thread_rng();
                    clients.shuffle(&mut rng);
                    clients.truncate(clients_to_use);
                    println!("Using {} of {} available clients", clients_to_use, server_guard.ready_clients.len());
                } else if clients.len() < clients_to_use {
                    println!("Warning: Only {} clients available (requested {})", clients.len(), clients_to_use);
                }
                clients
            };
            
            if ready_clients.is_empty() {
                println!("No ready clients for round {} of model {}", round, model_name);
                sleep(Duration::from_secs(1)).await;
                continue;
            }

            // Get current model parameters
            let (weights_data, bias_data) = {
                let server_guard = server.lock().await;
                if let Some((model, _, _)) = server_guard.get_model(model_name) {
                    (
                        model.weight()?.to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>(),
                        model.bias()?.to_vec1::<f32>()?
                    )
                } else {
                    return Err(anyhow::anyhow!("Model {} not initialized", model_name));
                }
            };
            
            // Serialize for transmission
            let weights = bincode::serialize(&weights_data)?;
            let bias = bincode::serialize(&bias_data)?;

            // Distribute to clients and collect responses
            let mut handles = Vec::new();
            for client_ip in &ready_clients {
                let tx = tx.clone();
                let weights = weights.clone();
                let bias = bias.clone();
                let client_ip = client_ip.clone();
                let model_name = model_name.to_string();
                
                let handle = tokio::spawn(async move {
                    match TcpStream::connect(&client_ip).await {
                        Ok(mut stream) => {
                            // Send training request to client
                            let train_message = format!(
                                "TRAIN|{}|{}|{}|{}",
                                model_name,
                                base64::engine::general_purpose::STANDARD.encode(&weights),
                                base64::engine::general_purpose::STANDARD.encode(&bias),
                                epochs
                            );
                            
                            println!("Sending TRAIN to {} for model {} with {} epochs", client_ip, model_name, epochs);
                            stream.write_all(train_message.as_bytes()).await?;
                            stream.flush().await?;

                            // Receive client response
                            let mut buffer = [0; 65536];
                            if let Ok(n) = stream.read(&mut buffer).await {
                                let response = String::from_utf8_lossy(&buffer[..n]);
                                if response.starts_with("UPDATE|") {
                                    let parts: Vec<&str> = response.split('|').collect();
                                    if parts.len() >= 3 {
                                        let weights_data: Vec<f32> = bincode::deserialize(
                                            &base64::engine::general_purpose::STANDARD.decode(parts[1])?,
                                        )?;
                                        let bias_data: Vec<f32> = bincode::deserialize(
                                            &base64::engine::general_purpose::STANDARD.decode(parts[2])?,
                                        )?;
                                        tx.send((weights_data, bias_data)).await?;
                                    }
                                }
                            }
                        }
                        Err(e) => eprintln!("Failed to connect to client {}: {}", client_ip, e),
                    }
                    Ok::<(), anyhow::Error>(())
                });
                
                handles.push(handle);
            }

            // Collect updates
            let mut updates = Vec::new();
            for _ in 0..ready_clients.len() {
                if let Some(update) = rx.recv().await {
                    updates.push(update);
                }
            }
            
            // Wait for all client communication to complete
            for handle in handles {
                if let Err(e) = handle.await {
                    eprintln!("Error in client communication task: {}", e);
                }
            }

            // Aggregate updates if any received
            if !updates.is_empty() {
                let mut server_guard = server.lock().await;
                server_guard.aggregate_updates(model_name, updates).await?;
                println!("Completed training round {} for model {}", round, model_name);
                
                // Test global model after each round
                match server_guard.test(model_name) {
                    Ok(accuracy) => println!("Global model accuracy after round {}: {:.2}%", round, accuracy * 100.0),
                    Err(e) => println!("Failed to test global model: {}", e),
                }
            } else {
                println!("No updates received in round {} for model {}", round, model_name);
            }
            
            sleep(Duration::from_secs(1)).await;
        }

        // Notify clients of completion
        let client_ips = {
            let server_guard = server.lock().await;
            server_guard.clients
                .iter()
                .filter(|(_, m)| *m == model_name)
                .map(|(ip, _)| ip.clone())
                .collect::<Vec<String>>()
        };
        
        for client_ip in client_ips {
            if let Ok(mut stream) = TcpStream::connect(&client_ip).await {
                stream.write_all(b"COMPLETE").await?;
                stream.flush().await?;
            } else {
                eprintln!("Failed to notify client {}", client_ip);
            }
        }

        // Update model status to ready
        {
            let mut server_guard = server.lock().await;
            if let Some((_, _, status)) = server_guard.models.get_mut(model_name) {
                *status = ModelStatus::Ready;
            }
        }

        Ok(())
    }

    // Test the global model accuracy
    fn test(&self, model_name: &str) -> CandleResult<f32> {
        let (model, _, _) = self.models.get(model_name).ok_or_else(|| {
            candle_core::Error::Msg(format!("Model {} not initialized", model_name))
        })?;
        
        let test_dataset = self.test_datasets.get(model_name).ok_or_else(|| {
            candle_core::Error::Msg(format!("Test dataset for model {} not loaded", model_name))
        })?;
        
        let dev = &Device::Cpu;
        let test_images = test_dataset.test_images.to_device(dev)?;
        let test_labels = test_dataset.test_labels.to_dtype(DType::U32)?.to_device(dev)?;
        
        let logits = model.forward(&test_images)?;
        let sum_ok = logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
            
        let accuracy = sum_ok / test_labels.dims1()? as f32;
        Ok(accuracy)
    }

    // Handle client connections
    async fn handle_client(stream: TcpStream, server: Arc<Mutex<Server>>) -> Result<()> {
        let mut buffer = [0; 65536];
        let peer_addr = stream.peer_addr()?.to_string();
        println!("Handling client connection from {}", peer_addr);
        let mut client_listening_addr: Option<String> = None;
        let mut client_model: Option<String> = None;

        let mut stream = stream;

        loop {
            match stream.read(&mut buffer).await {
                Ok(0) => {
                    println!("Client {} disconnected", peer_addr);
                    if let Some(ref client_ip) = client_listening_addr {
                        let mut server_guard = server.lock().await;
                        server_guard.remove_client(client_ip);
                    }
                    break;
                }
                Ok(n) => {
                    let message = String::from_utf8_lossy(&buffer[..n]).to_string();
                    let parts: Vec<&str> = message.split('|').collect();

                    let mut server_guard = server.lock().await;
                    match parts[0] {
                        "REGISTER" if parts.len() >= 2 => {
                            let client_ip = parts[1].to_string();
                            let model_name = if parts.len() >= 3 { parts[2].to_string() } else { "mnist".to_string() };
                            
                            if let Err(e) = server_guard.register(client_ip.clone(), model_name.clone()) {
                                stream.write_all(format!("Error: {}", e).as_bytes()).await?;
                            } else {
                                client_listening_addr = Some(client_ip);
                                client_model = Some(model_name);
                                stream.write_all(b"Registered successfully").await?;
                            }
                            stream.flush().await?;
                        }
                        "READY" => {
                            if let Some(ref client_ip) = client_listening_addr {
                                server_guard.mark_ready(client_ip);
                                stream.write_all(b"Waiting for training round").await?;
                                stream.flush().await?;
                            } else {
                                stream.write_all(b"Error: Client not registered").await?;
                                stream.flush().await?;
                            }
                        }
                        "GET" if parts.len() >= 2 => {
                            let model_name = parts[1];
                            if let Some((model, _, status)) = server_guard.get_model(model_name) {
                                let weights_data = model.weight()?.to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>();
                                let bias_data = model.bias()?.to_vec1::<f32>()?;
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
                                stream.write_all(format!("Model {} not found", model_name).as_bytes()).await?;
                            }
                            stream.flush().await?;
                        }
                        "TEST" if parts.len() >= 2 => {
                            let model_name = parts[1];
                            match server_guard.test(model_name) {
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
                        _ => {
                            stream.write_all(b"Invalid command").await?;
                            stream.flush().await?;
                        }
                    }
                    drop(server_guard);
                }
                Err(e) => {
                    eprintln!("Error reading from client {}: {}", peer_addr, e);
                    if let Some(ref client_ip) = client_listening_addr {
                        let mut server_guard = server.lock().await;
                        server_guard.remove_client(client_ip);
                    }
                    break;
                }
            }
        }
        Ok(())
    }

    // Handle command line interface
    async fn handle_commands(server: Arc<Mutex<Server>>) -> Result<()> {
        println!("Parameter Server CLI");
        println!("Available commands:");
        println!("  INIT <model> - Initialize or reset a model");
        println!("  TRAIN <model> <clients> <rounds> <epochs> - Start training process");
        println!("  GET <model> - Get model parameters and status");
        println!("  TEST <model> - Test model accuracy");
        println!("  exit - Exit the server");
        
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::new(stdin);

        loop {
            print!("> ");
            io::stdout().flush()?;
            
            // Create a new String for each command
            let mut input = String::new();
            reader.read_line(&mut input).await?;
            let command = input.trim().to_string();
            
            if command.eq_ignore_ascii_case("exit") {
                println!("Shutting down server...");
                break;
            }

            let parts: Vec<&str> = command.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            match parts[0].to_uppercase().as_str() {
                "INIT" => {
                    if parts.len() < 2 {
                        println!("Usage: INIT <model>");
                        continue;
                    }
                    let model_name = parts[1];
                    let mut server_guard = server.lock().await;
                    match server_guard.init(model_name) {
                        Ok(_) => println!("Model {} initialized successfully", model_name),
                        Err(e) => eprintln!("Init error: {}", e),
                    }
                }
                "TRAIN" => {
                    if parts.len() < 5 {
                        println!("Usage: TRAIN <model> <clients> <rounds> <epochs>");
                        continue;
                    }
                    
                    let model_name = parts[1];
                    
                    let clients = match parts[2].parse::<usize>() {
                        Ok(c) => c,
                        Err(e) => {
                            println!("Invalid client count: {}", e);
                            continue;
                        }
                    };
                    
                    let rounds = match parts[3].parse::<usize>() {
                        Ok(r) => r,
                        Err(e) => {
                            println!("Invalid rounds: {}", e);
                            continue;
                        }
                    };
                    
                    let epochs = match parts[4].parse::<usize>() {
                        Ok(e) => e,
                        Err(e) => {
                            println!("Invalid epochs: {}", e);
                            continue;
                        }
                    };
                    
                    // Verify model exists before starting training
                    {
                        let server_guard = server.lock().await;
                        if !server_guard.models.contains_key(model_name) {
                            println!("Model {} not initialized. Please initialize it first with INIT command.", model_name);
                            continue;
                        }
                    }
                    
                    let server_clone = Arc::clone(&server);
                    let model_name_clone = model_name.to_string();
                    tokio::spawn(async move {
                        let server_instance = Server::new();
                        if let Err(e) = server_instance.train(&model_name_clone, clients, rounds, epochs, server_clone).await {
                            eprintln!("Training error: {}", e);
                        } else {
                            println!("Training completed for model {}", model_name_clone);
                        }
                    });
                    
                    println!("Training started for model {} with {} clients, {} rounds, and {} epochs", model_name, clients, rounds, epochs);
                }
                "GET" => {
                    if parts.len() < 2 {
                        println!("Usage: GET <model>");
                        continue;
                    }
                    
                    let model_name = parts[1];
                    let server_guard = server.lock().await;
                    
                    if let Some((model, _, status)) = server_guard.get_model(model_name) {
                        println!("Model: {}", model_name);
                        println!("Status: {}", status.to_string());
                        println!("Parameters: weights shape {:?}, bias shape {:?}", 
                            model.weight().ok().map(|t| t.shape()),
                            model.bias().ok().map(|t| t.shape()));
                    } else {
                        println!("Model '{}' not found", model_name);
                    }
                }
                "TEST" => {
                    if parts.len() < 2 {
                        println!("Usage: TEST <model>");
                        continue;
                    }
                    
                    let model_name = parts[1];
                    let server_guard = server.lock().await;
                    
                    match server_guard.test(model_name) {
                        Ok(accuracy) => println!("Model {} accuracy: {:.2}%", model_name, accuracy * 100.0),
                        Err(e) => eprintln!("Test error: {}", e),
                    }
                }
                _ => {
                    println!("Invalid command. Available commands: INIT, TRAIN, GET, TEST, exit");
                }
            }
        }
        Ok(())
    }
}

// Helper constants
const LABELS: usize = 10;
const IMAGE_DIM: usize = 784;

#[tokio::main]
async fn main() -> Result<()> {
    let server = Arc::new(Mutex::new(Server::new()));
    let listener = TcpListener::bind("127.0.0.1:50051").await?;
    println!("Parameter Server listening on 127.0.0.1:50051");

    // Handle incoming client connections
    let server_clone = Arc::clone(&server);
    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((stream, addr)) => {
                    println!("New connection from {}", addr);
                    let server_clone_inner = Arc::clone(&server_clone);
                    tokio::spawn(async move {
                        if let Err(e) = Server::handle_client(stream, server_clone_inner).await {
                            eprintln!("Error handling client {}: {}", addr, e);
                        }
                    });
                }
                Err(e) => {
                    eprintln!("Error accepting connection: {}", e);
                    sleep(Duration::from_secs(1)).await;
                }
            }
        }
    });

    // Start command handler
    Server::handle_commands(server).await?;
    Ok(())
}