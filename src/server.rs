use std::collections::HashMap;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::io::{Read, Write};
use std::time::Duration;

use candle_core::{DType, Device, Result};
use candle_datasets::vision::mnist;
use candle_nn::{VarBuilder, VarMap};
use candle_datasets::vision::Dataset;

use bincode::{deserialize, serialize};

use crate::model::{LinearModel, Model};
use crate::protocol::{ClientInfo, Message, ModelParameters, ModelStatus};

// Global Model container
struct GlobalModel {
    parameters: ModelParameters,
    status: ModelStatus,
    clients: Vec<ClientInfo>,
    received_models: HashMap<usize, ModelParameters>, // client_id -> model params
}

pub struct Server {
    listener: TcpListener,
    addr: SocketAddr,
    models: Arc<Mutex<HashMap<String, GlobalModel>>>,
    clients: Arc<Mutex<Vec<ClientInfo>>>,
    dataset: Option<Dataset>,
    device: Device,
    next_client_id: Arc<Mutex<usize>>,
}

impl Server {
    pub fn new(addr: SocketAddr) -> Result<Self> {
        let listener = TcpListener::bind(addr).map_err(|e| candle_core::Error::Msg(format!("Failed to bind: {}", e)))?;
        listener.set_nonblocking(true).map_err(|e| candle_core::Error::Msg(format!("Failed to set non-blocking: {}", e)))?;
        
        // Load MNIST dataset
        let dataset = mnist::load()?;
        
        println!("Server started at {}", addr);
        
        // Use CPU device for compatibility
        let device = Device::Cpu;
        
        Ok(Self {
            listener,
            addr,
            models: Arc::new(Mutex::new(HashMap::new())),
            clients: Arc::new(Mutex::new(Vec::new())),
            dataset: Some(dataset),
            device,
            next_client_id: Arc::new(Mutex::new(0)),
        })
    }
    
    // Start the server
    pub fn start(&self) -> Result<()> {
        let models = Arc::clone(&self.models);
        let clients = Arc::clone(&self.clients);
        let next_client_id = Arc::clone(&self.next_client_id);
        let listener = self.listener.try_clone().map_err(|e| candle_core::Error::Msg(format!("Failed to clone listener: {}", e)))?;
        
        // Spawn a thread to handle incoming connections
        thread::spawn(move || {
            loop {
                if let Ok((stream, addr)) = listener.accept() {
                    // Handle client connection
                    let models_clone = Arc::clone(&models);
                    let clients_clone = Arc::clone(&clients);
                    let next_id_clone = Arc::clone(&next_client_id);
                    
                    thread::spawn(move || {
                        if let Err(e) = Self::handle_client_connection(stream, addr, models_clone, clients_clone, next_id_clone) {
                            eprintln!("Error handling client: {}", e);
                        }
                    });
                }
                
                thread::sleep(Duration::from_millis(100));
            }
        });
        
        Ok(())
    }
    
    // Handle a client connection
    fn handle_client_connection(
        mut stream: TcpStream,
        addr: SocketAddr,
        models: Arc<Mutex<HashMap<String, GlobalModel>>>,
        clients: Arc<Mutex<Vec<ClientInfo>>>,
        next_client_id: Arc<Mutex<usize>>,
    ) -> Result<()> {
        let mut buffer = Vec::new();
        stream.read_to_end(&mut buffer).map_err(|e| candle_core::Error::Msg(format!("Failed to read from client: {}", e)))?;
        
        let message: Message = deserialize(&buffer).map_err(|e| candle_core::Error::Msg(format!("Failed to deserialize message: {}", e)))?;
        
        // Process message based on type
        // Note: In a real implementation, we would handle each message type
        // For now, just log the message type
        println!("Received message from {}: {:?}", addr, message);
        
        Ok(())
    }
    
    // Accept and handle client connections (stub for original function)
    fn accept_client(
        _models: &Arc<Mutex<HashMap<String, GlobalModel>>>,
        _clients: &Arc<Mutex<Vec<ClientInfo>>>,
        _next_client_id: &Arc<Mutex<usize>>,
    ) -> Result<()> {
        // This function will be called in a loop to accept new clients
        // Now handled by handle_client_connection
        Ok(())
    }
    
    // Initialize a model
    pub fn init_model(&self, model_name: &str) -> Result<()> {
        let mut models = self.models.lock().unwrap();
        
        // Initialize a new linear model
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &self.device);
        let model = LinearModel::new(vs)?;
        
        // Convert to parameters
        let parameters = model.to_parameters()?;
        
        models.insert(model_name.to_string(), GlobalModel {
            parameters,
            status: ModelStatus::Initialized,
            clients: Vec::new(),
            received_models: HashMap::new(),
        });
        
        println!("Model '{}' initialized", model_name);
        Ok(())
    }
    
    // Register a client
    pub fn register_client(&self, client_addr: SocketAddr, model_name: &str) -> Result<usize> {
        let mut models = self.models.lock().unwrap();
        let mut clients = self.clients.lock().unwrap();
        let mut next_id = self.next_client_id.lock().unwrap();
        
        // Check if model exists
        if !models.contains_key(model_name) {
            return Err(candle_core::Error::Msg(format!("Model '{}' not found", model_name)));
        }
        
        let client_id = *next_id;
        *next_id += 1;
        
        // Create client info
        let client_info = ClientInfo {
            addr: client_addr,
            model_name: model_name.to_string(),
            id: client_id,
        };
        
        // Add client to global list and model's client list
        clients.push(client_info.clone());
        if let Some(global_model) = models.get_mut(model_name) {
            global_model.clients.push(client_info);
        }
        
        println!("Client {} registered for model '{}'", client_id, model_name);
        Ok(client_id)
    }
    
    // Start training rounds
    pub fn train_model(&self, model_name: &str, rounds: usize) -> Result<()> {
        for round in 1..=rounds {
            println!("Starting training round {} for model '{}'", round, model_name);
            self.training_round(model_name, round)?;
        }
        Ok(())
    }
    
    // Single training round
    fn training_round(&self, model_name: &str, round: usize) -> Result<()> {
        let mut models = self.models.lock().unwrap();
        
        // Get the global model
        let global_model = models.get_mut(model_name)
            .ok_or_else(|| candle_core::Error::Msg(format!("Model '{}' not found", model_name)))?;
        
        // Update status
        global_model.status = ModelStatus::Training;
        global_model.received_models.clear();
        
        // Get client list
        let clients = global_model.clients.clone();
        if clients.is_empty() {
            return Err(candle_core::Error::Msg(format!("No clients registered for model '{}'", model_name)));
        }
        
        // Distribute model to clients for training
        let client_count = clients.len();
        let dataset = self.dataset.as_ref().unwrap();
        let train_samples = dataset.train_images.dims()[0];
        
        // Create data partitions (for simplicity, we'll split evenly)
        let samples_per_client = train_samples / client_count;
        
        // Drop the lock before communicating with clients
        drop(models);
        
        for (i, client) in clients.iter().enumerate() {
            // Calculate data range for this client
            let start_idx = i * samples_per_client;
            let end_idx = if i == client_count - 1 {
                train_samples
            } else {
                (i + 1) * samples_per_client
            };
            
            let indices: Vec<usize> = (start_idx..end_idx).collect();
            
            // Get the current model parameters
            let parameters = {
                let models = self.models.lock().unwrap();
                let global_model = models.get(model_name).unwrap();
                global_model.parameters.clone()
            };
            
            // Send the model to the client
            self.send_training_request(client, parameters, indices)?;
        }
        
        // Wait for all clients to finish training
        let mut received_all = false;
        while !received_all {
            thread::sleep(Duration::from_millis(500));
            
            let models = self.models.lock().unwrap();
            let global_model = models.get(model_name).unwrap();
            
            received_all = global_model.received_models.len() == client_count;
            
            // Optional timeout could be added here
        }
        
        // Aggregate models (Federated Averaging)
        self.aggregate_models(model_name)?;
        
        println!("Training round {} completed for model '{}'", round, model_name);
        Ok(())
    }
    
    // Send training request to a client
    fn send_training_request(
        &self, 
        client: &ClientInfo, 
        parameters: ModelParameters,
        data_indices: Vec<usize>,
    ) -> Result<()> {
        let message = Message::StartTraining {
            model_name: client.model_name.clone(),
            parameters,
            data_indices,
        };
        
        self.send_message_to_client(client.addr, message)?;
        Ok(())
    }
    
    // Helper to send a message to a client
    fn send_message_to_client(&self, addr: SocketAddr, message: Message) -> Result<()> {
        let stream = TcpStream::connect(addr)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to connect to client: {}", e)))?;
            
        let serialized = serialize(&message)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to serialize message: {}", e)))?;
            
        let mut stream = stream;
        stream.write_all(&serialized)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to send message: {}", e)))?;
            
        Ok(())
    }
    
    // Aggregate models using Federated Averaging
    fn aggregate_models(&self, model_name: &str) -> Result<()> {
        let mut models = self.models.lock().unwrap();
        
        // Get the global model
        let global_model = models.get_mut(model_name)
            .ok_or_else(|| candle_core::Error::Msg(format!("Model '{}' not found", model_name)))?;
            
        // Get all the received models
        let received_models: Vec<&ModelParameters> = global_model.received_models.values().collect();
        
        if received_models.is_empty() {
            return Err(candle_core::Error::Msg("No models received for aggregation".to_string()));
        }
        
        // Initialize aggregated model parameters
        let model_count = received_models.len() as f32;
        let first_model = &received_models[0];
        let weights_len = first_model.weights.len();
        let biases_len = first_model.biases.len();
        
        let mut aggregated_weights = vec![0.0; weights_len];
        let mut aggregated_biases = vec![0.0; biases_len];
        
        // Sum all parameters
        for model_params in received_models {
            for (i, &weight) in model_params.weights.iter().enumerate() {
                aggregated_weights[i] += weight / model_count;
            }
            
            for (i, &bias) in model_params.biases.iter().enumerate() {
                aggregated_biases[i] += bias / model_count;
            }
        }
        
        // Update global model
        global_model.parameters.weights = aggregated_weights;
        global_model.parameters.biases = aggregated_biases;
        global_model.status = ModelStatus::Ready;
        
        println!("Model '{}' aggregated from {} clients", model_name, model_count);
        Ok(())
    }
    
    // Get model parameters
    pub fn get_model(&self, model_name: &str) -> Result<Option<ModelParameters>> {
        let models = self.models.lock().unwrap();
        
        Ok(models.get(model_name).map(|global_model| global_model.parameters.clone()))
    }
    
    // Test the model accuracy
    pub fn test_model(&self, model_name: &str) -> Result<f32> {
        let models = self.models.lock().unwrap();
        
        // Get the global model
        let global_model = models.get(model_name)
            .ok_or_else(|| candle_core::Error::Msg(format!("Model '{}' not found", model_name)))?;
            
        // Convert parameters to model
        let model = LinearModel::from_parameters(&global_model.parameters, &self.device)?;
        
        // Test on the MNIST test set
        let dataset = self.dataset.as_ref().unwrap();
        let test_images = dataset.test_images.to_device(&self.device)?;
        let test_labels = dataset.test_labels.to_dtype(DType::U32)?.to_device(&self.device)?;
        
        // Forward pass
        let logits = model.forward(&test_images)?;
        let sum_ok = logits
            .argmax(candle_core::D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
            
        let accuracy = sum_ok / test_labels.dims1()? as f32;
        
        println!("Model '{}' test accuracy: {:.2}%", model_name, 100.0 * accuracy);
        Ok(accuracy)
    }
    
    // Handle client messages
    fn handle_client_message(&self, message: Message) -> Result<Message> {
        match message {
            Message::RegisterClient { client_addr, model_name } => {
                let client_id = self.register_client(client_addr, &model_name)?;
                Ok(Message::RegisterResponse {
                    client_id,
                    success: true,
                    message: format!("Registered for model '{}'", model_name),
                })
            },
            
            Message::TrainingCompleted { client_id, model_name, parameters } => {
                // Store the received model
                let mut models = self.models.lock().unwrap();
                if let Some(global_model) = models.get_mut(&model_name) {
                    global_model.received_models.insert(client_id, parameters);
                    println!("Received trained model from client {}", client_id);
                }
                
                Ok(Message::RegisterResponse {
                    client_id,
                    success: true,
                    message: "Training results received".to_string(),
                })
            },
            
            Message::GetModelRequest { model_name } => {
                let parameters = self.get_model(&model_name)?;
                let status = if let Some(params) = &parameters {
                    params.status
                } else {
                    ModelStatus::Initialized
                };
                
                Ok(Message::GetModelResponse {
                    model_name,
                    parameters,
                    status,
                })
            },
            
            Message::TestModelRequest { model_name } => {
                let accuracy = self.test_model(&model_name)?;
                
                Ok(Message::TestModelResponse {
                    model_name,
                    accuracy,
                })
            },
            
            _ => Err(candle_core::Error::Msg("Unsupported message type".to_string())),
        }
    }
    
    // Main server loop to handle incoming connections
    pub fn run(&self) -> Result<()> {
        self.start()?;
        
        // This would be the main loop that keeps the server running
        loop {
            thread::sleep(Duration::from_secs(1));
            // In a real implementation, we might have additional controls here
        }
    }
}