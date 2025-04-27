use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::io::{Read, Write};
use std::time::Duration;
use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor, IndexOp};
use candle_datasets::vision::mnist;
use candle_datasets::vision::Dataset;

use bincode::{deserialize, serialize};

use crate::model::{LinearModel, Model, train_model, test_model};
use crate::protocol::{Message, ModelParameters, ModelStatus};

// Local model container
struct LocalModel {
    parameters: ModelParameters,
    status: ModelStatus,
}

pub struct Client {
    listener: TcpListener,
    server_addr: SocketAddr,
    client_addr: SocketAddr,
    client_id: Arc<Mutex<Option<usize>>>,
    models: Arc<Mutex<HashMap<String, LocalModel>>>,
    dataset: Option<Dataset>,
    device: Device,
}

impl Client {
    pub fn new(client_addr: SocketAddr, server_addr: SocketAddr) -> Result<Self> {
        let listener = TcpListener::bind(client_addr)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to bind client: {}", e)))?;
        listener.set_nonblocking(true)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to set non-blocking: {}", e)))?;
            
        // Load MNIST dataset
        let dataset = mnist::load()?;
        
        println!("Client started at {}, server at {}", client_addr, server_addr);
        
        // Use CPU device for compatibility
        let device = Device::Cpu;
        
        Ok(Self {
            listener,
            server_addr,
            client_addr,
            client_id: Arc::new(Mutex::new(None)),
            models: Arc::new(Mutex::new(HashMap::new())),
            dataset: Some(dataset),
            device,
        })
    }
    
    // Start the client
    pub fn start(&self) -> Result<()> {
        let models = Arc::clone(&self.models);
        let client_id = Arc::clone(&self.client_id);
        
        // Spawn a thread to handle incoming messages from server
        let listener = self.listener.try_clone()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to clone listener: {}", e)))?;
            
        thread::spawn(move || {
            loop {
                // Accept incoming connections (from server)
                if let Ok((mut stream, _)) = listener.accept() {
                    let models = Arc::clone(&models);
                    let client_id = Arc::clone(&client_id);
                    
                    // Handle this connection in a new thread
                    thread::spawn(move || {
                        // Read the message
                        let mut buffer = Vec::new();
                        if let Err(e) = stream.read_to_end(&mut buffer) {
                            eprintln!("Error reading from server: {}", e);
                            return;
                        }
                        
                        // Deserialize the message
                        let message: Message = match deserialize(&buffer) {
                            Ok(msg) => msg,
                            Err(e) => {
                                eprintln!("Error deserializing message: {}", e);
                                return;
                            }
                        };
                        
                        // Handle the message
                        match Self::handle_server_message(message, models, client_id) {
                            Ok(_) => {},
                            Err(e) => eprintln!("Error handling server message: {}", e),
                        }
                    });
                }
                
                thread::sleep(Duration::from_millis(100));
            }
        });
        
        Ok(())
    }
    
    // Handle messages from the server
    fn handle_server_message(
        message: Message,
        models: Arc<Mutex<HashMap<String, LocalModel>>>,
        client_id: Arc<Mutex<Option<usize>>>,
    ) -> Result<()> {
        match message {
            Message::RegisterResponse { client_id: id, success, message: msg } => {
                if success {
                    let mut client_id_guard = client_id.lock().unwrap();
                    *client_id_guard = Some(id);
                    println!("Successfully registered with server. Client ID: {}. {}", id, msg);
                } else {
                    eprintln!("Failed to register with server: {}", msg);
                }
            },
            
            Message::StartTraining { model_name, parameters, data_indices } => {
                println!("Received training request for model '{}' with {} samples", model_name, data_indices.len());
                
                // Store the model parameters
                let mut models_guard = models.lock().unwrap();
                models_guard.insert(model_name.clone(), LocalModel {
                    parameters: parameters.clone(),
                    status: ModelStatus::Training,
                });
                drop(models_guard);
                
                // Get client ID
                let client_id_guard = client_id.lock().unwrap();
                let client_id = client_id_guard.unwrap();
                drop(client_id_guard);
                
                // Train the model in a separate thread
                let models_clone = Arc::clone(&models);
                let model_name_clone = model_name.clone();
                
                thread::spawn(move || {
                    // Create a device
                    match Device::Cpu {
                        device => {
                            match Self::perform_local_training(
                                &model_name_clone, 
                                parameters, 
                                data_indices, 
                                client_id, 
                                models_clone,
                                &device
                            ) {
                                Ok(_) => println!("Training completed and results sent to server"),
                                Err(e) => eprintln!("Error during training: {}", e),
                            }
                        }
                    }
                });
            },
            
            Message::GetModelResponse { model_name, parameters, status } => {
                if let Some(params) = parameters {
                    let mut models_guard = models.lock().unwrap();
                    models_guard.insert(model_name.clone(), LocalModel {
                        parameters: params,
                        status,
                    });
                    
                    println!("Received model '{}' from server, status: {:?}", model_name, status);
                } else {
                    println!("Model '{}' not found on server", model_name);
                }
            },
            
            Message::TestModelResponse { model_name, accuracy } => {
                println!("Test results for model '{}': accuracy = {:.2}%", model_name, 100.0 * accuracy);
            },
            
            _ => eprintln!("Received unsupported message type from server"),
        }
        
        Ok(())
    }
    
    // Perform local training
    fn perform_local_training(
        model_name: &str,
        parameters: ModelParameters,
        data_indices: Vec<usize>,
        client_id: usize,
        models: Arc<Mutex<HashMap<String, LocalModel>>>,
        device: &Device,
    ) -> Result<()> {
        // Convert parameters to model
        let mut model = LinearModel::from_parameters(&parameters, device)?;
        
        // Load dataset
        let dataset = mnist::load()?;
        
        // Prepare training and test data
        let train_images = dataset.train_images.to_device(device)?;
        let train_labels = dataset.train_labels.to_dtype(DType::U32)?.to_device(device)?;
        let test_images = dataset.test_images.to_device(device)?;
        let test_labels = dataset.test_labels.to_dtype(DType::U32)?.to_device(device)?;
        
        // Extract the specific data indices for this client
        let mut client_train_images = Vec::new();
        let mut client_train_labels = Vec::new();
        
        for &idx in &data_indices {
            client_train_images.push(train_images.i(idx)?);
            client_train_labels.push(train_labels.i(idx)?);
        }
        
        // Stack the tensors
        let client_train_images = Tensor::cat(&client_train_images, 0)?;
        let client_train_labels = Tensor::cat(&client_train_labels, 0)?;
        
        // Train the model
        println!("Client {} starting local training on {} samples", client_id, data_indices.len());
        train_model(
            &mut model,
            &client_train_images,
            &client_train_labels,
            &test_images,
            &test_labels,
            1.0, // Learning rate
            5,   // Local epochs
        )?;
        
        // Extract parameters
        let trained_parameters = model.to_parameters()?;
        
        // Update local model
        {
            let mut models_guard = models.lock().unwrap();
            models_guard.insert(model_name.to_string(), LocalModel {
                parameters: trained_parameters.clone(),
                status: ModelStatus::Ready,
            });
        }
        
        // Send trained model back to server
        println!("Client {} sending trained model to server", client_id);
        let server_addr = SocketAddr::from(([127, 0, 0, 1], 8000)); // Default server address
        
        let message = Message::TrainingCompleted {
            client_id,
            model_name: model_name.to_string(),
            parameters: trained_parameters,
        };
        
        // Connect to server and send the message
        let mut stream = match TcpStream::connect(server_addr) {
            Ok(stream) => stream,
            Err(e) => {
                return Err(candle_core::Error::Msg(format!("Failed to connect to server: {}", e)));
            }
        };
        
        let serialized = match serialize(&message) {
            Ok(data) => data,
            Err(e) => {
                return Err(candle_core::Error::Msg(format!("Failed to serialize message: {}", e)));
            }
        };
        
        if let Err(e) = stream.write_all(&serialized) {
            return Err(candle_core::Error::Msg(format!("Failed to send message to server: {}", e)));
        }
        
        println!("Client {} completed training and sent results to server", client_id);
        Ok(())
    }
    
    // Join the server for a model
    pub fn join(&self, model_name: &str) -> Result<()> {
        let message = Message::RegisterClient {
            client_addr: self.client_addr,
            model_name: model_name.to_string(),
        };
        
        self.send_message_to_server(message)?;
        println!("Sent registration request to server for model '{}'", model_name);
        
        // In a real implementation, we would wait for the response
        thread::sleep(Duration::from_secs(1));
        
        Ok(())
    }
    
    // Get model parameters
    pub fn get_model(&self, model_name: &str) -> Result<()> {
        let message = Message::GetModelRequest {
            model_name: model_name.to_string(),
        };
        
        self.send_message_to_server(message)?;
        println!("Requested model '{}' from server", model_name);
        
        Ok(())
    }
    
    // Get local model parameters
    pub fn get_local_model(&self, model_name: &str) -> Result<Option<ModelParameters>> {
        let models = self.models.lock().unwrap();
        
        Ok(models.get(model_name).map(|local_model| local_model.parameters.clone()))
    }
    
    // Test the model
    pub fn test_model(&self, model_name: &str) -> Result<()> {
        // First check if we have the model locally
        let local_model = {
            let models = self.models.lock().unwrap();
            models.get(model_name).map(|m| m.parameters.clone())
        };
        
        if let Some(parameters) = local_model {
            // Convert parameters to model
            let model = LinearModel::from_parameters(&parameters, &self.device)?;
            
            // Test on the MNIST test set
            let dataset = self.dataset.as_ref().unwrap();
            let test_images = dataset.test_images.to_device(&self.device)?;
            let test_labels = dataset.test_labels.to_dtype(DType::U32)?.to_device(&self.device)?;
            
            let accuracy = test_model(&model, &test_images, &test_labels)?;
            
            println!("Local model '{}' test accuracy: {:.2}%", model_name, 100.0 * accuracy);
        } else {
            // Request test from server
            let message = Message::TestModelRequest {
                model_name: model_name.to_string(),
            };
            
            self.send_message_to_server(message)?;
            println!("Requested test for model '{}' from server", model_name);
        }
        
        Ok(())
    }
    
    // Helper to send a message to the server
    fn send_message_to_server(&self, message: Message) -> Result<()> {
        let mut stream = TcpStream::connect(self.server_addr)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to connect to server: {}", e)))?;
            
        let serialized = serialize(&message)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to serialize message: {}", e)))?;
            
        stream.write_all(&serialized)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to send message to server: {}", e)))?;
            
        Ok(())
    }
    
    // Run the client
    pub fn run(&self) -> Result<()> {
        self.start()?;
        
        // Keep the client running
        loop {
            thread::sleep(Duration::from_secs(1));
            // Additional controls could be added here
        }
    }
}