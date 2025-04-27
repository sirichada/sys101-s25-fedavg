use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

// Model status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ModelStatus {
    Initialized,
    Training,
    Ready,
}

// Model parameters (weights and biases for our linear model)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameters {
    pub weights: Vec<f32>,    // Flattened weights
    pub biases: Vec<f32>,     // Biases
    pub input_dim: usize,     // Input dimension
    pub output_dim: usize,    // Output dimension
    pub status: ModelStatus,  // Model status
}

// Client information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    pub addr: SocketAddr,
    pub model_name: String,
    pub id: usize,
}

// Messages sent between server and clients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    // Client -> Server messages
    RegisterClient {
        client_addr: SocketAddr,
        model_name: String,
    },
    TrainingCompleted {
        client_id: usize,
        model_name: String,
        parameters: ModelParameters,
    },
    GetModelRequest {
        model_name: String,
    },
    TestModelRequest {
        model_name: String,
    },
    
    // Server -> Client messages
    RegisterResponse {
        client_id: usize,
        success: bool,
        message: String,
    },
    StartTraining {
        model_name: String,
        parameters: ModelParameters,
        data_indices: Vec<usize>, // Indices of data to use for training
    },
    GetModelResponse {
        model_name: String,
        parameters: Option<ModelParameters>,
        status: ModelStatus,
    },
    TestModelResponse {
        model_name: String,
        accuracy: f32,
    },
}