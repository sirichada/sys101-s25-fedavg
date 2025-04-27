mod model;
mod protocol;
mod server;
mod client;

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::thread;
use std::time::Duration;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    // Run as server
    Server {
        #[arg(short, long, default_value_t = 8000)]
        port: u16,
        
        #[arg(long, default_value = "mnist")]
        model: String,
        
        #[arg(short, long, default_value_t = 100)]
        epochs: usize,
        
        #[arg(short, long, default_value_t = 5)]
        clients: usize,
    },
    
    // Run as client
    Client {
        #[arg(short, long, default_value_t = 0)]
        port: u16,
        
        #[arg(short, long, default_value = "127.0.0.1:8000")]
        server: String,
        
        #[arg(long, default_value = "mnist")]
        model: String,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Server { port, model, epochs, clients } => {
            run_server(port, &model, epochs, clients)?;
        },
        Commands::Client { port, server, model } => {
            run_client(port, &server, &model)?;
        },
    }
    
    Ok(())
}

fn run_server(port: u16, model_name: &str, epochs: usize, client_count: usize) -> anyhow::Result<()> {
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), port);
    
    let server = server::Server::new(addr).map_err(|e| anyhow::anyhow!("Server initialization error: {}", e))?;
    
    // Initialize model
    server.init_model(model_name).map_err(|e| anyhow::anyhow!("Model initialization error: {}", e))?;
    println!("Server initialized model '{}'", model_name);
    
    // Wait for clients to connect
    println!("Waiting for {} clients to connect...", client_count);
    let mut connected_clients = 0;
    while connected_clients < client_count {
        thread::sleep(Duration::from_secs(1));
        
        // In a complete implementation, we would check the number of registered clients
        // For now, we'll just simulate waiting
        connected_clients += 1;
        println!("Client {} connected", connected_clients);
    }
    
    // Start training
    println!("Starting training for {} epochs", epochs);
    server.train_model(model_name, epochs).map_err(|e| anyhow::anyhow!("Training error: {}", e))?;
    
    // Test the final model
    let accuracy = server.test_model(model_name).map_err(|e| anyhow::anyhow!("Testing error: {}", e))?;
    println!("Final model accuracy: {:.2}%", 100.0 * accuracy);
    
    // Keep server running
    loop {
        thread::sleep(Duration::from_secs(1));
    }
}

fn run_client(port: u16, server_addr: &str, model_name: &str) -> anyhow::Result<()> {
    // Convert server address string to SocketAddr
    let server_addr: SocketAddr = server_addr.parse()
        .map_err(|e| anyhow::anyhow!("Invalid server address: {}", e))?;
    
    // Create a unique client port if 0 was provided
    let client_port = if port == 0 {
        // In a real implementation, we would find an available port
        // For simplicity, we'll just use a random port in the range 8001-9000
        8001 + (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_secs() % 1000) as u16
    } else {
        port
    };
    
    let client_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), client_port);
    
    let client = client::Client::new(client_addr, server_addr)
        .map_err(|e| anyhow::anyhow!("Client initialization error: {}", e))?;
    
    // Join the server for the specified model
    client.join(model_name).map_err(|e| anyhow::anyhow!("Client join error: {}", e))?;
    println!("Joined server for model '{}'", model_name);
    
    // Keep client running to receive training requests
    client.run().map_err(|e| anyhow::anyhow!("Client run error: {}", e))?;
    
    Ok(())
}