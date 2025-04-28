use clap::{Parser, ValueEnum};

use candle_core::{DType, Result, Tensor, D, IndexOp};
use candle_nn::{loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap, SGD};
use candle_app::{LinearModel, Model, IMAGE_DIM, LABELS};

trait SequentialModel: Sized {
    fn new(vs: VarBuilder) -> Result<Self>;
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
    fn weight(&self) -> Result<&Tensor>;
    fn bias(&self) -> Result<&Tensor>;
}

impl<T: Model> SequentialModel for T {
    fn new(vs: VarBuilder) -> Result<Self> {
        Model::new(vs)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Model::forward(self, xs)
    }

    fn weight(&self) -> Result<&Tensor> {
        Model::weight(self)
    }

    fn bias(&self) -> Result<&Tensor> {
        Model::bias(self)
    }
}

struct TrainingArgs {
    learning_rate: f64,
    epochs: usize,
}

fn model_train<M: SequentialModel>(
    model: M,
    test_images: &Tensor,
    test_labels: &Tensor,
    train_images: &Tensor,
    train_labels: &Tensor,
    mut sgd: SGD,
    epochs: usize
) -> Result<M> {
    for epoch in 1..=epochs {
        let logits = model.forward(&train_images)?;
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &train_labels)?;
        sgd.backward_step(&loss)?;

        let test_logits = model.forward(&test_images)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!(
            "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
            loss.to_scalar::<f32>()?,
            100. * test_accuracy
        );
    }
    Ok(model)
}

// This simulates federated learning in a sequential manner
fn training_loop<M: SequentialModel>(
    m: candle_datasets::vision::Dataset,
    args: &TrainingArgs,
) -> anyhow::Result<()> {
    // Use CPU device since we've disabled CUDA
    let dev = candle_core::Device::Cpu;

    let train_labels = m.train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    let train_images = m.train_images.to_device(&dev)?;
    let test_images = m.test_images.to_device(&dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    // Train 1st model (simulating first client)
    let varmap1 = VarMap::new();
    let vs1 = VarBuilder::from_varmap(&varmap1, DType::F32, &dev);
    let half_test = test_images.shape().dims()[0]/2;
    let half_train = train_images.shape().dims()[0]/2;
    
    println!("Training first client model");
    let model_1 = model_train(
        M::new(vs1.clone())?, 
        &test_images.i(..half_test)?,
        &test_labels.i(..half_test)?,
        &train_images.i(..half_train)?,
        &train_labels.i(..half_train)?,
        SGD::new(varmap1.all_vars(), args.learning_rate)?, 
        args.epochs
    )?;

    // Train 2nd model (simulating second client)
    println!("\nTraining second client model");
    let varmap2 = VarMap::new();
    let vs2 = VarBuilder::from_varmap(&varmap2, DType::F32, &dev);
    let model_2 = model_train(
        M::new(vs2.clone())?,
        &test_images.i(half_test..)?,
        &test_labels.i(half_test..)?,
        &train_images.i(half_train..)?,
        &train_labels.i(half_train..)?,
        SGD::new(varmap2.all_vars(), args.learning_rate)?, 
        args.epochs
    )?;

    // Simulate federated averaging by creating an average model
    println!("\nPerforming federated averaging");
    
    // Properly handle Results when calculating average weights and biases
    let w1 = model_1.weight()?;
    let w2 = model_2.weight()?;
    let b1 = model_1.bias()?;
    let b2 = model_2.bias()?;
    
    // Add tensors and unwrap the result before dividing
    let sum_weights = (w1 + w2)?;
    let sum_bias = (b1 + b2)?;
    
    // Now divide by 2.0 and unwrap those results too
    let avg_weights = (&sum_weights / 2.0)?;
    let avg_bias = (&sum_bias / 2.0)?;
    
    let avg_model = Linear::new(avg_weights, Some(avg_bias));
    
    // Test the averaged model (global model)
    let test_logits = avg_model.forward(&test_images)?;
    let sum_ok = test_logits
        .argmax(D::Minus1)?
        .eq(&test_labels)?
        .to_dtype(DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;
    let test_accuracy = sum_ok / test_labels.dims1()? as f32;
    println!("Global model (federated average) test accuracy: {:5.2}%", 100. * test_accuracy);

    Ok(())
}

#[derive(ValueEnum, Clone)]
enum WhichModel {
    Linear,
}

#[derive(Parser)]
struct Args {
    #[clap(value_enum, default_value_t = WhichModel::Linear)]
    model: WhichModel,

    #[arg(long)]
    learning_rate: Option<f64>,

    #[arg(long, default_value_t = 10)]
    epochs: usize,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    
    // Load the dataset
    let m = candle_datasets::vision::mnist::load()?;

    println!("Train images: {:?}", m.train_images.shape());
    println!("Train labels: {:?}", m.train_labels.shape());
    println!("Test images: {:?}", m.test_images.shape());
    println!("Test labels: {:?}", m.test_labels.shape());
    println!();

    // Set up training arguments
    let default_learning_rate = match args.model {
        WhichModel::Linear => 1.,
    };
    
    let training_args = TrainingArgs {
        epochs: args.epochs,
        learning_rate: args.learning_rate.unwrap_or(default_learning_rate),
    };
    
    // Run sequential simulation of federated learning
    match args.model {
        WhichModel::Linear => {
            println!("Running sequential simulation of federated learning:");
            training_loop::<LinearModel>(m, &training_args)?;
        }
    }
    
    println!("\nNote: This is a non-distributed sequential simulation.");
    println!("For distributed federated learning, run the server and client components.");
    
    Ok(())
}