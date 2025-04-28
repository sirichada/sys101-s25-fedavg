use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

pub const IMAGE_DIM: usize = 784;
pub const LABELS: usize = 10;

pub trait Model: Sized {
    fn new(vs: VarBuilder) -> Result<Self>;
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
    fn weight(&self) -> Result<&Tensor>;
    fn bias(&self) -> Result<&Tensor>;
}

pub struct LinearModel {
    pub linear: Linear,
}

impl LinearModel {
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let vs = vs.pp("linear");
        let linear = Linear::new(
            vs.get_with_hints((LABELS, IMAGE_DIM), "weight", candle_nn::init::DEFAULT_KAIMING_NORMAL)?,
            Some(vs.get_with_hints(LABELS, "bias", candle_nn::init::ZERO)?),
        );
        Ok(Self { linear })
    }
}

impl Module for LinearModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.linear.forward(xs)
    }
}

impl Model for LinearModel {
    fn new(vs: VarBuilder) -> Result<Self> {
        Self::new(vs)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.linear.forward(xs)
    }

    fn weight(&self) -> Result<&Tensor> {
        Ok(self.linear.weight())
    }

    fn bias(&self) -> Result<&Tensor> {
        Ok(self.linear.bias().unwrap())
    }
}

// Model status for communication between client and server
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelStatus {
    Initialized,
    Training,
    Ready,
}

impl ToString for ModelStatus {
    fn to_string(&self) -> String {
        match self {
            ModelStatus::Initialized => "initialized".to_string(),
            ModelStatus::Training => "training".to_string(),
            ModelStatus::Ready => "ready".to_string(),
        }
    }
}

impl From<&str> for ModelStatus {
    fn from(s: &str) -> Self {
        match s {
            "initialized" => ModelStatus::Initialized,
            "training" => ModelStatus::Training,
            "ready" => ModelStatus::Ready,
            _ => ModelStatus::Initialized,
        }
    }
}