use ndarray::Array;

pub type NdArray = Array<f32, ndarray::IxDyn>;

pub trait MNISTModel {
    fn new() -> Self;
    fn train(&self);
    fn validate(&self) -> f64;
    fn load_model(&mut self) -> Result<(), std::io::Error>;
}
