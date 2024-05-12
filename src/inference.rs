use burn::backend::wgpu::WgpuDevice;
use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    record::{BinBytesRecorder, BinFileRecorder, CompactRecorder, FullPrecisionSettings, Recorder},
    tensor::backend::AutodiffBackend,
    tensor::{activation, Bool, Device, ElementConversion, Int, Tensor},
};
use std::env;

use crate::model::{TextGenerationModel, TextGenerationModelConfig};
use crate::tokenizer::{BpeTokenizer, Tokenizer};

pub fn infer<B: AutodiffBackend>(tokenizer: impl Tokenizer, device: B::Device) {
    // Include the model file as a reference to a byte array
    static MODEL_BYTES: &[u8] = include_bytes!("../artifacts/model.bin");

    // Load model binary record in full precision
    let record = BinBytesRecorder::<FullPrecisionSettings>::default()
        .load(MODEL_BYTES.to_vec(), &device)
        .expect("Should be able to load model the model weights from bytes");

    let current_dir = env::current_dir().expect("Failed to get current directory");
    let model_dir = current_dir.join("data/124M");

    let model_config = TextGenerationModelConfig::from_dir(model_dir.clone());
    let model: TextGenerationModel<B> =
        model_config.init_model_from_dir(model_dir.join("exploded_model"), &device);

    // let prompt = "Elementray";
    // let num_tokens = 20;
    let prompt = "Alan Turing theorized that computers would one day become";
    let num_tokens = 8;

    let token_ids = tokenizer.encode(&prompt).unwrap();

    // Load that record with the model
    let output_ids = model.load_record(record).infer(token_ids, num_tokens);
    let decoded = tokenizer.decode(&output_ids);

    println!("prompt = {prompt:?}");
    println!("output = {decoded:?}");
}
