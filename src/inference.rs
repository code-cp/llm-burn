use burn::backend::wgpu::WgpuDevice;
use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    record::{
        BinFileRecorder, CompactRecorder, FullPrecisionSettings, NamedMpkFileRecorder, Recorder,
    },
    tensor::backend::AutodiffBackend,
    tensor::{activation, backend::Backend, Bool, Device, ElementConversion, Int, Tensor},
};
use std::env;

use crate::model::{TextGenerationModel, TextGenerationModelConfig};
use crate::tokenizer::{BpeTokenizer, Tokenizer};

pub fn load_finetuned_model<B: Backend>(device: &B::Device) -> TextGenerationModel<B> {
    let current_dir = env::current_dir().expect("Failed to get current directory");
    let model_dir = current_dir.join("data/124M");
    let model_config = TextGenerationModelConfig::from_dir(model_dir.clone());

    // Load model in full precision from MessagePack file
    let model_path = current_dir.join("artifacts/model.bin");
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model: TextGenerationModel<B> = model_config
        .init(device)
        .load_file(model_path, &recorder, device)
        .expect("Should be able to load the model weights from the provided file");

    model
}

pub fn load_pretrained_model() {}

pub fn infer<B: AutodiffBackend>(
    tokenizer: impl Tokenizer,
    model: TextGenerationModel<B>,
    prompt: &str,
    num_tokens: usize,
) {
    let token_ids = tokenizer.encode(&prompt).unwrap();
    // Load that record with the model
    let output_ids = model.infer(token_ids, num_tokens);
    let decoded = tokenizer.decode(&output_ids);

    println!("prompt = {prompt:?}");
    println!("output = {decoded:?}");
}
