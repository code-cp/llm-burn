use burn::backend::wgpu::WgpuDevice;
use burn::backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu};
use burn::optim::decay::WeightDecayConfig;
use std::{fs, io::prelude::*, path::Path};
// use burn::backend::{libtorch::LibTorchDevice, LibTorch};

use std::env;

use llm_burn::data::CustomDataset;
use llm_burn::inference::{infer, load_finetuned_model};
use llm_burn::tokenizer::BpeTokenizer;
use llm_burn::training::{load_pretrained_model, train, ExperimentConfig};

type Backend = burn::backend::Autodiff<burn::backend::Wgpu>;
// type Backend = burn::backend::Autodiff<burn::backend::LibTorch<Elem>>;

struct TrainConfig {
    batch_size: usize,
    block_size: usize,
    max_iters: usize,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mode = if args.len() < 2 {
        "train_small".to_string()
    } else {
        String::from(&args[1])
    };

    let current_dir = env::current_dir().expect("Failed to get current directory");
    let dataset_dir = current_dir.join("data/124M");

    let tokenizer = BpeTokenizer::from_dir(dataset_dir.clone()).unwrap();
    let device = WgpuDevice::default();

    // let prompt = "Elementray";
    // let num_tokens = 20;
    let prompt = "Alan Turing theorized that computers would one day become";
    let num_tokens = 8;

    if mode == "infer_from_finetuned_weights" {
        let model = load_finetuned_model(&device);
        infer::<Backend>(tokenizer, model, prompt, num_tokens);
        return;
    } else if mode == "infer_from_pretrained_weights" {
        let model = load_pretrained_model(&device);
        infer::<Backend>(tokenizer, model, prompt, num_tokens);
        return;
    };

    let train_config = match mode.as_str() {
        "train_small" => TrainConfig {
            batch_size: 2,
            block_size: 5,
            max_iters: 10,
        },
        "train_full" => TrainConfig {
            batch_size: 64,
            block_size: 100,
            max_iters: 50,
        },
        _ => TrainConfig {
            batch_size: 64,
            block_size: 100,
            max_iters: 50,
        },
    };

    let config = ExperimentConfig::new(
        768,
        burn::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-6))),
        train_config.batch_size,
        train_config.block_size,
        train_config.max_iters,
    );

    let dataset_dir = current_dir.join("data/dataset.txt");
    let data_dir = dataset_dir.to_str().unwrap();
    let artifact_dir = "./artifacts";

    train::<Backend, CustomDataset>(
        device,
        CustomDataset::train(data_dir, config.batch_size),
        CustomDataset::test(data_dir, config.batch_size),
        config,
        tokenizer,
        artifact_dir,
    );
}
