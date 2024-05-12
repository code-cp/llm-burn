use burn::backend::wgpu::WgpuDevice;
use burn::backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu};
use burn::optim::decay::WeightDecayConfig;
use std::{fs, io::prelude::*, path::Path};
// use burn::backend::{libtorch::LibTorchDevice, LibTorch};

use std::env;

use llm_burn::data::CustomDataset;
use llm_burn::inference::infer;
use llm_burn::tokenizer::BpeTokenizer;
use llm_burn::training::{train, ExperimentConfig};

type Backend = burn::backend::Autodiff<burn::backend::Wgpu>;
// type Backend = burn::backend::Autodiff<burn::backend::LibTorch<Elem>>;

struct TrainConfig {
    batch_size: usize,
    block_size: usize,
    max_iters: usize,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        panic!("Usage: [mode]", &args[0]);
    }
    let mode: String = String::from(&args[1]);

    let tokenizer = BpeTokenizer::from_dir(dataset_dir.clone()).unwrap();
    let device = WgpuDevice::default();

    if mode == "infer" {
        infer(tokenizer, device);
        return;
    };

    let train_config = match mode.as_str() {
        "train_small" => TrainConfig {
            batch_size: 1,
            block_size: 2,
            max_iters: 1,
        },
        "train_full" => TrainConfig {
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

    let current_dir = env::current_dir().expect("Failed to get current directory");
    let dataset_dir = current_dir.join("data/dataset.txt");
    let artifact_dir = "./artifacts";

    let data_dir = dataset_dir.to_str().unwrap();

    train::<Backend, CustomDataset>(
        device,
        // LibTorchDevice::Cuda(0),
        // if cfg!(target_os = "macos") {
        //     burn::tensor::Device::<Backend>::Mps
        // } else {
        //     burn::tensor::Device::<Backend>::Cuda(0)
        // },
        CustomDataset::train(data_dir, config.batch_size),
        CustomDataset::test(data_dir, config.batch_size),
        config,
        tokenizer,
        artifact_dir,
    );
}
