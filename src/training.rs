use burn::data::dataset::Dataset;
use burn::{
    config::Config,
    data::{
        dataloader::{DataLoader, DataLoaderBuilder},
        dataset::transform::SamplerDataset,
    },
    lr_scheduler::noam::NoamLrSchedulerConfig,
    module::Module,
    optim::{Adam, AdamConfig},
    record::{CompactRecorder, FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::backend::{AutodiffBackend, Backend},
    train::{
        metric::{AccuracyMetric, CudaMetric, LearningRateMetric, LossMetric},
        LearnerBuilder,
    },
};
use std::env;
use std::sync::Arc;

use crate::data::{TextGenerationBatcher, TextGenerationItem};
use crate::model::{TextGenerationModel, TextGenerationModelConfig};
use crate::tokenizer::BpeTokenizer;

pub fn load_pretrained_model<B: Backend>(device: &B::Device) -> TextGenerationModel<B> {
    let current_dir = env::current_dir().expect("Failed to get current directory");
    let model_dir = current_dir.join("data/124M");

    let model_config = TextGenerationModelConfig::from_dir(model_dir.clone());
    let model: TextGenerationModel<B> =
        model_config.init_from_pretrained_weights(&model_dir.join("exploded_model"), device);

    model
}

#[derive(Config)]
pub struct ExperimentConfig {
    pub n_embd: usize,
    pub optimizer: AdamConfig,
    pub batch_size: usize,
    pub block_size: usize,
    pub max_iters: usize,
}

pub fn train<B: AutodiffBackend, D: Dataset<TextGenerationItem> + 'static>(
    device: B::Device,
    dataset_train: D,
    dataset_test: D,
    config: ExperimentConfig,
    tokenizer: BpeTokenizer,
    artifact_dir: &str,
) {
    let tokenizer = Arc::new(tokenizer);
    let batcher_train = TextGenerationBatcher::new(tokenizer.clone(), config.block_size);
    let batcher_test = TextGenerationBatcher::new(tokenizer.clone(), config.block_size);

    let model: TextGenerationModel<B> = load_pretrained_model(&device);

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(SamplerDataset::new(dataset_train, 50_000));

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(SamplerDataset::new(dataset_test, 1000));

    let accum = 1;
    let optim = config.optimizer.init();
    let lr_scheduler = NoamLrSchedulerConfig::new(0.01 / accum as f64)
        .with_warmup_steps(6000)
        .with_model_size(config.n_embd)
        .init();

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train(LossMetric::new())
        .metric_valid(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device])
        .grads_accumulation(accum)
        .num_epochs(config.max_iters)
        .build(model, optim, lr_scheduler);

    let model = learner.fit(dataloader_train, dataloader_test);

    config.save(format!("{artifact_dir}/config.json")).unwrap();

    // Save model in binary format with full precision
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .save_file(format!("{artifact_dir}/model.bin"), &recorder)
        .expect("Should be able to save the model");
}
