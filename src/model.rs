use anyhow::{bail, Context, Result};
use burn::{
    config::Config,
    module::{Module, Param},
    nn::{attention, Gelu, LayerNorm},
    tensor::{backend::Backend, ElementConversion, Tensor},
};
use burn::{
    nn::{loss::CrossEntropyLossConfig, Embedding, EmbeddingConfig, Linear, LinearConfig},
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};
use burn_tensor::{activation, Data, Int, Shape};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use std::path::{Path, PathBuf};
use std::{fs::File, io::BufReader};

use crate::data::batcher::*;
use crate::tokenizer::{Token, TokenId};

#[derive(Config)]
pub struct TextGenerationModelConfig {
    /// Number of tokens in the vocabulary.
    /// GPT-2 has a vocabulary size of 50,257,
    /// which corresponds to the 256 bytes base tokens,
    /// a special end-of-text token and the symbols learned with 50,000 merges
    pub n_vocab: usize,
    /// Maximum context / prompt sequence.
    pub n_ctx: usize,
    /// Number of attention heads.
    /// Must be a divisor of `network_width`.
    pub n_head: usize,
    /// Width of the network, or the embeding dimension.
    pub n_embd: usize,
    /// Width of the network.
    pub n_layer: usize,
}

impl TextGenerationModelConfig {
    pub fn from_dir(model_dir: PathBuf) -> Self {
        let file = File::open(model_dir.join("hparams.json")).expect("should load params");

        let buffer = BufReader::new(file);

        let config: Self = serde_json::from_reader(buffer).expect("should load params");

        config
    }

    pub fn init_from_pretrained_weights<B: Backend>(
        &self,
        model_dir: &PathBuf,
        device: &B::Device,
    ) -> TextGenerationModel<B> {
        let token_embedding_arr: Array2<f32> =
            ndarray_npy::read_npy(model_dir.join("wte.npy")).expect("should load wte");
        let token_embedding_vec: Vec<f32> = token_embedding_arr.iter().copied().collect();

        let token_embedding_tensor: Tensor<B, 2> = Tensor::<B, 2>::from_data(
            Data::new(
                token_embedding_vec.clone(),
                Shape::new([
                    token_embedding_arr.shape()[0],
                    token_embedding_arr.shape()[1],
                ]),
            )
            .convert(),
            device,
        );

        let mut token_embedding: Embedding<B> =
            EmbeddingConfig::new(self.n_vocab, self.n_embd).init(device);
        token_embedding.weight = Param::from_tensor(token_embedding_tensor);

        let position_embedding_arr: Array2<f32> =
            ndarray_npy::read_npy(model_dir.join("wpe.npy")).expect("should load wpe");
        let position_embedding_vec: Vec<f32> = position_embedding_arr.iter().copied().collect();

        let position_embedding_tensor: Tensor<B, 2> = Tensor::<B, 2>::from_data(
            Data::new(
                position_embedding_vec.clone(),
                Shape::new([
                    position_embedding_arr.shape()[0],
                    position_embedding_arr.shape()[1],
                ]),
            )
            .convert(),
            device,
        );

        let mut position_embedding: Embedding<B> =
            EmbeddingConfig::new(self.n_vocab, self.n_embd).init(device);
        position_embedding.weight = Param::from_tensor(position_embedding_tensor);

        let layer_norm_config = Gpt2LayerNormConfig {
            n_embed: self.n_embd,
        };

        // freeze the layer norm params during training
        let layer_norm = layer_norm_config
            .init_from_pretrained_weights(model_dir.join("ln_f"), device)
            .no_grad();

        let block_config = BlockConfig {
            num_heads: self.n_head,
            n_embed: self.n_embd,
            attn_expand_dim: self.n_embd * 3,
            ffn_expand_dim: self.n_embd * 4,
            depth: self.n_layer,
        };

        let blocks = block_config.init_from_pretrained_weights(&model_dir, device);

        TextGenerationModel {
            token_embedding,
            position_embedding,
            blocks,
            layer_norm,
            n_vocab: self.n_vocab,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> TextGenerationModel<B> {
        // freeze the embeddings during training
        let token_embedding = EmbeddingConfig::new(self.n_vocab, self.n_embd).init(device);
        let position_embedding = EmbeddingConfig::new(self.n_ctx, self.n_embd).init(device);

        let layer_norm_config = Gpt2LayerNormConfig {
            n_embed: self.n_embd,
        };

        let block_config = BlockConfig {
            num_heads: self.n_head,
            n_embed: self.n_embd,
            attn_expand_dim: 2304,
            ffn_expand_dim: 3072,
            depth: self.n_layer,
        };

        let blocks = block_config.init(device);
        let layer_norm = layer_norm_config.init(device);
        let n_vocab = self.n_vocab;

        TextGenerationModel {
            token_embedding,
            position_embedding,
            blocks,
            layer_norm,
            n_vocab,
        }
    }
}

#[derive(Module, Debug)]
pub struct TextGenerationModel<B: Backend> {
    pub token_embedding: Embedding<B>,
    pub position_embedding: Embedding<B>,
    pub blocks: Vec<Block<B>>,
    pub layer_norm: Gpt2LayerNorm<B>,
    pub n_vocab: usize,
}

impl<B: Backend> TextGenerationModel<B> {
    pub fn infer(&self, mut inputs: Vec<TokenId>, num_tokens: usize) -> Vec<TokenId> {
        let device = B::Device::default();

        for _ in 0..num_tokens {
            #[allow(clippy::cast_possible_truncation)]
            let indices: Vec<i32> = inputs.iter().map(|token_id| *token_id as i32).collect();

            let indices = Tensor::<B, 1, Int>::from_data(
                Data::new(indices.clone(), Shape::new([indices.len()])).convert(),
                &device,
            )
            .unsqueeze_dim(0);

            let logits = self.forward(indices.clone());

            // select the last row from logits
            // which corresponds to the logits for next token
            let next_token_idx = Tensor::<B, 1, Int>::from_data(
                Data::new(vec![(logits.dims()[0] - 1) as i32], Shape::new([1])).convert(),
                &device,
            );
            // size (1xvocab size)
            let selected_logits = logits.select(0, next_token_idx);
            // argmax along dim 1, not dim 0
            let next_token_id = selected_logits.argmax(1);
            let next_token_id = next_token_id.into_scalar().elem::<i32>();

            println!("next_token_id {next_token_id:?}");

            inputs.push(next_token_id as u64);
        }

        inputs[inputs.len() - num_tokens..].to_vec()
    }

    fn forward(&self, inputs: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let device = B::Device::default();

        let token_embeddings = self.token_embedding.forward(inputs.clone());

        println!("indices {:?}", inputs.clone().to_data());

        let block_size = inputs.dims()[0];
        let indices = Tensor::arange(0..block_size as i64, &device).reshape([1, block_size]);
        let position_embeddings = self.position_embedding.forward(indices.clone());

        // x size (10x768)
        // remove the batch dimension, since input only contains one batch
        let mut x = (token_embeddings + position_embeddings).squeeze(0);

        println!(
            "x max {:?} x min {:?}",
            x.clone().max().into_scalar().elem::<f32>(),
            x.clone().min().into_scalar().elem::<f32>()
        );

        for block in &self.blocks {
            x = block.forward(x);
        }

        println!(
            "x max {:?} x min {:?}",
            x.clone().max().into_scalar().elem::<f32>(),
            x.clone().min().into_scalar().elem::<f32>()
        );

        let x = self.layer_norm.forward(x);

        println!(
            "x max {:?} x min {:?}",
            x.clone().max().into_scalar().elem::<f32>(),
            x.clone().min().into_scalar().elem::<f32>()
        );

        // reuse the embedding matrix wte for the projection
        let output_to_logits = self.token_embedding.weight.val().transpose();
        let x = x.matmul(output_to_logits);

        println!(
            "x max {:?} x min {:?}",
            x.clone().max().into_scalar().elem::<f32>(),
            x.clone().min().into_scalar().elem::<f32>()
        );

        x
    }

    pub fn forward_training(
        &self,
        item: TrainingTextGenerationBatch<B>,
    ) -> ClassificationOutput<B> {
        let [batch_size, block_size] = item.tokens.dims();
        let device = B::Device::default();

        let inputs = item.tokens.to_device(&device);
        let targets = item.targets.to_device(&device);
        let targets_flatten = targets.reshape([batch_size * block_size]);

        let mut output_flatten =
            Tensor::zeros(Shape::new([batch_size * block_size, self.n_vocab]), &device);

        for batch_idx in 0..batch_size {
            for position_idx in 0..block_size {
                let logits = self
                    .forward(
                        inputs
                            .clone()
                            .slice([batch_idx..batch_idx + 1, 0..position_idx + 1]),
                    )
                    .squeeze(1);
                let output_index =
                    Tensor::from_ints([(batch_idx * block_size + position_idx) as i32], &device);
                output_flatten = output_flatten.select_assign(0, output_index, logits);
            }
        }

        // for cross_entropy loss, the inputs are
        // unnormalized logits and Ground truth class indices
        // ref https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch-nn-functional-cross-entropy
        let loss = CrossEntropyLossConfig::new().init(&device);
        let loss = loss.forward(output_flatten.clone(), targets_flatten.clone());

        // inputs to ClassificationOutput should be logits
        ClassificationOutput {
            loss,
            output: output_flatten,
            targets: targets_flatten,
        }
    }
}

/// Need to implement my own config
/// instead of using LinearConfig from Burn
/// since need to add the init from weights function
#[derive(Config)]
pub struct Gpt2LinearLayerConfig {
    pub input_dim: usize,
    pub output_dim: usize,
}

impl Gpt2LinearLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Gpt2LinearLayer<B> {
        let linear_layer = LinearConfig::new(self.input_dim, self.output_dim).init(device);

        Gpt2LinearLayer { linear_layer }
    }

    pub fn init_from_pretrained_weights<B: Backend>(
        &self,
        weights_dir: PathBuf,
        device: &B::Device,
    ) -> Gpt2LinearLayer<B> {
        let weights_arr: Array2<f32> =
            ndarray_npy::read_npy(weights_dir.join("w.npy")).expect("should load w.npy");
        let bias_arr: Array1<f32> =
            ndarray_npy::read_npy(weights_dir.join("b.npy")).expect("should load b.npy");

        let weights_vec = weights_arr.iter().cloned().collect::<Vec<_>>();
        let bias_vec = bias_arr.iter().cloned().collect::<Vec<_>>();

        let weights: Tensor<B, 2> = Tensor::<B, 2>::from_data(
            Data::new(
                weights_vec.clone(),
                Shape::new([weights_arr.shape()[0], weights_arr.shape()[1]]),
            )
            .convert(),
            device,
        );
        let bias: Tensor<B, 1> = Tensor::<B, 1>::from_data(
            Data::new(bias_vec.clone(), Shape::new([bias_vec.len()])).convert(),
            device,
        );

        let weights = Param::from_tensor(weights);
        let bias: Param<Tensor<B, 1>> = Param::from_tensor(bias);

        let mut linear_layer = LinearConfig::new(self.input_dim, self.output_dim).init(device);
        linear_layer.weight = weights;
        linear_layer.bias = Some(bias);

        Gpt2LinearLayer { linear_layer }
    }
}

#[derive(Module, Debug)]
pub struct Gpt2LinearLayer<B: Backend> {
    pub linear_layer: Linear<B>,
}

impl<B: Backend> Gpt2LinearLayer<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear_layer.forward(x)
    }
}

#[derive(Config)]
pub struct FeedForwardConfig {
    pub n_embd: usize,
    pub expand_dim: usize,
}

impl FeedForwardConfig {
    pub fn init_from_pretrained_weights<B: Backend>(
        &self,
        block_dir: &PathBuf,
        device: &B::Device,
    ) -> FeedForward<B> {
        let expand_layer_config = Gpt2LinearLayerConfig {
            input_dim: self.n_embd,
            output_dim: self.expand_dim,
        };
        let expand =
            expand_layer_config.init_from_pretrained_weights(block_dir.join("mlp/c_fc"), device);

        let contract_layer_config = Gpt2LinearLayerConfig {
            input_dim: self.expand_dim,
            output_dim: self.n_embd,
        };
        let contract = contract_layer_config
            .init_from_pretrained_weights(block_dir.join("mlp/c_proj"), device);

        let layer_norm_config = Gpt2LayerNormConfig {
            n_embed: self.n_embd,
        };
        let layer_norm =
            layer_norm_config.init_from_pretrained_weights(block_dir.join("ln_2"), device);

        FeedForward {
            layer_norm,
            expand,
            contract,
            activation: Gelu::new(),
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        let expand_layer_config = Gpt2LinearLayerConfig {
            input_dim: self.expand_dim,
            output_dim: self.n_embd,
        };
        let expand = expand_layer_config.init(device);

        let contract_layer_config = Gpt2LinearLayerConfig {
            input_dim: self.n_embd,
            output_dim: self.expand_dim,
        };
        let contract = contract_layer_config.init(device);

        let layer_norm_config = Gpt2LayerNormConfig {
            n_embed: self.n_embd,
        };
        let layer_norm = layer_norm_config.init(device);

        FeedForward {
            layer_norm,
            expand,
            contract,
            activation: Gelu::new(),
        }
    }
}

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    pub layer_norm: Gpt2LayerNorm<B>,
    pub expand: Gpt2LinearLayer<B>,
    pub contract: Gpt2LinearLayer<B>,
    pub activation: Gelu,
}

impl<B: Backend> FeedForward<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.layer_norm.forward(x);
        let x = self.expand.forward(x);
        let x = self.activation.forward(x);

        let output = self.contract.forward(x);

        output
    }
}

#[derive(Config)]
pub struct AttentionConfig {
    pub num_heads: usize,
    pub n_embd: usize,
    pub expand_dim: usize,
}

impl AttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Attention<B> {
        let expand_layer_config = Gpt2LinearLayerConfig {
            input_dim: self.n_embd,
            output_dim: self.expand_dim,
        };
        let expand = expand_layer_config.init(device);

        let contract_layer_config = Gpt2LinearLayerConfig {
            input_dim: self.n_embd,
            output_dim: self.n_embd,
        };
        let contract = contract_layer_config.init(device);

        let layer_norm_config = Gpt2LayerNormConfig {
            n_embed: self.n_embd,
        };
        let layer_norm = layer_norm_config.init(device);

        Attention {
            layer_norm,
            expand,
            contract,
            num_heads: self.num_heads,
        }
    }

    pub fn init_from_pretrained_weights<B: Backend>(
        &self,
        block_dir: &PathBuf,
        device: &B::Device,
    ) -> Attention<B> {
        let expand_layer_config = Gpt2LinearLayerConfig {
            input_dim: self.n_embd,
            output_dim: self.expand_dim,
        };

        let expand =
            expand_layer_config.init_from_pretrained_weights(block_dir.join("attn/c_attn"), device);

        let contract_layer_config = Gpt2LinearLayerConfig {
            input_dim: self.n_embd,
            output_dim: self.n_embd,
        };

        let contract = contract_layer_config
            .init_from_pretrained_weights(block_dir.join("attn/c_proj"), device);

        let layer_norm_config = Gpt2LayerNormConfig {
            n_embed: self.n_embd,
        };
        let layer_norm =
            layer_norm_config.init_from_pretrained_weights(block_dir.join("ln_1"), device);

        Attention {
            layer_norm,
            expand,
            contract,
            num_heads: self.num_heads,
        }
    }
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    pub layer_norm: Gpt2LayerNorm<B>,
    pub expand: Gpt2LinearLayer<B>,
    pub contract: Gpt2LinearLayer<B>,
    pub num_heads: usize,
}

impl<B: Backend> Attention<B> {
    fn attention(
        q: &Tensor<B, 2>,
        k: &Tensor<B, 2>,
        v: &Tensor<B, 2>,
        causal_mask: &Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        println!("q {:?}", q.clone().max().into_scalar().elem::<f32>());
        println!("k {:?}", k.clone().max().into_scalar().elem::<f32>());
        println!("v {:?}", v.clone().max().into_scalar().elem::<f32>());

        println!("causal_mask {:?}", causal_mask.dims());

        let d = (k.dims()[1] as f32).sqrt();
        let kt = k.clone().transpose();
        let qk = q.clone().matmul(kt) / d + causal_mask.clone();
        let probs = activation::softmax(qk, 1);
        let v = probs.matmul(v.clone());
        v
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = B::Device::default();

        println!(
            "x max {:?} x min {:?}",
            x.clone().max().into_scalar().elem::<f32>(),
            x.clone().min().into_scalar().elem::<f32>()
        );

        let x = self.layer_norm.forward(x);

        println!(
            "x max {:?} x min {:?}",
            x.clone().max().into_scalar().elem::<f32>(),
            x.clone().min().into_scalar().elem::<f32>()
        );

        // x size (10x2034)
        let x = self.expand.forward(x.clone());
        println!("x dim {:?}", x.clone().dims());
        println!(
            "x max {:?} x min {:?}",
            x.clone().max().into_scalar().elem::<f32>(),
            x.clone().min().into_scalar().elem::<f32>()
        );

        let qkv = x.clone().chunk(3, 1);
        let qkv_heads = qkv
            .iter()
            .map(|v| v.clone().chunk(self.num_heads, 1))
            .collect::<Vec<_>>();

        let head_shape = x.dims()[0];
        // mask size (10x10)
        let causal_mask = (Tensor::ones(Shape::new([head_shape, head_shape]), &device)
            - Tensor::ones(Shape::new([head_shape, head_shape]), &device).tril(0))
            * -1.0e4;

        let out_heads = std::iter::zip(std::iter::zip(&qkv_heads[0], &qkv_heads[1]), &qkv_heads[2])
            .map(|((q, k), v)| Self::attention(q, k, v, &causal_mask))
            .collect();
        let out_heads_concat = Tensor::cat(out_heads, 1);
        let x = self.contract.forward(out_heads_concat);

        println!(
            "x max {:?} x min {:?}",
            x.clone().max().into_scalar().elem::<f32>(),
            x.clone().min().into_scalar().elem::<f32>()
        );

        x
    }
}

#[derive(Config)]
pub struct BlockConfig {
    pub num_heads: usize,
    pub n_embed: usize,
    pub attn_expand_dim: usize,
    pub ffn_expand_dim: usize,
    pub depth: usize,
}

impl BlockConfig {
    pub fn init_block_from_pretrained_weights<B: Backend>(
        &self,
        block_dir: &PathBuf,
        device: &B::Device,
    ) -> Block<B> {
        let attention_config = AttentionConfig {
            n_embd: self.n_embed,
            expand_dim: self.attn_expand_dim,
            num_heads: self.num_heads,
        };
        let attention = attention_config.init_from_pretrained_weights(block_dir, device);

        let feedforward_config = FeedForwardConfig {
            n_embd: self.n_embed,
            expand_dim: self.ffn_expand_dim,
        };
        let feedforward = feedforward_config.init_from_pretrained_weights(block_dir, device);

        Block {
            attention,
            feedforward,
        }
    }

    pub fn init_from_pretrained_weights<B: Backend>(
        &self,
        model_dir: &PathBuf,
        device: &B::Device,
    ) -> Vec<Block<B>> {
        (0..self.depth)
            .map(|block_idx| {
                self.init_block_from_pretrained_weights(
                    &model_dir.join(format!("h{block_idx}")),
                    device,
                )
            })
            .collect()
    }

    pub fn init_block<B: Backend>(&self, device: &B::Device) -> Block<B> {
        let attention_config = AttentionConfig {
            n_embd: self.n_embed,
            expand_dim: self.attn_expand_dim,
            num_heads: self.num_heads,
        };
        let attention = attention_config.init(device);

        let feedforward_config = FeedForwardConfig {
            n_embd: self.n_embed,
            expand_dim: self.ffn_expand_dim,
        };
        let feedforward = feedforward_config.init(device);

        Block {
            attention,
            feedforward,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Vec<Block<B>> {
        (0..self.depth)
            .map(|_block_idx| self.init_block(device))
            .collect()
    }
}

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    pub attention: Attention<B>,
    pub feedforward: FeedForward<B>,
}

impl<B: Backend> Block<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = x.clone() + self.attention.forward(x);
        println!(
            "x max {:?} x min {:?}",
            x.clone().max().into_scalar().elem::<f32>(),
            x.clone().min().into_scalar().elem::<f32>()
        );
        let x = x.clone() + self.feedforward.forward(x);
        println!(
            "x max {:?} x min {:?}",
            x.clone().max().into_scalar().elem::<f32>(),
            x.clone().min().into_scalar().elem::<f32>()
        );

        x
    }
}

#[derive(Config)]
pub struct Gpt2LayerNormConfig {
    pub n_embed: usize,
}

impl Gpt2LayerNormConfig {
    pub fn init_from_pretrained_weights<B: Backend>(
        &self,
        model_dir: PathBuf,
        device: &B::Device,
    ) -> Gpt2LayerNorm<B> {
        let beta_arr: Array1<f32> =
            ndarray_npy::read_npy(model_dir.join("b.npy")).expect("should load b.npy");
        let beta_vec = beta_arr.iter().cloned().collect::<Vec<_>>();
        let gamma_arr: Array1<f32> =
            ndarray_npy::read_npy(model_dir.join("g.npy")).expect("should load g.npy");
        let gamma_vec = gamma_arr.iter().cloned().collect::<Vec<_>>();

        let beta: Tensor<B, 1> = Tensor::<B, 1>::from_data(
            Data::new(beta_vec.clone(), Shape::new([beta_vec.len()])).convert(),
            device,
        );
        let gamma: Tensor<B, 1> = Tensor::<B, 1>::from_data(
            Data::new(gamma_vec.clone(), Shape::new([gamma_vec.len()])).convert(),
            device,
        );

        let beta = Param::from_tensor(beta);
        let gamma = Param::from_tensor(gamma);

        Gpt2LayerNorm { beta, gamma }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Gpt2LayerNorm<B> {
        let gamma: Tensor<B, 1> = Tensor::<B, 1>::zeros(Shape::new([self.n_embed]), device);
        let beta: Tensor<B, 1> = Tensor::<B, 1>::zeros(Shape::new([self.n_embed]), device);

        let beta: Param<Tensor<B, 1>> = Param::from_tensor(beta);
        let gamma: Param<Tensor<B, 1>> = Param::from_tensor(gamma);

        Gpt2LayerNorm { beta, gamma }
    }
}

/// this struct loads the learned parameters for layer norm
#[derive(Module, Debug)]
pub struct Gpt2LayerNorm<B: Backend> {
    pub beta: Param<Tensor<B, 1>>,
    pub gamma: Param<Tensor<B, 1>>,
}

impl<B: Backend> Gpt2LayerNorm<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let eps = 1e-5;

        let mean = x.clone().mean_dim(1);

        println!(
            "mean max {:?} mean min {:?}",
            mean.clone().max().into_scalar().elem::<f32>(),
            mean.clone().min().into_scalar().elem::<f32>()
        );

        let var = x.clone().var(1);

        println!(
            "var max {:?} var min {:?}",
            var.clone().max().into_scalar().elem::<f32>(),
            var.clone().min().into_scalar().elem::<f32>()
        );

        let x = (x - mean) / (var + eps).sqrt();

        println!(
            "x max {:?} x min {:?}",
            x.clone().max().into_scalar().elem::<f32>(),
            x.clone().min().into_scalar().elem::<f32>()
        );

        let gamma = self.gamma.val().unsqueeze::<2>();
        let gamma: Tensor<B, 2> = gamma.repeat(0, x.dims()[0]);
        let beta: Tensor<B, 2> = self.beta.val().unsqueeze::<2>();
        let beta = beta.repeat(0, x.dims()[0]);

        // size (10x1) * (10x768) + (10x1)
        let output = gamma * x + beta;

        output
    }
}

impl<B: AutodiffBackend> TrainStep<TrainingTextGenerationBatch<B>, ClassificationOutput<B>>
    for TextGenerationModel<B>
{
    fn step(&self, item: TrainingTextGenerationBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_training(item);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> ValidStep<TrainingTextGenerationBatch<B>, ClassificationOutput<B>>
    for TextGenerationModel<B>
{
    fn step(&self, item: TrainingTextGenerationBatch<B>) -> ClassificationOutput<B> {
        self.forward_training(item)
    }
}
