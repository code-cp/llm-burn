mod bpe;
mod ext;
mod utils;

pub use bpe::*;
pub use ext::*;
pub use utils::*;

pub type Token = String;
pub type TokenId = u64;
pub type StringPair = (String, String);

use anyhow::Result;

pub trait Tokenizer: Send + Sync {
    fn encode(&self, string: &str) -> Result<Vec<TokenId>>;
    fn decode(&self, tokens: &[TokenId]) -> String;
}
