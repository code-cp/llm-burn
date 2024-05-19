use std::env;
use std::fs;

use burn::data::dataset::Dataset;
use derive_new::new;

#[derive(new, Clone, Debug)]
pub struct TextGenerationItem {
    pub text: String,
}

pub struct CustomDataset {
    dataset: String,
    block_size: usize,
}

/// Implement dataset trait for custom dataset
/// ref <https://docs.rs/burn-dataset/latest/burn_dataset/trait.Dataset.html>
impl Dataset<TextGenerationItem> for CustomDataset {
    fn get(&self, index: usize) -> Option<TextGenerationItem> {
        if index >= self.len() {
            return None;
        }

        let data = self
            .dataset
            .chars()
            .skip(index)
            .take(self.block_size)
            .collect::<String>();

        Some(TextGenerationItem::new(data))
    }

    fn len(&self) -> usize {
        self.dataset.len() - self.block_size + 1
    }
}

impl CustomDataset {
    pub fn new_with_contents(contents: &str, split: &str, block_size: usize) -> Self {
        let (train, test) = contents.split_at(contents.len() * 9 / 10);

        let dataset = match split {
            "train" => train,
            "test" => test,
            _ => panic!("{} is not train or test", split),
        };

        Self {
            dataset: String::from(dataset),
            block_size,
        }
    }

    pub fn new(data_file: &str, split: &str, block_size: usize) -> Self {
        let contents = fs::read_to_string(data_file).unwrap();
        Self::new_with_contents(&contents, split, block_size)
    }

    pub fn train(data_file: &str, block_size: usize) -> Self {
        Self::new(data_file, "train", block_size)
    }

    pub fn test(data_file: &str, block_size: usize) -> Self {
        Self::new(data_file, "test", block_size)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::{fs, io::prelude::*, path::Path};

    #[test]
    pub fn check_length() {
        let dataset = CustomDataset::new_with_contents("0123456789", "train", 4);
        assert_eq!(6, dataset.len());
    }

    #[test]
    pub fn check_get() {
        let dataset = CustomDataset::new_with_contents("0123456789", "train", 4);
        assert_eq!(Some("0123".to_string()), dataset.get(0).map(|x| x.text));
    }

    #[test]
    fn test_custom_dataset() {
        let current_dir = env::current_dir().expect("Failed to get current directory");
        let dataset_dir = current_dir.join("data/dataset.txt");
        let dataset_char = fs::read_to_string(dataset_dir).expect("Should read dataset");
        let dataset = CustomDataset {
            dataset: dataset_char,
            block_size: 8,
        };

        println!("dataset len {}", dataset.len());
        println!("{:?}", dataset.get(58185).unwrap());
    }
}
