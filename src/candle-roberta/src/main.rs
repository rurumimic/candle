use anyhow::{Error, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_roberta::models::roberta::{RobertaConfig, RobertaModel};
use hf_hub::{Cache, Repo};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let device = Device::Cpu;
    let model_id = "./models/cardiffnlp/twitter-roberta-base-sentiment-latest";
    let repo = Repo::model(model_id.to_string());

    let config_filename = "models/cardiffnlp/twitter-roberta-base-sentiment-latest/config.json";
    let config_filename = std::fs::read_to_string(config_filename)?;
    let config: RobertaConfig = serde_json::from_str(&config_filename)?;

    let tokenizer_filename =
        "models/cardiffnlp/twitter-roberta-base-sentiment-latest/tokenizer.json";
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

    let weights_filename =
        "models/cardiffnlp/twitter-roberta-base-sentiment-latest/model.safetensors";
    let weights =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device)? };

    let model = RobertaModel::load(weights, &config)?;

    Ok(())
}
