use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_roberta::models::roberta::{RobertaConfig, RobertaModel};
use hf_hub::Repo;
use tokenizers::tokenizer::Encoding;
use tokenizers::Tokenizer;

fn round_to_decimal_places(n: f32, places: u32) -> f32 {
    let multiplier: f32 = 10f32.powi(places as i32);
    (n * multiplier).round() / multiplier
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let model_id = "./models/FacebookAI/roberta-base";
    //let model_id = "./models/cardiffnlp/twitter-roberta-base-sentiment-latest";
    let repo = Repo::model(model_id.to_string());

    let config_filename = "models/FacebookAI/roberta-base/config.json";
    //let config_filename = "models/cardiffnlp/twitter-roberta-base-sentiment-latest/config.json";
    let config_filename = std::fs::read_to_string(config_filename)?;
    let config: RobertaConfig = serde_json::from_str(&config_filename)?;

    let tokenizer_filename = "models/FacebookAI/roberta-base/tokenizer.json";
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

    let weights_filename = "models/FacebookAI/roberta-base/model.safetensors";
    let weights =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device)? };
    let weights = weights.set_prefix("roberta");

    //let weights_filename =
    //    "models/cardiffnlp/twitter-roberta-base-sentiment-latest/pytorch_model.bin";
    //let weights =
    //    VarBuilder::from_pth(&weights_filename, DType::F32, &device)?.set_prefix("roberta");

    let model = RobertaModel::load(weights, &config)?;

    let input = "I love programming in Rust!";
    let encoding = tokenizer.encode(input, false).map_err(anyhow::Error::msg)?;
    println!("Encoding: {:?}", encoding);

    //let input_ids = encoding.get_ids();
    let input_ids = &[[0u32, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]];
    println!("Input IDs: {:?}", input_ids);

    let input_ids = Tensor::new(input_ids, &device)?;
    let token_ids = input_ids.zeros_like()?;

    let output = model.forward(
        &input_ids,
        Some(&token_ids),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
    )?;

    println!("Output: {:?}", output);

    let output = output.squeeze(0)?;
    let output = output.to_vec2::<f32>()?;
    let output: Vec<Vec<f32>> = output
        .iter()
        .take(12)
        .map(|nested_vec| {
            nested_vec
                .iter()
                .take(4)
                .map(|&x| round_to_decimal_places(x, 4))
                .collect()
        })
        .collect();
    println!("Output: {:?}", output);

    Ok(())
}
