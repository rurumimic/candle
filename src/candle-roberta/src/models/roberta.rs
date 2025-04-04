use std::collections::HashMap;

use anyhow::{Error, Result};
use candle_core::scalar::{TensorOrScalar, TensorScalar};
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct RobertaConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    hidden_act: String,
    hidden_dropout_prob: f64,
    attention_probs_dropout_prob: f64,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    initializer_range: f64,
    layer_norm_eps: f64,
    pad_token_id: usize,
    bos_token_id: usize,
    eos_token_id: usize,
    position_embedding_type: String,
    use_cache: bool,
    classifier_dropout: Option<f64>,
    model_type: Option<String>,
    _num_labels: Option<usize>,
    id2label: Option<HashMap<String, String>>,
    label2id: Option<HashMap<String, usize>>,
}

impl Default for RobertaConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50265,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 1,
            bos_token_id: 0,
            eos_token_id: 2,
            position_embedding_type: "absolute".to_string(),
            use_cache: true,
            classifier_dropout: None,
            model_type: Some("roberta".to_string()),
            _num_labels: Some(3),
            id2label: None,
            label2id: None,
        }
    }
}

pub struct RobertaModel {
    pub padding_idx: u32,
}

impl RobertaModel {
    pub fn load(vb: VarBuilder, config: &RobertaConfig) -> Result<Self> {
        Ok(Self {
            padding_idx: config.pad_token_id as u32,
        })
    }

    pub fn create_position_ids_from_inputs_embeds(&self, inputs_embeds: &Tensor) -> Result<Tensor> {
        let input_shape = inputs_embeds.dims3()?;
        let sequence_length = input_shape.1;

        let position_ids = Tensor::arange(
            self.padding_idx + 1,
            sequence_length as u32 + self.padding_idx as u32 + 1,
            inputs_embeds.device(),
        )?;

        Ok(position_ids
            .unsqueeze(0)?
            .expand((input_shape.0, input_shape.1))?)
    }
}

fn create_position_ids_from_input_ids(
    input_ids: &Tensor,
    padding_idx: u32,
    past_key_values_length: u8,
) -> Result<Tensor> {
    let mask = input_ids.ne(padding_idx)?;
    let incremental_indices = mask
        .cumsum(1)?
        .broadcast_add(&Tensor::new(&[past_key_values_length], input_ids.device())?)?
        .mul(&mask)?
        .broadcast_add(&Tensor::new(&[padding_idx], input_ids.device())?)?;

    Ok(incremental_indices)
}
