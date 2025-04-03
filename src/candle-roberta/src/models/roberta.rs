use std::collections::HashMap;

use anyhow::Result;
use candle_nn::var_builder::VarBuilderArgs;
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

pub struct RobertaModel {}

impl RobertaModel {
    pub fn load(vb: VarBuilder, config: &RobertaConfig) -> Result<Self> {
        Ok(Self {})
    }
}
