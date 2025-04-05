use std::collections::HashMap;

use anyhow::{Error, Result};
use candle_core::scalar::{TensorOrScalar, TensorScalar};
use candle_core::{DType, Tensor, D};
use candle_nn::ops::softmax;
use candle_nn::{
    embedding, layer_norm, linear, Dropout, Embedding, LayerNorm, Linear, Module, VarBuilder,
};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct RobertaConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    hidden_act: String,
    hidden_dropout_prob: f32,
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
    is_decoder: bool,
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
            is_decoder: false,
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

pub struct RobertaEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    position_embedding_type: String,
    pub padding_idx: u32,
}

impl RobertaEmbeddings {
    pub fn load(vb: VarBuilder, config: &RobertaConfig) -> Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("word_embeddings"),
        )?;

        let position_embeddings = embedding(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("position_embeddings"),
        )?;

        let token_type_embeddings = embedding(
            config.type_vocab_size,
            config.hidden_size,
            vb.pp("token_type_embeddings"),
        )?;

        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;

        let dropout = Dropout::new(config.hidden_dropout_prob as f32);

        let position_embedding_type = "absolute".to_string();

        let padding_idx = config.pad_token_id as u32;

        Ok(Self {
            word_embeddings,
            position_embeddings: Some(position_embeddings),
            token_type_embeddings,
            layer_norm,
            dropout,
            position_embedding_type,
            padding_idx,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        position_ids: &Tensor,
        inputs_embeds: &Tensor,
        past_key_values_length: u8,
    ) -> Result<Tensor> {
        //let position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds);

        let token_type_embeddings = self.token_type_embeddings.forward(&token_type_ids)?;

        let mut embeddings = (inputs_embeds + token_type_embeddings)?;
        embeddings = self.layer_norm.forward(&embeddings)?;
        //embeddings = self.dropout.forward(&embeddings, false)?;
        Ok(embeddings)
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

struct RobertaSelfAttention {
    num_attention_heads: usize,
    attention_head_size: usize,
    all_head_size: usize,
    query: Linear,
    key: Linear,
    value: Linear,
    dropout: Dropout,
    position_embedding_type: Option<String>,
    max_position_embeddings: usize,
    distance_embedding: Embedding,
    is_decoder: bool,
}

impl RobertaSelfAttention {
    fn load(vb: VarBuilder, config: &RobertaConfig) -> Result<Self> {
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = num_attention_heads * attention_head_size;
        let query = linear(config.hidden_size, all_head_size, vb.pp("query"))?;
        let key = linear(config.hidden_size, all_head_size, vb.pp("key"))?;
        let value = linear(config.hidden_size, all_head_size, vb.pp("value"))?;
        let dropout = Dropout::new(config.attention_probs_dropout_prob as f32);
        let position_embedding_type = config.position_embedding_type.clone();
        let max_position_embeddings = config.max_position_embeddings;
        let distance_embedding = embedding(
            2 * config.max_position_embeddings + 1,
            attention_head_size,
            vb.pp("distance_embedding"),
        )?;
        let is_decoder = config.is_decoder;

        Ok(Self {
            num_attention_heads,
            attention_head_size,
            all_head_size,
            query,
            key,
            value,
            dropout,
            position_embedding_type: Some(position_embedding_type),
            max_position_embeddings,
            distance_embedding,
            is_decoder,
        })
    }

    fn transpose_for_scores(&self, x: &Tensor) -> Result<Tensor> {
        let mut new_x_shape = x.dims().to_vec();
        new_x_shape[1] = self.num_attention_heads;
        new_x_shape.push(self.attention_head_size);
        let x = x.reshape(new_x_shape)?;
        let x = x.permute((0, 2, 1, 3))?;
        Ok(x.contiguous()?)
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        head_mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        encoder_attention_mask: Option<&Tensor>,
        past_key_value: Option<(&Tensor, &Tensor)>,
        output_attentions: bool,
    ) -> Result<Tensor> {
        let mixed_query_layer = self.query.forward(hidden_states)?;

        let is_cross_attention = encoder_hidden_states.is_some();

        let (key_layer, value_layer, attention_mask) = if is_cross_attention
            && past_key_value.is_some()
        {
            let key_layer = past_key_value.unwrap().0.clone();
            let value_layer = past_key_value.unwrap().1.clone();
            let attention_mask = encoder_attention_mask.unwrap().clone();
            (key_layer, value_layer, Some(attention_mask))
        } else if is_cross_attention {
            let key_layer =
                self.transpose_for_scores(&self.key.forward(encoder_hidden_states.unwrap())?)?;
            let value_layer =
                self.transpose_for_scores(&self.value.forward(encoder_hidden_states.unwrap())?)?;
            let attention_mask = encoder_attention_mask.unwrap().clone();
            (key_layer, value_layer, Some(attention_mask))
        } else if past_key_value.is_some() {
            let mut key_layer = self.transpose_for_scores(&self.key.forward(hidden_states)?)?;
            let mut value_layer = self.transpose_for_scores(&self.value.forward(hidden_states)?)?;
            key_layer = Tensor::cat(&[past_key_value.unwrap().0.clone(), key_layer], 2)?;
            value_layer = Tensor::cat(&[past_key_value.unwrap().1.clone(), value_layer], 2)?;
            (
                key_layer,
                value_layer,
                Some(attention_mask.unwrap().clone()),
            )
        } else {
            let key_layer = self.transpose_for_scores(&self.key.forward(hidden_states)?)?;
            let value_layer = self.transpose_for_scores(&self.value.forward(hidden_states)?)?;
            (
                key_layer,
                value_layer,
                Some(attention_mask.unwrap().clone()),
            )
        };

        let query_layer = self.transpose_for_scores(&mixed_query_layer)?;

        //let use_cache = past_key_value.is_some();

        //let attention_scores = query_layer.matmul(&key_layer.t()?)?;
        let attention_scores = query_layer.matmul(&key_layer.transpose(2, 3)?)?;
        let attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;
        let attention_scores = match attention_mask {
            Some(mask) => attention_scores.add(&mask)?,
            None => attention_scores,
        };

        let attention_probs = softmax(&attention_scores, D::Minus1)?;
        let attention_probs = self.dropout.forward(&attention_probs, false)?;
        let attention_probs = match head_mask {
            Some(mask) => attention_probs.mul(mask)?,
            None => attention_probs,
        };

        let context_layer = attention_probs
            .matmul(&value_layer)?
            .permute((0, 2, 1, 3))?
            .contiguous()?;

        let mut new_context_layer_shape =
            context_layer.dims()[..context_layer.dims().len() - 2].to_vec();
        new_context_layer_shape.push(self.all_head_size);
        let context_layer = context_layer.reshape(new_context_layer_shape)?;

        Ok(context_layer)
    }
}
