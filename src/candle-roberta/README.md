# candle-roberta

- [toluclassics/candle-tutorial](https://github.com/ToluClassics/candle-tutorial)
- huggingface
  - [roberta](https://huggingface.co/docs/transformers/model_doc/roberta)
  - [xlm-roberta](https://huggingface.co/docs/transformers/model_doc/xlm-roberta)
- github
  - [huggingface/transformers/src/transformers/models](https://github.com/huggingface/transformers/tree/main/src/transformers/models)
    - [modeling_roberta.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py)
  - [huggingface/candle/candle-transformers/src/models](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models)
    - [xlm_roberta.rs](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/xlm_roberta.rs)

---

## Examples

```bash
git clone https://github.com/ToluClassics/candle-tutorial.git
cd candle-tutorial
```

update `Cargo.toml`

```bash
cargo run

token_ids: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
input_ids: [[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2], [0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]]
output: Ok((2, 768))
logits: [0.1, 0.5]
logits: [0.2, 0.6]
```

---

## Setup

### New project

```bash
cargo new candle-roberta
cd candle-roberta
```

### Add candle

```bash
cargo add --git https://github.com/huggingface/candle.git candle-core
cargo add --git https://github.com/huggingface/candle.git candle-nn
```

### Add packages

```bash
cargo add anyhow serde serde_json hf-hub tokenizers
```

---

## Get a sample model

### Install a HuggingFace CLi

#### Set up a virtual environment

```bash
uv venv
source .venv/bin/activate
```

#### Install the HuggingFace Transfer

```bash
uv pip install "huggingface_hub[hf_transfer]"
uv pip install "huggingface_hub[hf_xet]"
```

### Download a model

- huggingface: [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- huggingface: [cardiffnlp/twitter-xlm-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment)
- docs
  - [environment_variables](https://huggingface.co/docs/huggingface_hub/main/package_reference/environment_variables#hfhubenablehftransfer)
  - [storage-backends](https://huggingface.co/docs/hub/storage-backends)

#### RoBERTa

Download a model:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 \
huggingface-cli download "cardiffnlp/twitter-roberta-base-sentiment-latest" --local-dir models/cardiffnlp/twitter-roberta-base-sentiment-latest
```

```bash
models/cardiffnlp/twitter-roberta-base-sentiment-latest
├── .cache/
├── config.json
├── .gitattributes
├── merges.txt
├── pytorch_model.bin
├── README.md
├── special_tokens_map.json
├── tf_model.h5
└── vocab.json
```

##### Describe a model

Install additional packages for model:

```bash
uv pip install transformers
uv pip install torch
```

load a model:

```py
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("./models/cardiffnlp/twitter-roberta-base-sentiment-latest")
```

```bash
Some weights of the model checkpoint at ./models/cardiffnlp/twitter-roberta-base-sentiment-latest were not used when initializing RobertaForSequenceClassification: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
- This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
```

describe a model:

```py
model
```

```py
RobertaForSequenceClassification(
  (roberta): RobertaModel(
    (embeddings): RobertaEmbeddings(
      (word_embeddings): Embedding(50265, 768, padding_idx=1)
      (position_embeddings): Embedding(514, 768, padding_idx=1)
      (token_type_embeddings): Embedding(1, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): RobertaEncoder(
      (layer): ModuleList(
        (0-11): 12 x RobertaLayer(
          (attention): RobertaAttention(
            (self): RobertaSdpaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): RobertaSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): RobertaIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): RobertaOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
  )
  (classifier): RobertaClassificationHead(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
    (out_proj): Linear(in_features=768, out_features=3, bias=True)
  )
)
```

##### Describe a tokenizer

load a tokenizer:

```py
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("./models/cardiffnlp/twitter-roberta-base-sentiment-latest")
```

describe a tokenizer:

```py
tokenizer
```

```py
RobertaTokenizerFast(name_or_path='./models/cardiffnlp/twitter-roberta-base-sentiment-latest', vocab_size=50265, model_max_length=1000000000000000019884624838656, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'sep_token': '</s>', 'pad_token': '<pad>', 'cls_token': '<s>', 'mask_token': '<mask>'}, clean_up_tokenization_spaces=False, added_tokens_decoder={
        0: AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
        1: AddedToken("<pad>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
        2: AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
        3: AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
        50264: AddedToken("<mask>", rstrip=False, lstrip=True, single_word=False, normalized=False, special=True),
}
)
```

#### XLM-RoBERTa

Download a model:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 \
huggingface-cli download "cardiffnlp/twitter-xlm-roberta-base-sentiment" --local-dir models/cardiffnlp/twitter-xlm-roberta-base-sentiment
```

```bash
models/cardiffnlp/twitter-xlm-roberta-base-sentiment
├── .cache/
├── config.json
├── .gitattributes
├── pytorch_model.bin
├── README.md
├── sentencepiece.bpe.model
├── special_tokens_map.json
└── tf_model.h5
```

##### Describe a model

Install additional packages for model:

```bash
uv pip install transformers
uv pip install torch
```

```py
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("./models/cardiffnlp/twitter-xlm-roberta-base-sentiment")

model
```

```py
XLMRobertaForSequenceClassification(
  (roberta): XLMRobertaModel(
    (embeddings): XLMRobertaEmbeddings(
      (word_embeddings): Embedding(250002, 768, padding_idx=1)
      (position_embeddings): Embedding(514, 768, padding_idx=1)
      (token_type_embeddings): Embedding(1, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): XLMRobertaEncoder(
      (layer): ModuleList(
        (0-11): 12 x XLMRobertaLayer(
          (attention): XLMRobertaAttention(
            (self): XLMRobertaSdpaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): XLMRobertaSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): XLMRobertaIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): XLMRobertaOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
  )
  (classifier): XLMRobertaClassificationHead(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
    (out_proj): Linear(in_features=768, out_features=3, bias=True)
  )
)
```

##### Describe a tokenizer

Install additional packages for tokenizer:

```bash
uv pip install sentencepiece protobuf
```

```py
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("./models/cardiffnlp/twitter-xlm-roberta-base-sentiment")

tokenizer
```

```py
XLMRobertaTokenizerFast(name_or_path='./models/cardiffnlp/twitter-xlm-roberta-base-sentiment', vocab_size=250002, model_max_length=1000000000000000019884624838656, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'sep_token': '</s>', 'pad_token': '<pad>', 'cls_token': '<s>', 'mask_token': '<mask>'}, clean_up_tokenization_spaces=False, added_tokens_decoder={
        0: AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        1: AddedToken("<pad>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        2: AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        3: AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        250001: AddedToken("<mask>", rstrip=False, lstrip=True, single_word=False, normalized=False, special=True),
}
)
```

