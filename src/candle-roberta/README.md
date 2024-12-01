# candle-roberta

- [toluclassics/candle-tutorial](https://github.com/ToluClassics/candle-tutorial)
- huggingface
  - [roberta](https://huggingface.co/docs/transformers/model_doc/roberta)
  - [xlm-roberta](https://huggingface.co/docs/transformers/model_doc/xlm-roberta)
- github
  - [huggingface/transformers/src/transformers/models](https://github.com/huggingface/transformers/tree/main/src/transformers/models)

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

