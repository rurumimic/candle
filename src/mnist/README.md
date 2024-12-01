# MNIST

- candle: [mnist](https://huggingface.github.io/candle/guide/hello_world.html)

```bash
cargo new mnist
cd mnist
```

## with cuda support

- [rurumimic/cuda](https://github.com/rurumimic/cuda)

```bash
nvcc --version

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Oct_29_23:50:19_PDT_2024
Cuda compilation tools, release 12.6, V12.6.85
Build cuda_12.6.r12.6/compiler.35059454_
```

```bash
nvidia-smi --query-gpu=compute_cap --format=csv

compute_cap
8.6
```

### Add candle with cuda

```bash
cargo add --git https://github.com/huggingface/candle.git candle-core \
          --features "cuda"
```

check build:

```bash
cargo build
```

