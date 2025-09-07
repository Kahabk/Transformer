# Transformer
rom-scratch 1M parameter LLM for low-VRAM GPUs (e.g., RTX 2050 4GB). Built with Python/PyTorch, inspired by "Attention Is All You Need." Features custom BPE tokenizer, 8-bit quantization, ~30% FLOP reduction via sparsity. Supports text gen, QA, translation. Trained on WikiText, Common Crawl

## Compact Transformer LLM
From-scratch 1M parameter LLM for low-VRAM GPUs (e.g., RTX 2050 4GB). Built with Python/PyTorch, using RoPE and multi-head attention from "Attention Is All You Need." Supports Q&A and text gen. Trained on LMSYS-Chat-1M subset. Features sparsity, mixed-precision, top-k sampling.
Setup

GPU ≥4GB, Python 3.8+, PyTorch 2.0+, transformers, datasets
git clone repo; pip install -r requirements.txt

### Usage
Train: python train.py --batch_size 2 --max_len 64Infer: python generate.py --question "What is AI?"
Optimize

## Filter clean Q&A data; adjust max_len.
Use mixed precision, --gradient_accumulation_steps 8.
Tune --top_k 50, --temperature 0.7.
Set --batch_size 1 for 4GB GPUs; monitor with nvidia-smi.

### Results

Perplexity: Competitive on LMSYS-Chat-1M.
Size: ~4MB post-training.
Speed: ~0.5s/token on RTX 2050.

