Domain-Specific Style Transfer Using QLoRA: A Comparative Study of QLoRA on a Finance Domain Dataset

> Coventry University — MSc Data Science and Computational Intelligence
> Task 1: Comparative LLM Fine-Tuning Project

---

Overview

This project investigates whether a small, freely available language model can be taught to rewrite everyday business sentences in a formal executive style — the kind of language you would find in an investor report or a CEO briefing.

The model used is TinyLlama-1.1B-Chat, fine-tuned using QLoRA (Quantised Low-Rank Adaptation) on a curated subset of a public text-transformation dataset. The fine-tuned model is compared against the baseline (no fine-tuning) using ROUGE and BLEU metrics, as well as qualitative analysis of real outputs.

Full fine-tuning was not possible on the available hardware (NVIDIA Tesla T4, 15.6 GB VRAM) and is formally justified in the paper and notebook.

---

Results Summary

| Metric   | Baseline | QLoRA | Change | % Improvement |
|----------|----------|-------|--------|---------------|
| ROUGE-1  | 0.3312   | 0.4851 | +0.1539 | +46.5% |
| ROUGE-2  | 0.0743   | 0.1969 | +0.1226 | +165.0% |
| ROUGE-L  | 0.2335   | 0.4034 | +0.1699 | +72.8% |
| BLEU     | 2.08     | 12.02  | +9.94   | +477.9% |

Training time: ~4 minutes on a free Google Colab T4 GPU.

---

Project Structure

```
├── executive_rewriter_finetuning.py   # Full training and evaluation script
├── README.md                          # This file
```

The notebook is designed to run end-to-end on Google Colab with no local setup required.

---

Quick Start

1. Open in Google Colab

Upload `executive_rewriter_finetuning.py` to a new Colab notebook, or copy the cells directly.

2. Install dependencies

```python
!pip install -q transformers datasets peft accelerate bitsandbytes trl rouge_score nltk
```

3. Run the notebook top to bottom

The notebook will:
- Load and filter the dataset
- Evaluate the baseline model
- Fine-tune using QLoRA
- Evaluate the fine-tuned model
- Print a full comparative results table
- Run a live inference demo on custom inputs

---

Model and Dataset

| Item | Detail |
|------|--------|
| Base model | [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) |
| Dataset | [sugiv/synthetic-text-transformation-dataset](https://huggingface.co/datasets/sugiv/synthetic-text-transformation-dataset) |
| Raw dataset size | 50,000 examples |
| After filtering | 369 examples |
| Training split | 295 (80%) |
| Validation split | 37 (10%) |
| Test split | 37 (10%) |

---

Training Configuration

| Setting | Value |
|---------|-------|
| GPU | NVIDIA Tesla T4, 15.6 GB |
| PyTorch / CUDA | 2.10.0 / 12.8 |
| LoRA rank / alpha | 16 / 32 |
| LoRA target layers | q, k, v, o projections |
| Quantisation | 4-bit NF4, double quantisation |
| Trainable parameters | 4,505,600 (0.41% of total) |
| Effective batch size | 16 (4 per device × 4 grad. accum.) |
| Epochs | 2 |
| Learning rate | 2e-4 (cosine schedule) |
| Max sequence length | 256 tokens |
| Final training loss | 1.2666 |
| Training time | ~247 seconds |

---

Data Filtering Pipeline

The raw dataset was filtered in five stages to keep only relevant, high-quality examples:

1. Finance keyword filter — kept examples containing terms like revenue, profit, margin, churn, liquidity, kpi (50,000 → 5,726)
2. Sentence length filter — inputs between 10 and 60 words only (5,726 → 5,113)
3. Length ratio filter — output/input word ratio between 0.9 and 1.6 (5,113 → 2,874)
4. Marketing hyperbole filter — removed outputs with terms like "game-changing" or "revolutionary" (2,874 → 2,668)
5. Tone and complexity filter — Professional tone, Advanced or Intermediate complexity only (2,668 → 369)

---

Example Outputs

Input:
> The company's recent financial performance has been disappointing, and this has led to a decline in investor confidence.

Baseline output:
> Dear [Recipient], I am writing to express my deep concern about the recent financial performance of our company...

QLoRA output:
> The company's recent financial performance has been markedly suboptimal, and this has resulted in a decline in investor sentiment. It is imperative that we address these challenges promptly to avert further damage.

---

Libraries Used

```
transformers
peft
trl
bitsandbytes
accelerate
datasets
rouge_score
nltk
torch
pandas
numpy
```

---

Hardware Note

This project was developed and tested entirely on **Google Colab free tier** (Tesla T4, 15.6 GB VRAM). Full fine-tuning was not feasible on this hardware — see the paper for a full technical justification.

If you have access to a more powerful GPU (e.g. A100 or H100), you can increase the LoRA rank, batch size, and number of epochs for potentially better results.