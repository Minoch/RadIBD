# RadIBD: Vision-Language Transfer Learning for CT Enterography

This repository contains the code for **"Representation geometry shapes task performance in vision-language modeling for CT enterography"**.

## Key Findings

1. **Classification-Retrieval Trade-off**: Mean pooling favors classification (59.2% accuracy), attention pooling favors retrieval (0.235 MRR) — this holds across all LoRA configurations tested.

2. **Per-slice contrast > spatial coverage**: Multi-window RGB encoding outperforms multiplanar sampling. Adding coronal/sagittal views hurts classification.

3. **RAG prevents severity-ordering failure**: Fine-tuned MedGemma achieves only chance-level severity ordering (70.4% within-1 vs 71% random). RAG scores 78-85%, improving ordinal MAE from 0.98 to 0.80-0.89.

## Method Overview

- **Base model**: BiomedCLIP (ViT-B/16 + PubMedBERT)
- **Volume encoding**: 2.5D slice-based with multi-window RGB (3 HU windows → RGB channels)
- **Aggregation**: Mean pooling, attention pooling, or lightweight transformer
- **Fine-tuning**: LoRA on vision and text encoders
- **Contrastive loss**: Multi-positive formulation for templated text
- **Report generation**: MedGemma-4B with RAG using learned embeddings
- **Labels**: Three-teacher pseudolabel ensemble (NegEx + BioMistral + Qwen2.5)

## Results Summary

| Task | Best Config | Performance |
|------|-------------|-------------|
| Classification | Mean, v8b12_t8b6 | 59.2% accuracy |
| Retrieval (T2I) | Attn, v4b6_t4b6 | 0.235 MRR |
| Label Consistency | RAG (any) | 0.80-0.89 ord. MAE |

## Requirements

```bash
pip install torch transformers open_clip_torch scikit-learn nltk sacrebleu bert-score
