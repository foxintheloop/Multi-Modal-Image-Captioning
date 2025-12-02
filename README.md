# Multi-Modal Image Captioning

**Systematic hyperparameter evaluation for CNN-LSTM image captioning.**

Ran 180 experiments comparing vectorization methods, optimizers, and schedulers to identify optimal configurations for generating image descriptions. Benchmarked custom models against BLIP (state-of-the-art).

![Example caption generation](assets/caption-example.png)

## Key Findings

| Finding | Detail |
|---------|--------|
| Best vectorization | **CLIP** — zero-shot transfer approach outperformed BERT and RoBERTa |
| Best optimizer | **Adagrad** — adaptive learning rate balanced visual/textual feature learning |
| Best scheduler | **StepLR** — stable progression; CosineAnnealing comparable |
| Best loss | **CrossEntropyLoss** — direct probability interpretation suited caption generation |
| Experiments | **180 configurations** (3 vectorizations × 5 optimizers × 3 schedulers × 3 loss functions) |

## Architecture

```
Image → CNN (ResNet, frozen early layers) → Adaptive Pooling → Linear Projection (2048 → 512)
                                                                        ↓
Caption Tokens → GloVe Embeddings (300d) → 2-Layer LSTM (512 hidden) ← concat
                                                    ↓
                                              Generated Caption
```

- **Image encoder**: Pre-trained CNN with final classification layer removed
- **Text decoder**: 2-layer LSTM with 512 hidden units
- **Embeddings**: GloVe 6B 300-dimensional vectors
- **Training**: Teacher forcing (ratio 0.5, decaying over epochs)

## Experiment Design

**Vectorization methods:**
- CLIP-ViT-base-patch32 (best — pre-trained on visual-textual relationships)
- BERT-base-uncased (moderate — struggled with scheduler sensitivity)
- RoBERTa-base (competitive METEOR scores — better semantic alignment)

**Optimizers:** Adam, AdamW, RMSprop, SGD, Adagrad

**Schedulers:** StepLR, ExponentialLR, CosineAnnealingLR

**Loss functions:** CrossEntropyLoss, CrossEntropy + label smoothing, NLLLoss

**Compute:** Local (NVIDIA GeForce) + Cloud (NVIDIA A100 40GB)

## Results

| Environment | Vectorization | Best Config | ROUGE | METEOR |
|-------------|---------------|-------------|-------|--------|
| A100 Cloud | CLIP | Adagrad + StepLR + CE | 0.199 | 0.128 |
| Local | CLIP | Adagrad + StepLR + CE | 0.194 | 0.069 |
| Local | RoBERTa | Adagrad + StepLR + CE | 0.157 | 0.091 |
| Local | BERT | Adagrad + ExponentialLR + CE | 0.161 | 0.038 |

**Why CLIP won:** Pre-trained for zero-shot transfer on image-text pairs. BERT and RoBERTa must learn visual representations from scratch.

**Why Adagrad won:** Adaptive learning rates per parameter — automatically balances learning between visual encoder and text decoder components.

**Evaluation insight:** BLEU scores were near-zero across all experiments. ROUGE and METEOR better capture semantic similarity for caption generation where multiple valid descriptions exist per image.

## Dataset

**Flickr30k**: 31,783 images with 158,914 captions (5 per image)

- Caption length: 10-20 words typical, peak at 12
- Lexical diversity (TTR): 0.02 — high vocabulary repetition (expected for visual descriptions)
- Caption similarity: 0.48 average across same-image annotations — significant linguistic variation

## Quick Start

```bash
git clone https://github.com/foxintheloop/Multi-Modal-Image-Captioning.git
cd Multi-Modal-Image-Captioning
pip install -r requirements.txt

# Run training
python train.py --vectorization clip --optimizer adagrad --scheduler steplr
```

## Stack

`Python` `PyTorch` `CLIP` `LSTM` `GloVe` `BLIP` `Flickr30k` `NLTK` `spaCy`

## Metrics

- **BLEU** (nltk, smoothing method 7, n-grams up to 4)
- **ROUGE** (rouge-score library — ROUGE-1, ROUGE-2, ROUGE-L)
- **METEOR** (nltk — exact, porter stem, WordNet synonym matching)

## Project Context

Northwestern University — MSDS 458: Artificial Intelligence and Deep Learning (Fall 2024)

## License

MIT
