# Nifty50 Multimodal Transformer

A multimodal Transformer that fuses tabular OHLCV features, real financial news (FinBERT), GAF/MTF time-series images (CNN), and a sector/peer knowledge graph to predict short-horizon outperformance vs the Nifty50 index. The pipeline is leakage-safe with mechanically enforced integration tests and walk-forward purged cross-validation. Modality contributions are quantified by independence measurement: real news reduced (tabular, text) distance correlation from near-1 (price-derived text) to 0.17; GAF/MTF encoding raised the image modality's independence from noise-floor (0.047) to clear signal (0.082). Headline finding: image (GAF/MTF) is the strongest single auxiliary modality (+2.8pp AUC); signal is detectable in stable market regimes and degrades on volatility shifts — identified and explained by the diagnostic framework, not discovered in deployment.

---

## Headline results

| Modality combination | Mean AUC | Δ vs tabular_only |
|---|---|---|
| tabular_only | 0.4963 | — |
| tabular + KG | 0.4974 | +0.001 |
| tabular + text (FinBERT real news) | 0.5104 | +0.014 |
| tabular + image (GAF/MTF + CNN) | 0.5242 | +0.028 |
| All four | 0.5222 | +0.026 |

*3-fold purged walk-forward CV, 20 epochs, 6 stocks, 1 year of data (1,242 samples). Full per-variant results in [`docs/findings.md`](docs/findings.md).*

Absolute AUC is modest: the dataset is small (1,242 samples) and spans a single volatile year. The ordering of contributions is coherent — image > text > KG — and the structural independence measurements confirm each modality contributes different information rather than redundant price-derived noise.

---

## Architecture

Four modalities are projected into a shared embedding space and mixed by Transformer self-attention:

- **Tabular**: 11 OHLCV-derived technical features over a 20-day rolling window (see [`src/data/features.py`](src/data/features.py)). No global normalization; leakage-free.
- **Text**: Real financial news fetched from `yfinance` and encoded by FinBERT (768-dim), filtered to `event_date <= prediction_date`. Falls back to deterministic summaries when news is unavailable for a date.
- **Image**: Gramian Angular Field (GAF) + Markov Transition Field (MTF) images from the 20-day close-price window, encoded by a 3-layer CNN (see [`src/models/image_cnn.py`](src/models/image_cnn.py)). Replaces the earlier candlestick PNG + ViT approach (see "What didn't" below).
- **Knowledge graph**: 4-dim sector, peer, event, and recent-return context aligned by `(stock_id, prediction_date)`.

```text
tabular tokens  ----\
image tokens    -----\
text tokens     ------> shared sequence -> FusionTransformer -> prediction
kg tokens       -----/
```

The fusion model ([`src/models/fusion.py`](src/models/fusion.py)) projects each modality into a common dimension, adds modality-type embeddings, concatenates the sequences, and applies multi-head self-attention. The classification head uses mean pooling over all tokens (not a CLS token — see trainer-collapse fix below).

Training uses BCE loss with output bias initialized to `logit(p_positive)`. Default: 3-fold purged walk-forward CV, 20 epochs, CPU-compatible.

For key design decisions (GAF/MTF, purged CV, FinBERT), see [`docs/design-notes.md`](docs/design-notes.md).

---

## Methodology

### Leakage safety

Every sample is keyed by `(stock_id, end_date)`. The pipeline enforces: tabular windows contain only rows with `date <= end_date`; text records are filtered to `event_date <= end_date`; GAF/MTF images are generated from the OHLCV series sliced to `date <= end_date` (filenames encode the cutoff date as `{SYMBOL}_{YYYYMMDD}.npy`); KG context carries `as_of_date = end_date`; labels use forward prices at `end_date+1` through `end_date+H`. These invariants are mechanically verified on every push by [`tests/integration/test_no_leakage.py`](tests/integration/test_no_leakage.py). A separate manual audit confirmed no leakage in any of the 11 tabular features — all features use only data available at or before date D.

### Cross-validation

Walk-forward expanding-window CV with purged label-window overlap and an optional embargo gap (see [`src/training/cv.py`](src/training/cv.py)). Default: 3 folds, horizon=3 days. Fold boundaries: fold 0 trains on 300 samples and validates on 311 (Sep–Dec 2025); fold 1 trains on 612 and validates on 311 (Dec 2025–Feb 2026); fold 2 trains on 924 and validates on 310 (Feb–May 2026). See [`docs/findings.md`](docs/findings.md) for fold details.

### Modality independence

Each modality's contribution is verified by distance correlation between mean-pooled embeddings. A score near the noise floor (~0.041) indicates the modality is redundant with others; a higher score indicates genuine complementarity. Before real news ETL, text tokens were price-derived and near-fully correlated with tabular features. After real news: (tabular, text) = 0.170. After GAF/MTF encoding: (tabular, image) = 0.082, up from 0.047 with the candlestick ViT. Independence is measured using [`scripts/check_modality_independence.py`](scripts/check_modality_independence.py). Full tables in [`docs/findings.md`](docs/findings.md).

---

## Experimental findings

### Modality contributions

The image modality (GAF/MTF + CNN) is the largest single addition (+0.028 AUC over tabular_only), more than doubling the text contribution (+0.014). The KG contribution (+0.001) is indistinguishable from noise at this dataset scale. The all-four combination (0.5222) is marginally below image-alone (0.5242), consistent with mild noise from text and KG on a 1,242-sample dataset.

With only 3 folds and a single random seed, the *ordering* of contributions is interpretable but individual deltas should not be taken as precise estimates. Multi-seed evaluation would be needed for confidence intervals. What the data supports: image carries independent temporal-shape signal that tabular rolling statistics discard; text carries recent news sentiment independently of price features; KG adds weak peer context at this dataset scale.

### Train→val transfer fails on volatility regime shifts

Per-fold logistic regression AUC (full detail in [`docs/findings.md`](docs/findings.md)):

| Fold | Period | Val AUC | Primary stress |
|---|---|---|---|
| 0 | Sep–Dec 2025 | 0.443 | Label base-rate shift (+9.8pp) |
| 1 | Dec 2025–Feb 2026 | **0.544** | Moderate — both stresses low |
| 2 | Feb–May 2026 | 0.413 | Nifty50 volatility 2.36× training period |

Fold 1 carries the meaningful signal: val AUC 0.544 on a logistic regression over 11 mean-pooled features, demonstrating extractable signal exists in stable regimes. Fold 0 fails because the label base rate shifts by 9.8pp (42% → 52% positive), driven by Nifty50 underperforming its constituents during that period. Fold 2 fails because Nifty50 daily volatility in the validation window is 2.36× higher than training — a structural regime change (consistent with global macro turbulence in early 2026) that features trained on a lower-volatility period cannot generalize across.

Feature audit found no leakage. This is a true regime-shift effect. Walk-forward CV surfaced the finding; a single train/test split would have hidden it. Full breakdown in [`docs/findings.md`](docs/findings.md).

### Backtest status

Using real 3-day forward returns replaced an earlier `y_true` proxy, but session 10a.2 found a second backtest bug: overlapping 3-day positions were compounded as sequential trades. The corrected backtest now aggregates concurrent holdings into daily portfolio returns before compounding and rejects duplicate `(stock_id, end_date)` prediction rows.

The old headline backtest numbers are therefore not cited here. Regenerate them with `scripts/run_backtest.py` or the Colab experiment runner before making any performance claim.

---

## What worked, what didn't, what's open

**Worked.** Leakage-safe pipeline with integration test enforcement. Walk-forward CV with purging — the regime-shift finding is what walk-forward CV is for. Trainer collapse fix: CLS token pooling was collapsing to constant output in shallow 16-dim encoders (probability range ≤ 0.006); switching to mean pooling over all tokens broke the saddle point and restored normal gradient flow, with post-fix tabular_only reaching AUC 0.561 over 50 epochs (see [`docs/findings.md`](docs/findings.md)). Real news ETL: (tabular, text) independence rose from near-0 to 0.170. GAF/MTF + CNN: strongest single auxiliary modality, independence meaningfully above noise floor. Backtest diagnostics now reject both the old label proxy and sequential compounding of overlapping horizon returns.

**Didn't.** ViT-from-scratch chart encoder: with 300–900 training samples per fold, the ViT could not learn discriminative spatial features; image tokens were effectively random noise at independence 0.047 ≈ noise floor. Candlestick PNG rendering: adds overhead and asks the model to learn visual feature extraction rather than exploit temporal structure. Both were identified by measurement and replaced with GAF/MTF + CNN. 16-dim image bottleneck from the ViT pipeline: removed.

**Open.** Universe expansion to 15–20 Nifty50 stocks with 3+ years of history is the highest-leverage next step — it addresses both the regime-shift sensitivity (more data covers multiple volatility periods) and the label non-stationarity (larger peer set reduces sector-rotation noise). Multi-seed evaluation for confidence intervals on modality deltas. Regime-conditional models or volatility-aware training. Longer prediction horizons. Attention attribution to identify which tokens drive predictions.

---

## Reproducing the results

To run on Google Colab instead of locally, open [`notebooks/colab/run_experiment.ipynb`](notebooks/colab/run_experiment.ipynb), set tickers and period in the config cell, and run unattended — results write to your Google Drive.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Run all tests:

```bash
pytest
```

Build a toy all-modality artifact:

```bash
python scripts/build_multimodal_samples.py --toy-output data/processed/multimodal_samples.npz
```

Run the real-world demo (3-stock universe, with ablations):

```bash
python scripts/run_real_world_demo.py \
  --output-dir data/processed/real_world_demo \
  --period 9mo --window-size 20 --horizon-days 3 \
  --run-ablations --epochs 1 --batch-size 4 --device cpu
```

To reproduce the headline results table (20 epochs, 3-fold CV), first run the demo to build the artifact, then:

```bash
python scripts/run_ablation_study.py \
  --dataset data/processed/real_world_demo/real_world_multimodal_samples_gaf.npz \
  --output-dir data/processed/ablations \
  --cv-splits 3 --horizon-days 3 --embargo-days 5 \
  --epochs 20 --batch-size 4 --device cpu \
  --model-dim 16 --num-heads 4 --num-layers 1 --ff-dim 32
```

Generate visualization artifacts:

```bash
python scripts/visualize_real_world_demo.py --demo-dir data/processed/real_world_demo
```

Targeted test checks:

```bash
pytest tests/unit/test_multimodal_sample_builder.py
pytest tests/unit/test_tabular_multimodal_samples.py
pytest tests/unit/test_kg_multimodal_samples.py
pytest tests/unit/test_image_multimodal_samples.py
pytest tests/unit/test_text_multimodal_samples.py
pytest tests/unit/test_ablation_runner.py
pytest tests/integration/test_no_leakage.py
```

---

## Project structure

```text
.
├── AGENTS.md                     # contributor workflow instructions
├── config/                       # ticker lists
├── docs/
│   ├── findings.md               # experimental findings and diagnostic narratives
│   ├── design-notes.md           # key design decisions and their rationale
│   └── figures/                  # visualization snapshots (embedding projections, ablation charts)
├── scripts/                      # runnable demos, ablations, backtest, diagnostics
├── src/
│   ├── data/                     # downloads, features, labels, sample builders
│   ├── kg/                       # graph construction and context retrieval
│   ├── models/                   # tabular, image (CNN), text, fusion models
│   ├── training/                 # train_fusion, cv splits, metrics
│   └── viz/                      # ranking, embedding projection utilities
└── tests/
    ├── integration/              # leakage gate, modality pipeline end-to-end
    └── unit/                     # per-module tests
```

---

## Responsible use

This is a coursework project. It is not financial advice, not a trading system, and not a validated investment model. The absolute AUC numbers are modest, and backtest numbers must be regenerated with the corrected daily aggregation path before being treated as evidence. The diagnostic framework — leakage safety, walk-forward CV, modality independence measurement — is the primary contribution.
