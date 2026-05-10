# Design Notes

Three non-obvious decisions in the pipeline, with evidence for each. This document exists so a future contributor does not revert them for reasons that were already considered.

---

## Why GAF/MTF over candlestick PNGs

Candlestick PNG rendering as input to a from-scratch ViT encoder produced image tokens at the noise floor: (tabular, image) independence = 0.047, indistinguishable from shuffled random vectors (~0.041). In the Transformer ablation, `tabular_image` with the ViT encoder was −2.3pp AUC vs `tabular_only` — the image modality was actively hurting performance. The ViT was not learning: with 300–900 training samples per fold, it cannot learn discriminative spatial features from scratch.

Gramian Angular Field (GAF) and Markov Transition Field (MTF) encode the close-price series mathematically rather than visually. GAF encodes angular correlations across the 20-day window; MTF encodes quantile-bin state transition probabilities. These are structures that scalar rolling statistics (the tabular features) explicitly discard by collapsing windows to single numbers. A 3-layer CNN with inductive biases suited to the scale (translation invariance, local receptive fields) can converge on discriminative patterns within 20 epochs. After switching, (tabular, image) rose to 0.082 and `tabular_image` became the highest-AUC variant (+2.8pp over `tabular_only`).

The implementation is in [`src/data/timeseries_images.py`](../src/data/timeseries_images.py) and [`src/models/image_cnn.py`](../src/models/image_cnn.py). Independence is measurable with [`scripts/check_modality_independence.py`](../scripts/check_modality_independence.py).

---

## Why purged walk-forward CV

A single chronological train/test split hides two problems. First, if the model is evaluated on a period immediately following training, the label horizon (3 days) can overlap the split boundary: a training sample whose label uses data from day D+1 through D+3 can cross into the test window, leaking future information. Purging drops training samples whose label window overlaps the test fold. Second, a single held-out period reveals nothing about whether the model generalises across regimes. The one period it happens to cover might be unusually stable or unusually volatile.

Walk-forward expanding-window CV with purging (and an optional embargo gap) is the standard fix. Running 3 folds exposed that performance varies dramatically by regime: fold 1 (stable mid-period) achieves logreg AUC 0.544; folds 0 and 2 produce 0.443 and 0.413, traced to label base-rate shift and a 2.36× volatility regime change respectively. A single held-out split landing on fold 1 would have looked fine; landing on fold 2 would have looked catastrophic; neither tells you anything generalizable. Walk-forward CV is what makes the regime-shift finding visible rather than hiding it behind a cherry-picked split.

The implementation is in [`src/training/cv.py`](../src/training/cv.py), with tests in [`tests/unit/test_cv_split.py`](../tests/unit/test_cv_split.py).

---

## Why FinBERT for text

Before real news ETL, text tokens were generated deterministically from OHLCV price statistics — effectively a re-encoding of the same features already present in the tabular modality. (tabular, text) independence was near-1 and the text modality contributed nothing. Replacing with real `yfinance` news headlines encoded by FinBERT (financial-domain BERT pretrained on financial corpora, 768-dim output) brought (tabular, text) independence to 0.170 — well above the noise floor of 0.041.

The 768-dim embedding is projected down to the shared fusion dimension (16 in the demo configuration), so the high dimensionality does not dominate the sequence. FinBERT was chosen over a generic BERT because financial text has domain-specific vocabulary (earnings, guidance, regulatory filings) that general-domain models systematically misrepresent.

The text pipeline is in [`src/data/text.py`](../src/data/text.py). Independence measurement uses [`scripts/check_modality_independence.py`](../scripts/check_modality_independence.py).
