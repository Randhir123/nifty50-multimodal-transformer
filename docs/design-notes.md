# Design Notes

Four non-obvious decisions in the pipeline, with evidence for each. This document exists so a future contributor does not revert them for reasons that were already considered.

---

## Why GAF/MTF over candlestick PNGs

Candlestick PNG rendering as input to a from-scratch ViT encoder produced image tokens at the noise floor in early experiments. The rendered-image path asked the model to learn visual feature extraction from a very small number of training samples per fold, which was not a good match for a coursework-scale dataset.

Gramian Angular Field (GAF) and Markov Transition Field (MTF) encode the close-price series mathematically rather than visually. GAF encodes angular correlations across the 20-day window; MTF encodes quantile-bin state transition probabilities. These are structures that scalar rolling statistics discard by collapsing windows to single numbers. A compact CNN has better inductive bias for this representation than a from-scratch ViT at this scale. In Run C, the image branch produced a `+0.041` ROC-AUC delta over `tabular_only`, tied with text as the strongest single auxiliary modality.

The implementation is in [`src/data/timeseries_images.py`](../src/data/timeseries_images.py) and [`src/models/image_cnn.py`](../src/models/image_cnn.py). Independence is measurable with [`scripts/check_modality_independence.py`](../scripts/check_modality_independence.py).

---

## Why purged walk-forward CV

A single chronological train/test split hides two problems. First, if the model is evaluated on a period immediately following training, the label horizon can overlap the split boundary: a training sample whose label uses data from day `D+1` through `D+H` can cross into the test window. Purging drops training samples whose label window overlaps the validation fold. Second, one held-out period reveals little about whether the model generalizes across regimes.

Walk-forward expanding-window CV with purging and an optional embargo gap is the standard fix. Run C uses 3 folds with a 3-day embargo. With only 1,260 samples, fold-level variance is still meaningful, but walk-forward CV is much more honest than a single split because it tests the model across multiple chronological validation periods.

The implementation is in [`src/training/cv.py`](../src/training/cv.py), with tests in [`tests/unit/test_cv_split.py`](../tests/unit/test_cv_split.py).

---

## Why FinBERT for text

Before real news ETL, text tokens were generated deterministically from OHLCV price statistics, which made the text modality a re-encoding of the same information already present in the tabular branch. Replacing those summaries with real `yfinance` news headlines encoded by FinBERT gives the text branch a source that is not mechanically derived from prices.

FinBERT produces a 768-dimensional embedding, which is projected down to the shared fusion dimension before entering the Transformer. The high dimensionality therefore does not dominate the sequence. FinBERT was chosen over a generic BERT because financial news has domain-specific vocabulary and phrasing: earnings, guidance, margins, regulatory actions, rating changes, and sector-specific language. In Run C, the text branch produced a `+0.041` ROC-AUC delta over `tabular_only`, tied with the image branch.

The text pipeline is in [`src/data/text.py`](../src/data/text.py). Independence measurement uses [`scripts/check_modality_independence.py`](../scripts/check_modality_independence.py).

---

## Why decouple peer universe from training universe

The KG modality is relational. Its 37 features include sector context, peer correlations, sector-relative ranks, peer dispersion, stock-versus-peer spreads, and market-regime indicators. Those features are only meaningful if the peer universe is representative. Computing them against the same 6-stock training universe makes many sectors contain only one or two stocks, which turns sector ranks and peer statistics into unstable or near-random features.

Decoupling the peer universe from the training universe fixes that without forcing the whole supervised pipeline to scale to every Nifty50 stock. The training universe can stay small and rich-data friendly — where news, image tensors, and labels are manageable in Colab — while KG features are computed from a broader OHLCV-only peer universe. Run C uses 6 training stocks and a 49-stock Nifty50 peer universe. That moved the KG result from an unstable `+0.045` delta on a too-small 6-peer universe to a more conservative `+0.014` delta, while `(tabular, kg)` independence remained meaningfully above the shuffled baseline.

The pattern generalizes beyond this project. In any multimodal model where one modality is entity-rich and another modality is relational, the relational source should not automatically be restricted to the supervised training subset. Use the largest leakage-safe peer universe available for relational features, and keep the training universe sized to the modalities that are expensive to collect or encode.

The implementation is in [`src/data/kg_features_v2.py`](../src/data/kg_features_v2.py). Leakage protection for the decoupled peer universe is tested by `tests/integration/test_no_leakage.py::test_kg_v2_peer_universe_no_leakage`.
