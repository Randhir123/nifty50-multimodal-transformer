# Spec: Real News Ingestion and FinBERT Text Modality

## 1. Objective
Replace the deterministic, OHLCV-derived "market summary" text records in the real-world demo with actual external news articles fetched via `yfinance`. Additionally, upgrade the text modality to use `ProsusAI/finbert` (or a similar financial language model) to generate the text embeddings, rather than a lightweight hashing strategy.

## 2. Inputs and outputs
- **Inputs**: 
  - Live news data from `yfinance.Ticker(ticker).news`.
- **Outputs**: 
  - `data/processed/real_world_demo/text_records.csv` containing actual news headlines and summaries.
  - The schema remains compatible: `stock_id`, `event_date`, `source_type`, `title`, `body_text`.
  - The `real_world_multimodal_samples.npz` artifact will contain `text_tokens` of dimension `768` (standard BERT hidden size) instead of the placeholder `16`.

## 3. Alignment and leakage rules
- **Strict No-Lookahead**: The `providerPublishTime` returned by `yfinance` must be used as the `event_date`.
- The existing `attach_text_tokens` and text normalizer strictly filter for `event_date <= sample end_date`. As long as we use the true publication timestamp, the leakage invariant is naturally maintained.

## 4. Implementation boundaries
- **In scope**: 
  - Upgrading `_build_text_records` in `scripts/run_real_world_demo.py` to fetch, parse, and save real news items. 
  - Updating the demo script to encode these text records using `src.models.text_encoder.TextEncoder` configured with a FinBERT backbone.
  - Adjusting the default `--text-dim` in the demo script to match the FinBERT output dimension (768).
- **Out of scope**: Scraping heavy external sources like SEC filings or Twitter. Fine-tuning FinBERT weights in this PR.

## 5. Testing strategy
- Run `pytest tests/integration/test_no_leakage.py` to ensure the real news dates do not violate the chronological leakage invariants.

## 6. Acceptance criteria
- Running `python scripts/run_real_world_demo.py` populates `text_records.csv` with real human-readable news headlines.
- The final `real_world_multimodal_samples.npz` artifact successfully encodes these new texts into `768`-dimensional embeddings.
- The ablation runner still succeeds and metrics incorporate the new real-world textual signal.

## 7. Open questions
- `yfinance` news history is typically limited to the most recent ~8 articles per ticker. This means older samples in our 9-month backtest window might naturally lack news context. We must ensure the text encoder and pipeline degrade gracefully (e.g., using empty/zero embeddings) for samples with no available text.