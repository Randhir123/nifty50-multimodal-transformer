# Tasks

## Milestone 1 — Repo scaffold

* create repo structure
* add README, pyproject, requirements, gitignore
* add src/data, src/models, src/training, src/kg, src/viz, src/app

## Milestone 2 — Data pipeline

* implement OHLCV feature generation
* implement next-3-day relative outperformance labels
* implement rolling window dataset builder

## Milestone 3 — Candlestick charts

* implement deterministic 60-day candlestick chart generation
* connect chart paths to dataset rows

## Milestone 4 — Tabular Transformer baseline

* implement tabular Transformer
* implement time-based train/validation split
* implement evaluation metrics

## Milestone 5 — Image branch

* implement candlestick image encoder
* train image-only classifier

## Milestone 6 — Text branch

* add text schema
* implement text encoder path
* train text-only classifier

## Milestone 7 — Knowledge augmentation

* build lightweight graph
* implement graph context retrieval
* expose KG features/tokens

## Milestone 8 — Multimodal fusion Transformer

* implement modality fusion
* support tabular + image
* support tabular + text
* support tabular + text + image + KG

## Milestone 9 — Visualization

* ranking table
* peer graph
* embedding projection

## Milestone 10 — OpenClaw operationalization

* workflow entry points
* integration docs
* minimal API surface
