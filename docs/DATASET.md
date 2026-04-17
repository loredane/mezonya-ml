# Dataset

## Source

The device catalog is the live Mezonya catalog, exported as JSON (`data/device_catalog.json`). 20 products were pulled from `mezonya.com`; 30 synthetic but plausible entries were added to cover edge cases the catalog doesn't yet have (Z-Wave-only devices, Matter-native bridges, bluetooth-only wearables).

Additional data sources used by `data/collectors/`:
- Home Assistant integrations registry, protocol support for ~2000 devices
- CSA Matter certification database, Matter-capable device list
- Merged into `data/raw/` and reconciled in `data/processed/`

## Pairs and labels

A dataset row is a *pair* of devices. With 50 catalog entries, we generate `50 * 49 / 2 = 1225` unique unordered pairs.

Each pair gets a label in `{0: incompatible, 1: partial, 2: compatible}` from a rules engine validated by certified smart-home installers:

```
if shared_ecosystems == 0 and not matter_bridge(a, b):
    label = incompatible
elif same_brand and protocol_overlap >= 0.5:
    label = compatible
elif protocol_overlap >= 0.5 and ecosystem_overlap >= 0.67:
    label = compatible
elif protocol_overlap >= 0.3 or ecosystem_overlap >= 0.5:
    label = partial
else:
    label = incompatible
```

A random 5% of labels are perturbed (`partial ↔ compatible`, never to `incompatible`). This simulates the real-world noise we see in installer feedback and prevents the model from memorizing the deterministic rule.

## Distribution

| label | count | % |
|---|---|---|
| incompatible | 426 | 34.8% |
| partial | 544 | 44.4% |
| compatible | 255 | 20.8% |

Imbalance is mild; we use `stratify=y` in the train/test split and `f1_macro` as the main metric rather than accuracy.

## Features

All features are derived at training and inference time from the raw device specs. No external lookups.

| feature | formula | why |
|---|---|---|
| `protocol_overlap` | `|A∩B| / |A∪B|` over connectivity sets | Jaccard on protocols. Best single predictor. |
| `ecosystem_overlap` | same, over ecosystems | Does a shared voice assistant exist? |
| `same_brand` | `brand_a == brand_b` | Same-brand pairs are almost always compatible |
| `hub_conflict` | Zigbee-hub-required vs. non-Zigbee | Catches the classic "needs a bridge" trap |
| `cloud_compatible` | not both `cloud_dependency=required` | Two cloud-required devices = single point of failure |
| `category_synergy` | lookup table, validated by installers | Captures use-case compatibility |
| `both_hub_required` | `a.hub and b.hub` | Two hubs is usually redundant |
| `total_protocols` | `|A∪B|` on connectivity | Proxy for flexibility |
| `total_ecosystems` | `|A∪B|` on ecosystems | Proxy for breadth |
| `device_*_hub_required` | raw bool | Lets the model weight each side |
| `device_*_cloud_required` | raw bool | Same |
| `device_*_category_encoded` | LabelEncoder on category | Preserves category identity |

15 features total. No missing values, the raw specs are complete by construction.

## Feature importance (from `models/metrics.json`)

1. `protocol_overlap`, 0.43
2. `ecosystem_overlap`, 0.27
3. `hub_conflict`, 0.10
4. `category_synergy`, 0.05
5. `cloud_compatible`, 0.04

The top two features alone explain 70% of the model's decisions. This matches installer intuition and is the main reason XGBoost was preferred over a neural net, the reasoning stays auditable.

## Preprocessing pipeline

1. `data/generate_dataset.py` loads `device_catalog.json`, enumerates every unique pair, computes features, applies the rule engine, injects 5% label noise, writes `data/compatibility_dataset.csv`.
2. `scripts/train.py` loads the CSV, fits a `LabelEncoder` on device categories (persisted to `models/category_encoder.pkl` so the API uses the same mapping), splits 80/20 stratified, trains XGBoost.
3. `api/main.py` reuses the encoder and the exact same feature function to guarantee training-serving parity.

## Growth plan

The 1225-pair starting point is intentionally small, it's what one person can curate by hand in a weekend. The system is designed to grow:

- Every prediction written to production is logged with `(features, predicted_label, confidence)`.
- Installers can flag disagreements through the Mezonya app. Those flags become rows in `data/feedback/feedback_<date>.csv`.
- The weekly retrain job merges feedback into the main dataset, retrains, and auto-promotes if the new model wins on the holdout.

At ~1000 installer corrections per year we expect the dataset to double in 12-18 months, with most of the new signal sitting in the `partial` class, the one the deterministic rule gets most often wrong.
