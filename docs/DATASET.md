# Dataset

## Source

The device catalog (`data/device_catalog.json`) is the committed source of truth and contains **70 real products**: 50 from the FORIA house brand, 20 from established third-party brands (Apple, Google, Aqara, Philips, Ecobee, Yale, Sonos, SwitchBot, Nanoleaf, Tado, Ring, Eve, Yeelight, Eufy, Meross, SONOFF). The non-FORIA entries were cross-checked against manufacturer specs and reviews in April 2026 (see `docs/MODEL_CARD.md > Change history > Phase 2`).

Catalog ingestion flows through `data/collectors/merge_catalogs.py` which reconciles:
- Home Assistant integrations registry (protocol support for ~2000 devices)
- CSA Matter certification database (Matter-capable device list)
- FORIA internal product sheets

Output is reviewed manually and committed to `data/device_catalog.json`. Synthetic devices are NOT committed — see *Augmentation* below.

## Pairs and labels

A dataset row is a *pair* of devices. The training pipeline augments the committed catalog with 30 synthetic devices generated in-memory, yielding 100 devices total and `100 * 99 / 2 = 4950` unique unordered pairs.

### Augmentation (training only)

`data/generate_dataset.py:augment_with_synthetic_devices` generates 30 synthetic devices with `seed=42` (deterministic). They exist purely to widen the feature distribution the model sees during training — specifically to provide coverage of Z-Wave-only devices, Matter-native bridges, and less-common protocol combinations that the real catalog is too small to represent on its own. **These synthetic devices never reach the API**; they are generated at training time in memory and discarded after the CSV is written.

This train-only augmentation was previously leaking into production (see `docs/MODEL_CARD.md > Change history > Phase 1`). The leak has been fixed and the separation is now enforced by the fact that `generate_dataset.py` does not write back to `device_catalog.json`.

### Labeling rule

Each pair gets a label in `{0: incompatible, 1: partial, 2: compatible}` from a deterministic weighted-score rule implemented in `data/generate_dataset.py:compute_compatibility_label`:

```python
score = protocol_overlap  * 35
      + ecosystem_overlap * 30
      + same_brand        * 10
      - hub_conflict      * 15
      + cloud_compatible  * 10
      + category_synergy  * 10

if score >= 55:    label = 2  # compatible
elif score >= 30:  label = 1  # partial
else:              label = 0  # incompatible
```

Weights and thresholds were initialized from heuristics documented in smart-home installation communities (Home Assistant forums, Matter Alliance documentation, manufacturer compatibility matrices). They have **not** been formally validated by a study with certified installers. The rule is a bootstrap — it will be progressively replaced by labels remapped from user/installer feedback via `scripts/monitor.py` → `scripts/retrain.py`. See `docs/MODEL_CARD.md > Relationship to the labeling rule` for the full rationale.

### Label noise

After the rule is applied, `add_noise` perturbs 5% of labels by ±1 class to simulate real-world installer disagreement on edge cases:

- `compatible (2)` → `partial (1)`
- `incompatible (0)` → `partial (1)`
- `partial (1)` → `incompatible (0)` OR `compatible (2)` (random choice)

Compatible never becomes incompatible (and vice-versa) within a single noise operation — labels shift by at most one class. This prevents the model from memorizing the rule's exact boundaries while keeping the noisy labels plausible.

## Distribution

| label | count | % |
|---|---|---|
| incompatible | 1569 | 31.7% |
| partial | 1786 | 36.1% |
| compatible | 1595 | 32.2% |

The distribution is balanced enough that we use `stratify=y` in the train/test split and `f1_macro` as the main metric. Imbalance was significantly worse before the Phase 1 cleanup (see MODEL_CARD change history).

## Features

All features are derived at training and inference time from the raw device specs. No external lookups.

| feature | formula | why |
|---|---|---|
| `protocol_overlap` | `\|A∩B\| / \|A∪B\|` over connectivity sets | Jaccard on protocols. Best single predictor. |
| `ecosystem_overlap` | same, over ecosystems | Does a shared voice assistant or home app exist? |
| `same_brand` | `brand_a == brand_b` | Same-brand pairs are almost always compatible |
| `hub_conflict` | one side needs a Zigbee hub, the other has no Zigbee | Catches the classic "needs a bridge" trap |
| `cloud_compatible` | NOT (both `cloud_dependency=required`) AND NOT (one `local_only` + one `required`) | Two cloud-required devices = single point of failure; local-only + cloud-required = no common control plane |
| `category_synergy` | lookup table over `(category_a, category_b)` pairs | Captures typical use-case combinations (security + security = 1.0, audio + comfort = 0.6, fallback 0.4) |
| `both_hub_required` | `a.hub_required AND b.hub_required` | Two hubs is usually redundant |
| `total_protocols` | `\|A∪B\|` on connectivity | Proxy for flexibility |
| `total_ecosystems` | `\|A∪B\|` on ecosystems | Proxy for breadth |
| `device_*_hub_required` | raw bool per side | Lets the model weight each side's hub requirement separately |
| `device_*_cloud_required` | raw bool per side | Same for cloud dependency |
| `device_*_category_encoded` | `LabelEncoder` on category | Preserves category identity |

**15 features total**. No missing values — the raw specs are complete by construction (enforced by the `DeviceSpec` Pydantic model and the schema validity test in `tests/test_catalog_schema_validity.py`).

## Feature importance (latest training run)

From `models/metrics.json`:

1. `protocol_overlap`, 0.30
2. `hub_conflict`, 0.14
3. `ecosystem_overlap`, 0.12
4. `same_brand`, 0.11
5. `category_synergy`, 0.05

The top three features alone explain 56% of the model's decisions. This matches installer intuition and is the main reason XGBoost was preferred over a neural net — feature-level reasoning stays auditable, which we rely on for model monitoring and for explaining decisions to installers.

The relative importance of `hub_conflict` shifted upward after the Phase 2 catalog corrections: previously 0.10, now 0.14. This is consistent with the two additional hub-dependent products (SwitchBot Curtain 3, Eve Motion Sensor 2nd gen) making hub-related pair patterns more informative.

## Preprocessing pipeline

1. `data/generate_dataset.py` loads `device_catalog.json`, augments in-memory with 30 synthetic devices, enumerates every unique pair, computes the 15 features, applies the rule engine, injects 5% label noise, and writes `data/compatibility_dataset.csv`.
2. `scripts/train.py` loads the CSV, fits a `LabelEncoder` on device categories (persisted to `models/category_encoder.pkl` so the API uses the same mapping), splits 80/20 stratified, trains XGBoost with 5-fold stratified CV for the reported `cv_f1_macro`.
3. `api/main.py` reuses the encoder and the exact same feature function (`build_features`) to guarantee training-serving parity.

## Growth plan

The 4950-pair starting point is small by design — the rule is a bootstrap, not a research-grade label source. The system is built to grow:

- Every prediction written to production is logged with `(features, predicted_label, confidence)` in JSONL form (`PREDICTION_LOG_PATH`, see `api/main.py:log_prediction`).
- Installers and users can flag disagreements through the Mezonya app. Flags become rows in `data/feedback/feedback_<date>.csv`.
- The weekly retrain job (`scripts/retrain.py`, Sunday 03:00 UTC) merges feedback into the main dataset, retrains, and auto-promotes only if the new model wins on F1 macro.

At ~1000 corrections per year the training set doubles in 12-18 months. As real feedback enters the dataset, the model's decision boundary will drift away from the original rule on ambiguous pairs — and most of the new signal is expected to land in the `partial` class, which is the class the current rule gets most often wrong.
