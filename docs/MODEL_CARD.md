# Model Card: Mezonya Compatibility Classifier

Following the [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) template.

## Model details

- **Model type**: gradient-boosted decision trees (XGBoost `multi:softprob`)
- **Version**: see `models/metrics.json` field `model_version` (format `YYYYMMDD_HHMMSS`, UTC)
- **Task**: 3-class classification over smart-home device *pairs*
- **Classes**: `incompatible` (0), `partial` (1), `compatible` (2)
- **Inputs**: 15 engineered features derived from two `DeviceSpec` objects (brand, category, connectivity, ecosystems, hub/cloud requirements). See `docs/DATASET.md`.
- **Training pipeline**: `scripts/train.py`. Reproducible from `data/compatibility_dataset.csv` via `random_state=42` everywhere.
- **Owner**: ML team, Mezonya. Contact via the issue tracker on the private GitHub repo.

## Intended use

- **Primary**: power the compatibility quiz and "find matching products" button on mezonya.com.
- **Secondary**: internal tooling to audit new devices before they enter the catalog.
- **Out of scope**: any safety- or health-critical decision. The model has never been validated for anything beyond purchase recommendation.

## Training data

**4950 device pairs** built from 100 devices: 70 real products in the committed catalog (50 FORIA + 20 non-FORIA industry references) plus 30 synthetic devices generated in-memory at training time (`augment_with_synthetic_devices`, deterministic with `seed=42`). The synthetic devices are NOT served by the production API — they only widen the feature distribution the model sees during training. See [Change history](#change-history) for the history of this split.

**Label generation** is deterministic and rule-based: `data/generate_dataset.py:compute_compatibility_label` computes a weighted score from the six engineered features and thresholds at 55 and 30 to produce the three-class label. Weights were initialized from expert heuristics documented in smart-home installation communities (Home Assistant forums, Matter Alliance documentation, manufacturer compatibility matrices) and are open to revision as user feedback accumulates. They have NOT been validated by a formal study with certified installers; that is future work (see `docs/DATASET.md > Labeling > Future work`).

**Label noise**: 5% of labels are randomly shifted by ±1 class (`add_noise`) to simulate the ~5% disagreement rate typically reported in informal surveys of installers on edge cases. This prevents the model from overfitting the rule boundary exactly.

**Class balance**: incompatible 32%, partial 36%, compatible 32%. More balanced than early versions of the dataset thanks to the Phase 1 cleanup.

## Relationship to the labeling rule

A reasonable question is: *if labels come from a deterministic rule, why train a ML model instead of just applying the rule at inference time?*

Three reasons:

1. **Swap path to supervised learning.** The rule is a bootstrap. Once installer feedback accumulates in the monitoring pipeline (`scripts/monitor.py` already ingests feedback and `scripts/retrain.py` merges it into the training set), labels will progressively shift from rule-generated to human-labeled. The ML model stays; only the training data changes. No inference-path rewrite needed.

2. **Soft outputs.** The API returns calibrated `probabilities` for all three classes plus a `confidence` score. A strict rule returns a hard label. Soft outputs let the frontend expose "99% compatible" vs "61% compatible" differently, and are what `confidence > 0.8` thresholds in [Evaluation](#evaluation) are measured against.

3. **Feature importance as an audit tool.** `models/metrics.json` publishes per-feature importance after every training run. This lets installers spot-check whether the model still relies on the features they expect to matter (currently `protocol_overlap` at 0.30, `hub_conflict` at 0.14, `ecosystem_overlap` at 0.12). A hardcoded rule offers no equivalent audit surface.

The model currently learns the rule with ~94% accuracy, and the 5% label noise is an upper bound on the residual disagreement it encodes. The expected evolution is: as real feedback accumulates, the model's predictions will diverge from the original rule on ambiguous pairs in favor of whatever installers agree on.

## Evaluation

Held-out test set (20%, stratified), plus 5-fold stratified cross-validation on the training portion.

| metric | value |
|---|---|
| accuracy | 0.94 |
| F1 macro | 0.94 |
| F1 weighted | 0.94 |
| CV F1 macro (5-fold) | 0.94 +/- 0.01 |
| mean prediction confidence | 0.92 |
| fraction with confidence > 0.8 | 0.93 |

See `models/metrics.json` for the live production numbers. Confusion matrix shape on the test set: errors are near-exclusively between adjacent classes (`partial` confused with `incompatible` or `compatible`); direct `incompatible`↔`compatible` confusion is 0 on the latest release.

## Limitations

- **Distribution**: trained on a catalog skewed toward the EU/MENA markets and toward mid-to-premium price tiers. Performance on budget no-name Shenzhen brands is untested.
- **Cold start**: a brand-new device category (not in `encoder.classes_`) falls back to a generic encoding. Expect confidence < 0.6; flag for installer review.
- **Matter bridges**: the feature set does not explicitly model Matter bridges. A pair where one side is a Matter bridge may be flagged `partial` when it's actually `compatible` through the bridge. Accepted trade-off while Matter adoption ramps.
- **Small dataset**: 4950 pairs is enough for stable F1 at this feature count, but does not support subgroup evaluation (see Fairness).
- **Rule-shaped decision boundary**: because labels come from a weighted-sum threshold rule, the model's decision boundary is approximately a hyperplane in feature space plus a 5% noise budget. It will not discover genuinely novel compatibility patterns until real user feedback enters the training set.

## Fairness and ethical considerations

- **No personal data**. Inputs are device specifications. No user identifiers, preferences, or PII reach the model.
- **Brand bias**: the model treats `same_brand` as a feature. Pairs within the same brand are correlated with `compatible` in the training data, which reflects reality (same-brand devices are designed to interoperate). A consequence is that a new, smaller brand may get systematically lower compatibility scores until enough cross-brand data accumulates. We monitor this via the installer feedback loop.
- **FORIA over-representation**: 50 of the 70 committed real devices are the FORIA house brand. Pair statistics are therefore dominated by FORIA-FORIA and FORIA-other combinations. In particular, `comfort` category is ~78% FORIA. Cross-brand predictions in `comfort` should be treated as lower-confidence until the non-FORIA `comfort` catalog grows.
- **Accessibility**: predictions include a plain-text `reason` field consumable by screen readers, and a discrete label consumable by users with color-vision deficiency. See `docs/ACCESSIBILITY.md`.
- **Environmental cost**: training is ~1 minute on a single CPU. Retraining runs weekly. Negligible carbon footprint relative to a single inference request on any cloud-hosted LLM.

## Monitoring and retraining

- Production predictions are logged to JSONL (`PREDICTION_LOG_PATH`), consumed by `scripts/monitor.py`.
- Drift detection uses Kolmogorov-Smirnov per feature (fallback) or Evidently (when installed). Alert if > 30% of features drift or if live accuracy vs installer feedback drops below 80%.
- Retraining runs every Sunday at 03:00 UTC via GitHub Actions. A new model is promoted only if F1 macro >= 0.85 and >= current production.
- Rollback: previous model artifacts are preserved as `model_<version>.pkl` in `models/`. Git revert of the weekly retrain commit restores them.

## Versioning

- **Model version**: timestamp-based (`YYYYMMDD_HHMMSS`), stored in `model_config.json`.
- **Code version**: git commit SHA of the repo at training time, stored in the Docker image tag deployed to Cloud Run.
- **Data version**: `compatibility_dataset.csv` is committed alongside the model in the retrain pipeline, so model-to-data traceability is git-native.

## Change history

### 2026-04-18 — Phase 2: Catalog factual corrections

Cross-checked the 20 real non-FORIA devices against manufacturer specs and reviews. Fixed 5 inaccuracies:

- **Sonos Era 300**: removed `google` from ecosystems (Sonos dropped Google Assistant support on the Era line at March 2023 launch, publicly acknowledged)
- **SwitchBot Curtain 3**: removed `wifi` from connectivity (no native Wi-Fi), set `hub_required` to true (voice/HomeKit integration requires a SwitchBot Hub)
- **Nanoleaf Shapes**: added `thread` to connectivity (Shapes controllers act as Thread Border Routers)
- **Eve Motion Sensor**: updated to current 2nd-generation specs — added `thread`, expanded ecosystems to `apple`+`google`+`alexa` via Matter, set `hub_required` to true for Thread Border Router requirement
- **FORIA Speaker Slave S530**: cleared ecosystems (Bluetooth-only slave has no direct voice-assistant integration; pairs with master speaker)

Metrics movement: accuracy 0.948 → 0.940, F1 macro 0.948 → 0.940, CV F1 macro stable at 0.941. The small drop reflects a genuinely harder labeling task after spurious ecosystem matches were removed; CV stability confirms the model was not overfitting on those artifacts. `hub_conflict` feature importance rose from 0.097 to 0.145, consistent with two additional hub-dependent products.

### 2026-04-18 — Phase 1: Train/serve leakage fix

Discovered that 30 synthetic devices generated by `augment_with_synthetic_devices` for training-time augmentation had been committed into `data/device_catalog.json` and were being served by the `/predict-catalog` production endpoint (users could receive recommendations for `Synthetic Device 7`). Also discovered duplicate-ID pollution in the training set: the synthetic devices were regenerated at training time with the same IDs as the leaked catalog entries, creating ~30 pairs with perfect feature overlap and trivially-correct `compatible` labels.

Fixes applied: removed the 30 synthetic entries from the committed catalog (70 real devices remaining), deleted a 200-line hardcoded `PRODUCTS` constant in `generate_dataset.py` that was dead code, regenerated `compatibility_dataset.csv` (8385 polluted pairs → 4950 clean pairs with better label balance).

Metrics movement: accuracy 0.873 → 0.948, F1 macro 0.868 → 0.948, CV F1 macro 0.904 → 0.941. The jump reflects the removal of spurious training pairs rather than any improvement in model generalization.

## Open questions and future work

- **Real installer validation**. Replace the bootstrap rule weights with weights learned from a pilot study on, say, 200 pairs labeled by 3+ independent installers. Target: run once the dataset grows past 1000 human-labeled pairs via the feedback loop.
- **Calibration check**. Are predicted confidences well-calibrated on the live distribution? Planned as part of the monthly monitoring report.
- **Per-ecosystem model**. Explore a separate head per dominant ecosystem (apple/google/alexa) once the dataset grows past 5000 pairs.
- **Matter-certified evaluation set**. Once `data/collectors/collect_matter_csa.py` is productionized, use CSA-certified Matter pairs as a held-out test set with human-audited labels.
