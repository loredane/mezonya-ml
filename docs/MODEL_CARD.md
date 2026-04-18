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

1225 device pairs built from a 50-device catalog (20 real Mezonya products + 30 synthetic edge-case devices). Labels come from a deterministic rule engine validated by certified smart-home installers, with 5% random label noise to simulate installer disagreement.

Class balance: 35% incompatible, 44% partial, 21% compatible. Dataset is small by design; feedback loop from production expected to grow it to ~3000 pairs in 12-18 months.

## Evaluation

Held-out test set (20%, stratified), plus 5-fold stratified cross-validation on the training portion.

| metric | value |
|---|---|
| accuracy | 0.87 |
| F1 macro | 0.87 |
| F1 weighted | 0.87 |
| CV F1 macro (5-fold) | 0.90 +/- 0.02 |
| mean prediction confidence | 0.93 |
| fraction with confidence > 0.8 | 0.90 |

See `models/metrics.json` for the live production numbers.

## Limitations

- **Distribution**: trained on a catalog skewed toward the EU/MENA markets and toward mid-to-premium price tiers. Performance on budget no-name Shenzhen brands is untested.
- **Cold start**: a brand-new device category (not in `encoder.classes_`) falls back to a generic encoding. Expect confidence < 0.6; flag for installer review.
- **Matter bridges**: the feature set does not explicitly model Matter bridges. A pair where one side is a Matter bridge may be flagged `partial` when it's actually `compatible` through the bridge. Accepted trade-off while Matter adoption ramps.
- **Small dataset**: 4950 pairs is enough for stable F1 at this feature count, but does not support subgroup evaluation (see Fairness).

## Fairness and ethical considerations

- **No personal data**. Inputs are device specifications. No user identifiers, preferences, or PII reach the model.
- **Brand bias**: the model treats `same_brand` as a feature. Pairs within the same brand are correlated with `compatible` in the training data, which reflects reality (same-brand devices are designed to interoperate). A consequence is that a new, smaller brand may get systematically lower compatibility scores until enough cross-brand data accumulates. We monitor this via the installer feedback loop.
- **Accessibility**: predictions include a plain-text `reason` field consumable by screen readers, and a discrete label consumable by users with color-vision deficiency. See `docs/ACCESSIBILITY.md`.
- **Environmental cost**: training is ~1 minute on a single CPU. Retraining runs weekly. Negligible carbon footprint relative to a single inference request on any cloud-hosted LLM.

## Monitoring and retraining

- Production predictions are logged to JSONL (`PREDICTION_LOG_PATH`), consumed by `scripts/monitor.py`.
- Drift detection uses Kolmogorov-Smirnov per feature (fallback) or Evidently (when installed). Alert if > 30% of features drift or if live accuracy vs installer feedback drops below 80%.
- Retraining runs every Sunday at 03:00 UTC via GitHub Actions. A new model is promoted only if F1 macro >= 0.85 and >= current production.
- Rollback: previous model artifacts are preserved as `model_backup_<version>.pkl` in the same directory. Git revert of the weekly retrain commit restores them.

## Versioning

- **Model version**: timestamp-based (`YYYYMMDD_HHMMSS`), stored in `model_config.json`.
- **Code version**: git commit SHA of the repo at training time, stored in the Docker image tag deployed to Cloud Run.
- **Data version**: the `compatibility_dataset.csv` is committed alongside the model in the retrain pipeline, so model-to-data traceability is git-native.

## Open questions and future work

- Add a calibration check: are predicted confidences well-calibrated on the live distribution?
- Explore a per-ecosystem model (separate head for apple/google/alexa dominant pairs) once the dataset grows past 5000 pairs.
- Evaluate on a held-out set of Matter-certified pairs once the CSA data pipeline is productionized (`data/collectors/collect_matter_csa.py`).
