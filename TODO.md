# mezonya-ml - TODO / Refinement Backlog

Post-AIA-Bloc-4 follow-ups. Keep this list small and actionable.

## ML / Dataset

- [ ] Surface `cloud_dependency` reason in `/predict` response (extend
      `explain()` to mention local-only vs cloud-required friction
      explicitly)
- [ ] Calibrate Option B weights: the friction penalty currently reuses the
      same magnitude for both-required and local_only+required pairs.
      Consider splitting into two feature columns if the model treats them
      as distinct failure modes.
- [ ] Replace the `augment_with_synthetic_devices` cloud probabilities with
      the real frequency observed in the merged collectors output (today
      `none=0.05` is a guess).

## Schema / Typing

- [ ] Share enum literals between `api/main.py` and a top-level
      `mezonya/schema.py` so collectors import them instead of duplicating
      the value lists.
- [ ] Add a pre-commit hook that runs
      `pytest tests/test_catalog_schema_validity.py` on any change to
      `data/device_catalog.json`.

## CI / Deploy

- [ ] Move the committed catalog out of the Docker image into a GCS bucket -
      avoids rebuilding the image when only the catalog changes.
- [ ] Add a scheduled CI job that runs the collectors and opens a PR when
      the merged catalog changes (today the catalog is refreshed manually).

## Frontend (repo separe mezonya.com)

- [ ] Wire `/predict-catalog` to the product page - show the top-5
      compatible devices for the currently-viewed FORIA / partner product.
- [ ] Accessibility: use `reason` (plain-text) as aria-label on the compat
      badge.
