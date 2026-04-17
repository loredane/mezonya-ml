"""Schema validity test - the committed catalog MUST match DeviceSpec.

Runs in CI and catches drift between:
  * new devices added to data/device_catalog.json
  * DeviceSpec enums in api/main.py (Category, Connectivity, Ecosystem,
    CloudDependency)

Also guards the Option B cloud_compatibility feature: it only behaves
correctly if cloud_dependency stays within the 4-value enum.
"""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from api.main import DeviceSpec

CATALOG_PATH = Path(__file__).parent.parent / "data" / "device_catalog.json"


@pytest.fixture(scope="module")
def catalog() -> list[dict]:
    with open(CATALOG_PATH) as f:
        return json.load(f)


def test_catalog_not_empty(catalog):
    assert len(catalog) > 0, "device_catalog.json is empty"


def test_every_device_parses_as_DeviceSpec(catalog):
    """Every committed device must be a valid DeviceSpec. Fail loud on drift."""
    errors = []
    for i, device in enumerate(catalog):
        try:
            DeviceSpec(**device)
        except ValidationError as e:
            errors.append(
                f"device[{i}] id={device.get('id')} "
                f"name={device.get('name')!r}: {e}"
            )
    assert not errors, "Invalid devices in catalog:\n" + "\n".join(errors)


def test_cloud_dependency_within_enum(catalog):
    """Explicit guard for Option B - cloud_compatibility depends on this enum."""
    allowed = {"required", "optional", "local_only", "none"}
    offenders = [
        (d.get("id"), d.get("name"), d.get("cloud_dependency"))
        for d in catalog
        if d.get("cloud_dependency") not in allowed
    ]
    assert not offenders, f"Devices with invalid cloud_dependency: {offenders}"
