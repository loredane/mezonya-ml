"""Fusionne les sources (Home Assistant + Matter CSA + catalogue Mezonya original)
et produit un catalogue unifié prêt pour la génération de paires de compatibilité.

Pipeline :
    1. Charger les 3 sources (HA, Matter, Mezonya original)
    2. Dédupliquer (fuzzy matching sur brand+name)
    3. Enrichir croisé (un produit dans HA ET Matter to fusion des specs)
    4. Valider les champs obligatoires
    5. Exporter le catalogue final

Usage :
    python data/collectors/merge_catalogs.py

Output :
    data/processed/device_catalog_extended.json  (1000+ devices)
    data/processed/catalog_stats.json            (stats de provenance)
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional


DATA_DIR = Path(__file__).parent.parent
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

SOURCES = {
    "mezonya": DATA_DIR / "device_catalog.json",  # Catalogue original (20 produits)
    "home_assistant": RAW_DIR / "home_assistant_devices.json",
    "matter_csa": RAW_DIR / "matter_csa_products.json",
}

OUTPUT_CATALOG = PROCESSED_DIR / "device_catalog_extended.json"
OUTPUT_STATS = PROCESSED_DIR / "catalog_stats.json"

REQUIRED_FIELDS = ["name", "brand", "category", "connectivity"]


def load_source(name: str, path: Path) -> list[dict]:
    """Charge une source JSON avec gestion d'erreur."""
    if not path.exists():
        print(f"[merge]   Source '{name}' not found at {path} - skipping")
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[merge]  Invalid JSON in {path}: {e}")
        return []

    if not isinstance(data, list):
        # Certains exports sont {"devices": [...]} - on normalise
        data = data.get("devices", []) if isinstance(data, dict) else []

    print(f"[merge] Loaded {name}: {len(data)} devices")
    return data


def normalize_brand(brand: str) -> str:
    """Normalise une marque pour le matching (lowercase, sans espaces)."""
    if not brand:
        return ""
    brand = brand.lower().strip()
    # Unifier les variantes communes
    brand_aliases = {
        "philips hue": "philips",
        "signify": "philips",
        "amazon echo": "amazon",
        "ring (amazon)": "ring",
        "google nest": "google",
        "nest": "google",
    }
    return brand_aliases.get(brand, brand)


def normalize_name(name: str) -> str:
    """Normalise un nom pour le matching."""
    if not name:
        return ""
    # Retirer ponctuation, parenthèses, mots génériques
    name = re.sub(r"[^\w\s]", " ", name.lower())
    name = re.sub(r"\b(integration|plugin|device|smart|home)\b", "", name)
    return " ".join(name.split())


def make_device_key(device: dict) -> str:
    """Clé de déduplication : brand_name normalisé."""
    brand = normalize_brand(device.get("brand", ""))
    name = normalize_name(device.get("name", ""))
    return f"{brand}__{name}"


VALID_CLOUD_DEPENDENCY = frozenset({"required", "optional", "local_only", "none"})


def _normalize_cloud_dependency(value, device_name: str = "") -> str:
    """Coerce any cloud_dependency value to the 4-value enum.

    Legacy values ('unknown', None, typos) fall back to 'none' with a warning
    so downstream schema validation stays green while flagging data-quality
    issues in the collector output.
    """
    if value in VALID_CLOUD_DEPENDENCY:
        return value
    import logging
    logging.getLogger("merge_catalogs").warning(
        "invalid cloud_dependency=%r for %r, defaulting to 'none'", value, device_name
    )
    return "none"


def unify_device_schema(device: dict) -> dict:
    """Force le schéma Mezonya standard sur un appareil de n'importe quelle source."""
    return {
        "id": device.get("id") or device.get("source_id") or make_device_key(device),
        "name": device.get("name", "Unknown").strip(),
        "brand": device.get("brand", "Unknown").strip(),
        "category": device.get("category", "other"),
        "connectivity": sorted(set(device.get("connectivity", []))),
        "ecosystems": sorted(set(device.get("ecosystems", []))),
        "hub_required": bool(device.get("hub_required", False)),
        "cloud_dependency": _normalize_cloud_dependency(
            device.get("cloud_dependency"), device.get("name", "")
        ),
        "source": device.get("source", "mezonya"),
        "metadata": {
            "matter_version": device.get("matter_version"),
            "iot_class": device.get("iot_class"),
            "quality_scale": device.get("quality_scale"),
            "doc_url": device.get("doc_url"),
            "certification_id": device.get("certification_id"),
        },
    }


# DEDUPLICATION + FUSION
def merge_device_specs(a: dict, b: dict) -> dict:
    """Fusionne 2 appareils identifiés comme étant le même produit.

    Logique :
    - Union des connectivity et ecosystems (plus d'infos = mieux)
    - Priorité Matter CSA > Mezonya original > Home Assistant pour les champs simples
      (car Matter = certifié officiellement, Mezonya = validé manuellement)
    """
    priority = {"matter_csa": 3, "mezonya": 2, "home_assistant": 1}
    primary, secondary = (a, b) if priority.get(a["source"], 0) >= priority.get(b["source"], 0) else (b, a)

    merged = dict(primary)
    merged["connectivity"] = sorted(set(a["connectivity"]) | set(b["connectivity"]))
    merged["ecosystems"] = sorted(set(a["ecosystems"]) | set(b["ecosystems"]))
    merged["hub_required"] = a["hub_required"] or b["hub_required"]

    # Fusion metadata (les clés non-None du primary gagnent)
    merged["metadata"] = {**secondary["metadata"], **{k: v for k, v in primary["metadata"].items() if v is not None}}

    # Traçabilité : on note toutes les sources
    merged["sources_merged"] = sorted({a["source"], b["source"]})

    return merged


def deduplicate(devices: list[dict]) -> list[dict]:
    """Dédupliquer par brand+name, en fusionnant les specs."""
    by_key: dict[str, dict] = {}
    duplicates_count = 0

    for device in devices:
        key = make_device_key(device)
        if not key or key == "__":
            continue  # Skip entrées sans nom ni marque

        if key in by_key:
            by_key[key] = merge_device_specs(by_key[key], device)
            duplicates_count += 1
        else:
            by_key[key] = device

    print(f"[merge] Deduplicated: {duplicates_count} duplicates fused")
    return list(by_key.values())


def is_valid_device(device: dict) -> bool:
    """Validation qualité minimum pour entrer dans le catalogue final."""
    for field in REQUIRED_FIELDS:
        if not device.get(field):
            return False

    # Au moins 1 protocole de connectivité connu
    known_protocols = {"zigbee", "zwave", "matter", "wifi", "bluetooth", "ethernet", "thread"}
    if not any(p in known_protocols for p in device["connectivity"]):
        return False

    return True


def compute_stats(devices: list[dict]) -> dict:
    """Calcule les statistiques du catalogue final."""
    return {
        "total_devices": len(devices),
        "by_source": dict(Counter(d["source"] for d in devices)),
        "by_category": dict(Counter(d["category"] for d in devices)),
        "by_connectivity": dict(Counter(
            p for d in devices for p in d["connectivity"]
        )),
        "by_ecosystem": dict(Counter(
            e for d in devices for e in d["ecosystems"]
        )),
        "by_cloud_dependency": dict(Counter(d["cloud_dependency"] for d in devices)),
        "hub_required_ratio": sum(1 for d in devices if d["hub_required"]) / max(len(devices), 1),
        "multi_source_devices": sum(1 for d in devices if "sources_merged" in d),
    }


def main():
    print("[merge] Starting catalog merge pipeline")
    print("=" * 60)

    # 1. Charger toutes les sources
    all_devices = []
    for name, path in SOURCES.items():
        raw = load_source(name, path)
        # Unifier le schéma
        unified = [unify_device_schema({**d, "source": d.get("source", name)}) for d in raw]
        all_devices.extend(unified)

    print(f"[merge] Total before dedup: {len(all_devices)}")

    # 2. Déduplication avec fusion
    deduplicated = deduplicate(all_devices)
    print(f"[merge] Total after dedup: {len(deduplicated)}")

    # 3. Validation
    valid = [d for d in deduplicated if is_valid_device(d)]
    invalid_count = len(deduplicated) - len(valid)
    print(f"[merge] Valid devices: {len(valid)} (filtered {invalid_count} invalid)")

    # 4. Sauvegarde
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CATALOG.write_text(
        json.dumps(valid, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    stats = compute_stats(valid)
    OUTPUT_STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("=" * 60)
    print(f"[merge]  Final catalog: {OUTPUT_CATALOG}")
    print(f"[merge] Stats: {OUTPUT_STATS}")
    print(f"\n[merge] Summary:")
    print(f"Total devices:    {stats['total_devices']}")
    print(f"Sources:          {stats['by_source']}")
    print(f"Top categories:   {dict(list(Counter(stats['by_category']).most_common(5)))}")
    print(f"Top protocols:    {dict(list(Counter(stats['by_connectivity']).most_common(5)))}")
    print(f"Multi-source:     {stats['multi_source_devices']} devices (enriched from 2+ sources)")


if __name__ == "__main__":
    main()
