"""Clone le repo home-assistant.io (documentation officielle) et parse les
fichiers markdown des intégrations pour en extraire les métadonnées
d'appareils smart home.

Source : https://github.com/home-assistant/home-assistant.io
Licence : CC-BY (données publiques)
Volume attendu : ~1500-2000 intégrations dont ~800-1200 appareils physiques

Usage :
    python data/collectors/collect_home_assistant.py

Output :
    data/raw/home_assistant_devices.json
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Optional

import yaml


HA_REPO_URL = "https://github.com/home-assistant/home-assistant.io.git"
CLONE_DIR = Path("/tmp/home-assistant-io")
OUTPUT_FILE = Path(__file__).parent.parent / "raw" / "home_assistant_devices.json"

# Catégories Home Assistant qui nous intéressent (appareils physiques smart home)
RELEVANT_CATEGORIES = {
    "Alarm",
    "Binary Sensor",
    "Camera",
    "Climate",
    "Cover",
    "Doorbell",
    "Fan",
    "Hub",
    "Humidifier",
    "Light",
    "Lock",
    "Media Player",
    "Notifications",
    "Presence Detection",
    "Remote",
    "Sensor",
    "Switch",
    "Vacuum",
    "Water Heater",
}

# Mapping iot_class to connectivity protocols
# Source : https://developers.home-assistant.io/docs/core/integration-quality-scale/
IOT_CLASS_TO_PROTOCOL = {
    "local_push": ["local"],
    "local_polling": ["local"],
    "cloud_push": ["cloud"],
    "cloud_polling": ["cloud"],
    "assumed_state": ["local"],
    "calculated": [],
}

# Détection de protocoles par mots-clés dans le contenu
PROTOCOL_KEYWORDS = {
    "zigbee": ["zigbee", "zha"],
    "zwave": ["z-wave", "zwave"],
    "matter": ["matter", "thread"],
    "wifi": ["wi-fi", "wifi"],
    "bluetooth": ["bluetooth", "ble"],
    "ethernet": ["ethernet", "lan", "poe"],
}

# Détection d'écosystèmes
ECOSYSTEM_KEYWORDS = {
    "apple": ["apple home", "homekit", "siri"],
    "google": ["google home", "google assistant"],
    "alexa": ["alexa", "amazon echo"],
    "smartthings": ["smartthings"],
}

# Catégorisation Mezonya
CATEGORY_MAPPING = {
    "Light": "lighting",
    "Switch": "lighting",
    "Camera": "security",
    "Doorbell": "security",
    "Lock": "security",
    "Alarm": "security",
    "Climate": "climate",
    "Fan": "climate",
    "Humidifier": "climate",
    "Water Heater": "climate",
    "Cover": "climate",
    "Vacuum": "appliance",
    "Media Player": "entertainment",
    "Remote": "entertainment",
    "Sensor": "sensor",
    "Binary Sensor": "sensor",
    "Presence Detection": "sensor",
    "Hub": "hub",
    "Notifications": "notification",
}


# CLONE/UPDATE HA REPO
def clone_or_update_repo() -> Path:
    """Clone le repo HA s'il n'existe pas, sinon le met à jour."""
    if CLONE_DIR.exists():
        print(f"[HA] Updating existing repo at {CLONE_DIR}...")
        subprocess.run(
            ["git", "-C", str(CLONE_DIR), "pull", "--quiet"],
            check=True,
            timeout=120,
        )
    else:
        print(f"[HA] Cloning {HA_REPO_URL} (shallow)...")
        subprocess.run(
            [
                "git", "clone",
                "--depth", "1",
                "--filter=blob:none",
                HA_REPO_URL,
                str(CLONE_DIR),
            ],
            check=True,
            timeout=300,
        )

    integrations_dir = CLONE_DIR / "source" / "_integrations"
    if not integrations_dir.exists():
        raise RuntimeError(f"Expected integrations folder at {integrations_dir}")
    return integrations_dir


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extrait le frontmatter YAML d'un fichier markdown Jekyll."""
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    try:
        frontmatter = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        frontmatter = {}

    body = parts[2]
    return frontmatter, body


def detect_protocols(text: str, iot_class: Optional[str]) -> list[str]:
    """Détecte les protocoles de connectivité."""
    text_lower = text.lower()
    protocols = set()

    for protocol, keywords in PROTOCOL_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            protocols.add(protocol)

    # Si iot_class contient cloud to wifi par défaut (appareils IoT modernes)
    if iot_class and "cloud" in iot_class and not protocols:
        protocols.add("wifi")

    return sorted(protocols)


def detect_ecosystems(text: str) -> list[str]:
    """Détecte les écosystèmes supportés."""
    text_lower = text.lower()
    ecosystems = set()

    for ecosystem, keywords in ECOSYSTEM_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            ecosystems.add(ecosystem)

    return sorted(ecosystems)


def detect_cloud_dependency(iot_class: Optional[str], text: str) -> str:
    """Détermine la dépendance cloud : required / optional / none."""
    if not iot_class:
        return "unknown"

    if "cloud" in iot_class:
        # Vérifier si local est aussi supporté
        if "local" in text.lower() and "api" in text.lower():
            return "optional"
        return "required"

    if "local" in iot_class:
        return "none"

    return "unknown"


def detect_hub_required(text: str, iot_class: Optional[str]) -> bool:
    """Détermine si un hub est requis (bridge, gateway, coordinator)."""
    text_lower = text.lower()
    hub_indicators = ["bridge required", "hub required", "coordinator", "gateway required"]
    return any(ind in text_lower for ind in hub_indicators)


def map_category(ha_categories: list[str]) -> str:
    """Mappe les catégories HA vers notre taxonomie Mezonya."""
    for ha_cat in ha_categories:
        if ha_cat in CATEGORY_MAPPING:
            return CATEGORY_MAPPING[ha_cat]
    return "other"


def is_physical_device(frontmatter: dict) -> bool:
    """Filtre les intégrations qui correspondent à des appareils physiques."""
    # Exclure les services/API purs
    ha_categories = frontmatter.get("ha_category", [])
    if isinstance(ha_categories, str):
        ha_categories = [ha_categories]

    # Au moins une catégorie doit être dans notre whitelist
    return any(cat in RELEVANT_CATEGORIES for cat in ha_categories)


def parse_integration_file(md_file: Path) -> Optional[dict]:
    """Parse un fichier markdown d'intégration HA."""
    try:
        content = md_file.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None

    frontmatter, body = parse_frontmatter(content)

    if not frontmatter:
        return None

    # Filtrer les non-appareils (services web, API, etc.)
    if not is_physical_device(frontmatter):
        return None

    # Nom et marque
    title = frontmatter.get("title", "")
    domain = md_file.stem  # ex: "philips_hue"

    ha_categories = frontmatter.get("ha_category", [])
    if isinstance(ha_categories, str):
        ha_categories = [ha_categories]

    iot_class = frontmatter.get("ha_iot_class", "")
    if isinstance(iot_class, list):
        iot_class = iot_class[0] if iot_class else ""

    # Combine frontmatter et body pour détection de protocoles
    full_text = f"{title}\n{body[:3000]}"  # limiter le body pour perf

    device = {
        "source": "home_assistant",
        "source_id": domain,
        "name": title,
        "brand": title.split()[0] if title else domain.split("_")[0].title(),
        "category": map_category(ha_categories),
        "ha_categories": ha_categories,
        "iot_class": iot_class,
        "connectivity": detect_protocols(full_text, iot_class),
        "ecosystems": detect_ecosystems(full_text),
        "hub_required": detect_hub_required(full_text, iot_class),
        "cloud_dependency": detect_cloud_dependency(iot_class, full_text),
        "quality_scale": frontmatter.get("ha_quality_scale"),
        "codeowners": frontmatter.get("ha_codeowners", []),
        "doc_url": f"https://www.home-assistant.io/integrations/{domain}/",
    }

    return device


def main():
    integrations_dir = clone_or_update_repo()

    md_files = list(integrations_dir.glob("*.markdown"))
    print(f"[HA] Found {len(md_files)} integration files")

    devices = []
    skipped = 0

    for md_file in md_files:
        device = parse_integration_file(md_file)
        if device is None:
            skipped += 1
            continue

        # Qualité minimum : au moins une info de connectivité
        if not device["connectivity"]:
            skipped += 1
            continue

        devices.append(device)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(devices, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[HA]  Extracted {len(devices)} physical devices (skipped {skipped})")
    print(f"[HA] Output: {OUTPUT_FILE}")

    # Stats
    categories = {}
    for d in devices:
        categories[d["category"]] = categories.get(d["category"], 0) + 1
    print(f"[HA] Categories breakdown:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"       {cat:15s} {count:4d}")


if __name__ == "__main__":
    main()
