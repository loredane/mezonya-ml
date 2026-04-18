"""Génère un dataset de paires d'appareils avec labels de compatibilité
à partir du catalogue réel Mezonya et de règles expertes validées
par des installateurs certifiés.

Labels:
  2 = compatible
  1 = partiellement compatible
  0 = incompatible
"""

import pandas as pd
import numpy as np
from itertools import combinations
import json
import os

# Catalog source of truth: data/device_catalog.json (loaded by _load_committed_catalog).
# The previously hardcoded PRODUCTS list has been removed; catalog ingestion now flows
# exclusively through data/collectors/merge_catalogs.py -> device_catalog.json.

# Protocoles et fréquences de communication
PROTOCOL_FREQUENCIES = {
    "zigbee": "2.4GHz",
    "wifi": "2.4GHz/5GHz",
    "bluetooth": "2.4GHz",
    "zwave": "908MHz",
    "matter": "varies",
    "thread": "2.4GHz",
}

# Règles expertes de compatibilité (validées par installateurs)
def compute_protocol_overlap(conn_a: list, conn_b: list) -> float:
    """Ratio de protocoles partagés entre deux appareils."""
    set_a, set_b = set(conn_a), set(conn_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_ecosystem_overlap(eco_a: list, eco_b: list) -> float:
    """Ratio d'écosystèmes partagés (Apple, Google, Alexa)."""
    set_a, set_b = set(eco_a), set(eco_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def same_brand(brand_a: str, brand_b: str) -> int:
    """Même marque = meilleure intégration probable."""
    return 1 if brand_a == brand_b else 0


def hub_conflict(hub_a: bool, hub_b: bool, conn_a: list, conn_b: list) -> int:
    """
    Conflit potentiel si un appareil nécessite un hub 
    et l'autre ne partage pas le protocole du hub.
    """
    if hub_a and not hub_b:
        # Device A needs hub (zigbee) - does B speak zigbee?
        if "zigbee" in conn_a and "zigbee" not in conn_b:
            return 1
    if hub_b and not hub_a:
        if "zigbee" in conn_b and "zigbee" not in conn_a:
            return 1
    return 0


def cloud_compatibility(cloud_a: str, cloud_b: str) -> int:
    """Feature Option B - flag cloud-path friction between a device pair.

    Returns 0 (friction) when:
        * both devices require their own cloud (two apps, no shared orchestration)
        * one is local_only and the other requires cloud (no shared control plane)
    Returns 1 otherwise. Mirrors api/main.py:_cloud_compatible at inference time.
    """
    pair = frozenset({cloud_a, cloud_b})
    if pair == frozenset({"required"}):
        return 0
    if pair == frozenset({"local_only", "required"}):
        return 0
    return 1


def category_synergy(cat_a: str, cat_b: str) -> float:
    """
    Certaines catégories fonctionnent naturellement ensemble
    (ex: security + security, lighting + comfort).
    """
    synergy_map = {
        frozenset({"security", "security"}): 1.0,
        frozenset({"lighting", "lighting"}): 0.9,
        frozenset({"security", "lighting"}): 0.8,
        frozenset({"security", "energy"}): 0.7,
        frozenset({"climate", "energy"}): 0.9,
        frozenset({"comfort", "lighting"}): 0.8,
        frozenset({"comfort", "climate"}): 0.7,
        frozenset({"audio", "comfort"}): 0.6,
        frozenset({"audio", "lighting"}): 0.7,
        frozenset({"access", "security"}): 0.9,
    }
    key = frozenset({cat_a, cat_b})
    return synergy_map.get(key, 0.4)


def compute_compatibility_label(features: dict) -> int:
    """
    Détermine le label de compatibilité basé sur les règles expertes.
    Score pondéré to seuils de classification.
    
    Returns: 2 (compatible), 1 (partial), 0 (incompatible)
    """
    score = 0.0
    # Protocol overlap est le facteur le plus critique
    score += features["protocol_overlap"] * 35
    # Ecosystem overlap détermine si les appareils se voient
    score += features["ecosystem_overlap"] * 30
    # Même marque = intégration native
    score += features["same_brand"] * 10
    # Conflit de hub = friction
    score -= features["hub_conflict"] * 15
    # Cloud compatibility
    score += features["cloud_compatible"] * 10
    # Catégorie synergy
    score += features["category_synergy"] * 10

    if score >= 55:
        return 2  # compatible
    elif score >= 30:
        return 1  # partiellement compatible
    else:
        return 0  # incompatible
def generate_pair_features(dev_a: dict, dev_b: dict) -> dict:
    """Calcule toutes les features pour une paire d'appareils."""
    protocol_overlap = compute_protocol_overlap(dev_a["connectivity"], dev_b["connectivity"])
    ecosystem_overlap = compute_ecosystem_overlap(dev_a["ecosystems"], dev_b["ecosystems"])
    brand_same = same_brand(dev_a["brand"], dev_b["brand"])
    hub_conf = hub_conflict(dev_a["hub_required"], dev_b["hub_required"],
                            dev_a["connectivity"], dev_b["connectivity"])
    cloud_compat = cloud_compatibility(dev_a["cloud_dependency"], dev_b["cloud_dependency"])
    cat_syn = category_synergy(dev_a["category"], dev_b["category"])

    # Features supplémentaires
    both_hub_required = int(dev_a["hub_required"] and dev_b["hub_required"])
    total_protocols = len(set(dev_a["connectivity"]) | set(dev_b["connectivity"]))
    total_ecosystems = len(set(dev_a["ecosystems"]) | set(dev_b["ecosystems"]))

    features = {
        "device_a_id": dev_a["id"],
        "device_b_id": dev_b["id"],
        "protocol_overlap": round(protocol_overlap, 4),
        "ecosystem_overlap": round(ecosystem_overlap, 4),
        "same_brand": brand_same,
        "hub_conflict": hub_conf,
        "cloud_compatible": cloud_compat,
        "category_synergy": round(cat_syn, 4),
        "both_hub_required": both_hub_required,
        "total_protocols": total_protocols,
        "total_ecosystems": total_ecosystems,
        # Features catégorielles encodées
        "device_a_category": dev_a["category"],
        "device_b_category": dev_b["category"],
        "device_a_hub_required": int(dev_a["hub_required"]),
        "device_b_hub_required": int(dev_b["hub_required"]),
        "device_a_cloud_required": int(dev_a["cloud_dependency"] == "required"),
        "device_b_cloud_required": int(dev_b["cloud_dependency"] == "required"),
    }

    features["compatibility_label"] = compute_compatibility_label(features)
    return features


def augment_with_synthetic_devices(n_synthetic: int = 30, seed: int = 42) -> list:
    """
    Augmente le catalogue avec des appareils synthétiques réalistes
    pour enrichir le dataset d'entraînement (simule l'arrivée de 
    nouveaux appareils dans le catalogue).
    """
    rng = np.random.RandomState(seed)
    brands = ["Aqara", "Philips", "Ecobee", "Yale", "Sonos", "SwitchBot",
              "Nanoleaf", "Tado", "Ring", "Eve", "Yeelight", "Eufy",
              "Meross", "SONOFF", "Tuya", "Shelly", "TP-Link", "Xiaomi"]
    categories = ["security", "lighting", "climate", "audio", "comfort", "energy", "access"]
    protocols = ["wifi", "zigbee", "bluetooth", "zwave", "matter", "thread"]
    ecos = ["apple", "google", "alexa"]
    # Option B - 4-value enum to match the real catalog distribution.
    clouds = ["optional", "required", "local_only", "none"]
    clouds_p = [0.45, 0.30, 0.20, 0.05]

    synthetic = []
    for i in range(n_synthetic):
        n_proto = rng.choice([1, 2, 3], p=[0.4, 0.45, 0.15])
        n_eco = rng.choice([1, 2, 3], p=[0.2, 0.3, 0.5])
        synthetic.append({
            "id": 100 + i,
            "name": f"Synthetic Device {i}",
            "brand": rng.choice(brands),
            "category": rng.choice(categories),
            "connectivity": list(rng.choice(protocols, size=n_proto, replace=False)),
            "ecosystems": list(rng.choice(ecos, size=n_eco, replace=False)),
            "hub_required": bool(rng.choice([True, False], p=[0.3, 0.7])),
            "cloud_dependency": str(rng.choice(clouds, p=clouds_p)),
        })
    return synthetic


def add_noise(df: pd.DataFrame, noise_rate: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """
    Ajoute du bruit réaliste pour simuler les désaccords entre installateurs
    sur les labels de compatibilité (~5% de désaccord dans le monde réel).
    """
    rng = np.random.RandomState(seed)
    n_noisy = int(len(df) * noise_rate)
    noisy_idx = rng.choice(df.index, size=n_noisy, replace=False)
    for idx in noisy_idx:
        current = df.loc[idx, "compatibility_label"]
        # Shift par +/-1 (un installateur peut diverger d'un cran)
        if current == 2:
            df.loc[idx, "compatibility_label"] = 1
        elif current == 0:
            df.loc[idx, "compatibility_label"] = 1
        else:
            df.loc[idx, "compatibility_label"] = rng.choice([0, 2])
    return df


def _load_committed_catalog() -> list[dict]:
    """Load data/device_catalog.json - the single source of truth for devices.

    The catalog is produced offline by data/collectors/merge_catalogs.py and
    committed to the repo. generate_dataset.py READS it (never rewrites it)
    so training tracks the real product population - including FORIA and
    every other brand added via collectors.
    """
    catalog_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "device_catalog.json"
    )
    with open(catalog_path) as f:
        return json.load(f)


def generate_dataset(augment: bool = True, noise: bool = True) -> pd.DataFrame:
    """Pipeline complet de génération du dataset.

    Source de vérité = data/device_catalog.json (committé). augment ajoute des
    appareils synthétiques uniquement pour l'entraînement (pas le catalogue API).
    """
    all_devices = _load_committed_catalog()
    if augment:
        all_devices.extend(augment_with_synthetic_devices())

    print(f"catalog: {len(all_devices)} devices")
    print(f"possible pairs: {len(all_devices) * (len(all_devices) - 1) // 2}")

    rows = []
    for dev_a, dev_b in combinations(all_devices, 2):
        rows.append(generate_pair_features(dev_a, dev_b))

    df = pd.DataFrame(rows)

    if noise:
        df = add_noise(df)

    print(f"generated: {len(df)} pairs")
    print(f"label distribution:")
    for label, count in df["compatibility_label"].value_counts().sort_index().items():
        labels_map = {0: "incompatible", 1: "partial", 2: "compatible"}
        print(f"   {labels_map[label]}: {count} ({count/len(df)*100:.1f}%)")

    return df


if __name__ == "__main__":
    df = generate_dataset()

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "compatibility_dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"\nsaved: {output_path}")
    # NOTE: device_catalog.json is read-only here. It is the committed source
    # of truth, produced offline by data/collectors/merge_catalogs.py.
