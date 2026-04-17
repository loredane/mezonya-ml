# Mezonya Device Catalog - Collectors

Pipeline de collecte automatique du catalogue d'appareils smart home pour
l'entraînement du modèle de compatibilité.

## Stratégie

Combinaison de **2 sources publiques** de haute qualité :

| Source | Méthode | Volume attendu | Légitimité |
|--------|---------|----------------|------------|
| **Home Assistant** docs repo | Clone Git + parse Markdown | ~800-1200 appareils | Repo public open source (Apache 2.0) |
| **Matter CSA** certified products | HTTP scraping respectueux | ~300-500 appareils | Données publiques de certification officielle |
| **Mezonya** catalogue original | Chargement JSON local | 20 appareils | Source interne |

Après déduplication et fusion : **~1000-1500 appareils uniques** dans le catalogue final.

## Conformité et éthique (défense devant le jury)

| Aspect | Comment c'est géré |
|--------|---------------------|
| Respect des CGU | Home Assistant = Apache 2.0 ; CSA = données publiques |
| Robots.txt | Vérifié avant scraping Matter CSA |
| Rate limiting | 2s entre chaque requête (configurable) |
| User-Agent identifiable | `Mezonya-Academic-Research/1.0` avec contact |
| Cache 24h | Évite le re-scraping inutile |
| Données personnelles | Aucune (uniquement specs produits) |
| Traçabilité | Chaque device garde son `source` + `doc_url` |

## Installation

```bash
pip install requests beautifulsoup4 pyyaml
# Git doit être installé (pour cloner le repo Home Assistant)
```

## Utilisation

### Tout lancer d'un coup (recommandé)

```bash
python data/collectors/run_all.py
```

Durée estimée :
- Home Assistant : ~2-3 min (clone + parse)
- Matter CSA : ~5-10 min (50 pages × 2s rate limit)
- Merge : instantané

### Étapes individuelles

```bash
# Juste Home Assistant
python data/collectors/collect_home_assistant.py

# Juste Matter (sans fetch des pages détail)
python data/collectors/collect_matter_csa.py --max-pages 50

# Juste Matter avec détails (beaucoup plus long, 10x plus lent)
python data/collectors/collect_matter_csa.py --max-pages 50 --fetch-details

# Juste le merge
python data/collectors/merge_catalogs.py
```

## Sorties

```
data/
├── raw/
│   ├── home_assistant_devices.json    # Export brut HA
│   └── matter_csa_products.json       # Export brut Matter
├── processed/
│   ├── device_catalog_extended.json   # Catalogue final (to entrée de generate_dataset.py)
│   └── catalog_stats.json             # Stats de provenance
└── device_catalog.json                # Catalogue Mezonya original (inchangé)
```

## Schéma de sortie (device_catalog_extended.json)

```json
{
  "id": "philips__hue_white_ambiance",
  "name": "Philips Hue White Ambiance",
  "brand": "Philips",
  "category": "lighting",
  "connectivity": ["zigbee", "bluetooth"],
  "ecosystems": ["apple", "google", "alexa"],
  "hub_required": true,
  "cloud_dependency": "optional",
  "source": "matter_csa",
  "sources_merged": ["home_assistant", "matter_csa"],
  "metadata": {
    "matter_version": "1.2",
    "iot_class": "local_push",
    "quality_scale": "platinum",
    "doc_url": "https://www.home-assistant.io/integrations/hue/",
    "certification_id": "CSA2024-XXXX"
  }
}
```

## Intégration avec generate_dataset.py

Dans `data/generate_dataset.py`, remplace :

```python
# Avant : catalogue de 20 produits
CATALOG_PATH = Path(__file__).parent / "device_catalog.json"
```

Par :

```python
# Après : catalogue étendu de 1000+ produits
CATALOG_PATH = Path(__file__).parent / "processed" / "device_catalog_extended.json"
```

Le reste du pipeline (génération de paires, features, labeling) fonctionne **sans modification** car le schéma est identique.

## Volume de paires attendu

- 20 appareils to 20² = **400 paires** (ancien dataset)
- 1000 appareils to 1000² = **1 000 000 paires** (potentiel)

**Attention** : générer 1M de paires = mémoire explosée + entraînement lent.

Stratégie d'échantillonnage (à ajouter dans `generate_dataset.py`) :
- Sampling stratifié : garder toutes les paires dans la même catégorie + 20% des paires inter-catégories
- Objectif : ~50 000 paires pour un modèle robuste sans exploser les ressources

## Pour le jury (Bloc 4 AIA)

Ce pipeline coche plusieurs compétences du référentiel :

- **Rédiger un cahier des charges** : docs + choix justifiés ici
- **Adapter l'infrastructure de données** : ETL multi-sources
- **Accessibilité** : sources publiques, reproductible, open source
- **Monitoring qualité** : validation des champs obligatoires + stats
- **Conformité** : robots.txt, rate limit, traçabilité des sources
