"""Lance l'ensemble du pipeline de collecte :
    1. Home Assistant integrations to data/raw/home_assistant_devices.json
    2. Matter CSA certified products to data/raw/matter_csa_products.json
    3. Merge + dedup + validation to data/processed/device_catalog_extended.json

Usage :
    python data/collectors/run_all.py                 # Tout lancer
    python data/collectors/run_all.py --skip-matter   # Sauter Matter (si rate limited)
    python data/collectors/run_all.py --matter-details  # Fetch pages détail Matter (long)
"""

import argparse
import sys
import traceback
from pathlib import Path

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent))

import collect_home_assistant
import collect_matter_csa
import merge_catalogs


def run_step(name: str, fn, *args, **kwargs):
    """Exécute une étape avec gestion d'erreur."""
    print("\n" + "=" * 70)
    print(f"STEP: {name}")
    print("=" * 70)
    try:
        fn(*args, **kwargs)
        print(f" {name} completed\n")
        return True
    except Exception as e:
        print(f" {name} failed: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ha", action="store_true", help="Skip Home Assistant collection")
    parser.add_argument("--skip-matter", action="store_true", help="Skip Matter CSA collection")
    parser.add_argument("--matter-details", action="store_true", help="Fetch Matter detail pages (slow)")
    parser.add_argument("--matter-pages", type=int, default=50, help="Max Matter pages to fetch")
    args = parser.parse_args()

    print("Mezonya Device Catalog - Collection Pipeline")
    print(f"Skip HA: {args.skip_ha}")
    print(f"Skip Matter: {args.skip_matter}")

    if not args.skip_ha:
        run_step("Home Assistant", collect_home_assistant.main)

    if not args.skip_matter:
        run_step(
            "Matter CSA",
            collect_matter_csa.main,
            max_pages=args.matter_pages,
            fetch_details=args.matter_details,
        )

    # Toujours lancer le merge (même si certaines sources ont échoué, on merge ce qu'on a)
    run_step("Merge catalogs", merge_catalogs.main)

    print("\nPipeline complete. Check data/processed/device_catalog_extended.json")


if __name__ == "__main__":
    main()
