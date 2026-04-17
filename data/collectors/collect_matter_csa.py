"""Scrape respectueusement le catalogue officiel des produits certifiés Matter
par la Connectivity Standards Alliance (CSA).

Source : https://csa-iot.org/csa-iot_products/
Licence : données publiques de certification (référentiel officiel)

Rate limiting : 2 secondes entre chaque requête, User-Agent identifiable.
Cache : les résultats sont cachés 24h pour éviter le re-scraping.

Usage :
    python data/collectors/collect_matter_csa.py

Output :
    data/raw/matter_csa_products.json
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup


CSA_BASE_URL = "https://csa-iot.org/csa-iot_products/"
USER_AGENT = (
    "Mezonya-Academic-Research/1.0 "
    "(+Contact via mezonya.com; educational project; "
    "respects robots.txt and rate limits)"
)
RATE_LIMIT_SECONDS = 2.0
CACHE_DIR = Path("/tmp/mezonya_matter_cache")
CACHE_TTL_HOURS = 24
OUTPUT_FILE = Path(__file__).parent.parent / "raw" / "matter_csa_products.json"

# Filtres : Matter-certified products (compliance = Matter)
QUERY_PARAMS = {
    "p_keywords": "",
    "p_type[]": "14",  # Type: Product
    "p_program_type[]": "1049",  # Program: Matter
}

# Catégories Matter officielles to taxonomie Mezonya
MATTER_CATEGORY_MAPPING = {
    "light": "lighting",
    "bulb": "lighting",
    "switch": "lighting",
    "plug": "lighting",
    "outlet": "lighting",
    "dimmer": "lighting",
    "camera": "security",
    "doorbell": "security",
    "lock": "security",
    "sensor": "sensor",
    "thermostat": "climate",
    "fan": "climate",
    "hvac": "climate",
    "blind": "climate",
    "shade": "climate",
    "shutter": "climate",
    "hub": "hub",
    "bridge": "hub",
    "gateway": "hub",
    "speaker": "entertainment",
    "tv": "entertainment",
    "display": "entertainment",
    "vacuum": "appliance",
    "refrigerator": "appliance",
    "washer": "appliance",
    "dryer": "appliance",
}


# HTTP CLIENT
def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return session


def respectful_get(session: requests.Session, url: str, **kwargs) -> Optional[str]:
    """GET avec rate limiting et cache 24h."""
    # Cache key = URL + params
    cache_key = url
    if "params" in kwargs:
        cache_key += "?" + "&".join(f"{k}={v}" for k, v in sorted(kwargs["params"].items()))

    cache_hash = str(hash(cache_key))
    cache_file = CACHE_DIR / f"{cache_hash}.html"

    # Check cache
    if cache_file.exists():
        age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if age < timedelta(hours=CACHE_TTL_HOURS):
            return cache_file.read_text(encoding="utf-8")

    # Fetch with rate limit
    time.sleep(RATE_LIMIT_SECONDS)
    try:
        resp = session.get(url, timeout=30, **kwargs)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[Matter]   Request failed for {url}: {e}")
        return None

    # Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(resp.text, encoding="utf-8")

    return resp.text


def parse_product_listing(html: str) -> list[dict]:
    """Parse une page de listing pour extraire les cartes produits."""
    soup = BeautifulSoup(html, "html.parser")
    products = []

    # Les cartes produits CSA ont une structure .product-item ou .csa-product
    # (structure observée en avril 2026, peut évoluer)
    for card in soup.select("article.product, .product-item, .csa-product-card"):
        name_elem = card.select_one("h2, h3, .product-name, .entry-title")
        brand_elem = card.select_one(".brand, .vendor, .company-name")
        category_elem = card.select_one(".category, .product-type")
        link_elem = card.select_one("a[href]")

        if not name_elem:
            continue

        product = {
            "name": name_elem.get_text(strip=True),
            "brand": brand_elem.get_text(strip=True) if brand_elem else None,
            "category_raw": category_elem.get_text(strip=True) if category_elem else None,
            "detail_url": link_elem["href"] if link_elem else None,
        }
        products.append(product)

    # Pagination : chercher le lien "next page"
    next_link = soup.select_one('a.next.page-numbers, .pagination a[rel="next"]')
    next_url = next_link["href"] if next_link else None

    return products, next_url


def parse_product_detail(html: str) -> dict:
    """Parse une page de détail produit pour extraire les specs Matter."""
    soup = BeautifulSoup(html, "html.parser")

    details = {
        "matter_version": None,
        "transport": [],  # Thread, Wi-Fi, Ethernet
        "device_types": [],  # On/Off Light, Door Lock, etc.
        "certification_id": None,
        "certification_date": None,
    }

    # Chercher les metadata dans les tables de spec
    for row in soup.select("table tr, .spec-row, .product-meta li"):
        label_elem = row.select_one("th, .label, .spec-label, strong")
        value_elem = row.select_one("td, .value, .spec-value")

        if not label_elem or not value_elem:
            continue

        label = label_elem.get_text(strip=True).lower()
        value = value_elem.get_text(strip=True)

        if "matter version" in label:
            details["matter_version"] = value
        elif "transport" in label or "network" in label:
            details["transport"] = [t.strip().lower() for t in value.split(",")]
        elif "device type" in label or "category" in label:
            details["device_types"] = [t.strip() for t in value.split(",")]
        elif "certification" in label and "id" in label:
            details["certification_id"] = value
        elif "certification" in label and "date" in label:
            details["certification_date"] = value

    return details


def map_matter_category(device_types: list[str], category_raw: Optional[str]) -> str:
    """Mappe les catégories Matter vers notre taxonomie Mezonya."""
    all_text = " ".join(device_types + [category_raw or ""]).lower()

    for keyword, mezonya_cat in MATTER_CATEGORY_MAPPING.items():
        if keyword in all_text:
            return mezonya_cat

    return "other"


def normalize_product(raw_product: dict, details: dict) -> dict:
    """Normalise un produit Matter dans le schéma Mezonya."""
    connectivity = ["matter"]
    for t in details.get("transport", []):
        if "thread" in t.lower():
            connectivity.append("thread")
        elif "wi-fi" in t.lower() or "wifi" in t.lower():
            connectivity.append("wifi")
        elif "ethernet" in t.lower():
            connectivity.append("ethernet")

    return {
        "source": "matter_csa",
        "source_id": details.get("certification_id") or raw_product.get("detail_url", ""),
        "name": raw_product["name"],
        "brand": raw_product.get("brand") or raw_product["name"].split()[0],
        "category": map_matter_category(
            details.get("device_types", []),
            raw_product.get("category_raw"),
        ),
        "connectivity": sorted(set(connectivity)),
        # Tous les produits Matter certifiés supportent Apple/Google/Alexa/SmartThings
        "ecosystems": ["apple", "google", "alexa", "smartthings"],
        "hub_required": "thread" in connectivity,  # Thread needs border router
        "cloud_dependency": "optional",  # Matter = local par conception
        "matter_version": details.get("matter_version"),
        "certification_id": details.get("certification_id"),
        "certification_date": details.get("certification_date"),
        "doc_url": raw_product.get("detail_url"),
    }


def main(max_pages: int = 50, fetch_details: bool = False):
    """
    Args:
        max_pages: Nombre max de pages de listing à parcourir (pagination)
        fetch_details: Si True, fetch chaque page détail (plus lent, +infos)
    """
    session = make_session()

    print(f"[Matter] Starting scrape with rate limit {RATE_LIMIT_SECONDS}s/req")
    print(f"[Matter] Max pages: {max_pages}, fetch_details: {fetch_details}")

    all_products = []
    url = CSA_BASE_URL
    params = QUERY_PARAMS
    page = 1

    while url and page <= max_pages:
        print(f"[Matter] Fetching page {page}: {url}")
        html = respectful_get(session, url, params=params if page == 1 else None)

        if html is None:
            print(f"[Matter]   Failed page {page}, stopping pagination")
            break

        products, next_url = parse_product_listing(html)
        print(f"[Matter]   to {len(products)} products on page {page}")

        if not products:
            print(f"[Matter] Empty page, stopping")
            break

        # Enrichir avec les détails si demandé
        for product in products:
            details = {}
            if fetch_details and product.get("detail_url"):
                detail_html = respectful_get(session, product["detail_url"])
                if detail_html:
                    details = parse_product_detail(detail_html)

            normalized = normalize_product(product, details)
            all_products.append(normalized)

        url = next_url
        page += 1

    # Sauvegarde
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(all_products, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[Matter]  Extracted {len(all_products)} certified products")
    print(f"[Matter] Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=50)
    parser.add_argument("--fetch-details", action="store_true")
    args = parser.parse_args()

    main(max_pages=args.max_pages, fetch_details=args.fetch_details)
