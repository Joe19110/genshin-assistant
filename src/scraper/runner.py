"""
src.scraper.runner
==================
Discovers all playable character build URLs from game8's "All Best Builds"
index page and bulk-scrapes them into character_data/.

CLI usage
---------
    py -m src.scraper.runner                 # scrape all (skips today's files)
    py -m src.scraper.runner --force         # re-scrape everything
    py -m src.scraper.runner --dry-run       # print discovered URLs, no scraping
    py -m src.scraper.runner --limit 5       # first N characters only
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .page_parser import scrape_character_page, HEADERS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INDEX_URL = "https://game8.co/games/Genshin-Impact/archives/530535"
OUTPUT_DIR = Path("character_data")
REQUEST_DELAY_SECONDS = 1.5

# Upcoming characters listed on game8 that actually have build pages
# (released but game8 hasn't moved them to the "Playable Characters" section yet).
# These are scraped alongside the playable characters.
_RELEASED_BUT_UPCOMING: list[tuple[str, str]] = [
    ("Vesna",      "https://game8.co/games/Genshin-Impact/archives/606901"),
    ("Vodyanitsa", "https://game8.co/games/Genshin-Impact/archives/606400"),
]

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_character_urls() -> list[tuple[str, str]]:
    """
    Return ``[(character_name, url), ...]`` for all playable characters
    listed on the game8 "All Best Builds" index page.

    Discovery is fully dynamic: we locate the ``<h4>Playable Characters``
    section in the page HTML and extract every character link within it,
    stopping at the ``<h4>Upcoming Characters`` heading.  This means new
    characters added to game8 are picked up automatically — no hardcoded
    name list to maintain.
    """
    print(f"Fetching character index: {INDEX_URL}")
    resp = requests.get(INDEX_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Find the "Playable Characters" section heading
    play_heading = None
    for h4 in soup.find_all("h4"):
        if h4.get_text(strip=True) == "Playable Characters":
            play_heading = h4
            break

    if not play_heading:
        print("WARNING: 'Playable Characters' heading not found — falling back to full-page scan.")
        return _discover_fallback(soup)

    # Walk siblings until "Upcoming Characters" or another h4
    seen_names: set[str] = set()
    results:    list[tuple[str, str]] = []

    for sib in play_heading.find_next_siblings():
        if sib.name in ("h2", "h3", "h4"):
            break  # reached the next section

        for a in sib.find_all("a", href=True):
            href = a["href"].split("#")[0]
            if not href or "/games/Genshin-Impact/archives/" not in href:
                continue

            text = a.get_text(strip=True)
            if not text or text.lower() in seen_names:
                continue

            full_url = "https://game8.co" + href if href.startswith("/") else href
            seen_names.add(text.lower())
            results.append((text, full_url))

    # Add characters that have build pages but are still listed under
    # "Upcoming Characters" on game8 (released but not yet moved to the
    # "Playable Characters" section).
    for name, url in _RELEASED_BUT_UPCOMING:
        if name.lower() not in seen_names:
            seen_names.add(name.lower())
            results.append((name, url))

    return results
    """
    Fallback discovery if the section heading isn't found.
    Scans the whole page for short archive links, excluding known
    non-character pages. Less precise but ensures we don't return nothing.
    """
    non_char = {
        "tier list", "best build", "weapon", "artifact", "team comp",
        "reroll", "exploration", "upcoming", "guide", "banner", "code",
    }
    seen: set[str] = set()
    results: list[tuple[str, str]] = []

    for a in soup.find_all("a", href=True):
        href = a["href"].split("#")[0]
        if not href or "/games/Genshin-Impact/archives/" not in href:
            continue
        text = a.get_text(strip=True)
        if not text or len(text) > 40 or text.lower() in seen:
            continue
        if any(kw in text.lower() for kw in non_char):
            continue
        seen.add(text.lower())
        full_url = "https://game8.co" + href if href.startswith("/") else href
        results.append((text, full_url))

    return results


# ---------------------------------------------------------------------------
# Filename / skip logic
# ---------------------------------------------------------------------------

def _slugify(char_name: str) -> str:
    """
    Convert an index-page character name into a stable filename stem.

    "Raiden"             -> "raiden"
    "Hu Tao"             -> "hu_tao"
    "Traveler (Anemo)"   -> "traveler_anemo"
    "Arataki Itto"       -> "arataki_itto"

    Parentheses are flattened (not dropped) so Traveler variants stay distinct.
    """
    slug = char_name.lower().strip()
    slug = slug.replace("(", "").replace(")", "")
    slug = re.sub(r"[^\w\s-]", "", slug)   # drop stray punctuation
    slug = re.sub(r"\s+", "_", slug)        # spaces -> underscores
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug


def _already_scraped_today(char_name: str) -> bool:
    """
    True if a JSON for *char_name* was written today.

    Uses file mtime rather than ``last_updated`` (which reflects game8's
    content revision date, not our scrape date). Matches on the exact slug
    now that filenames are derived from the stable index-page name.
    """
    path = OUTPUT_DIR / f"{_slugify(char_name)}.json"
    if path.exists():
        return date.fromtimestamp(path.stat().st_mtime) == date.today()
    return False


# ---------------------------------------------------------------------------
# Scrape loop
# ---------------------------------------------------------------------------

def scrape_all(
    force: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    characters = discover_character_urls()
    print(f"Discovered {len(characters)} character build pages.\n")

    if limit:
        characters = characters[:limit]
        print(f"Limited to first {limit} characters.\n")

    if dry_run:
        for name, url in characters:
            print(f"  {name:<30} {url}")
        return

    success = skipped = 0
    failed: list[tuple[str, str, str]] = []

    for i, (char_name, url) in enumerate(characters, 1):
        prefix = f"[{i}/{len(characters)}]"

        if not force and _already_scraped_today(char_name):
            print(f"{prefix} SKIP  {char_name}")
            skipped += 1
            continue

        print(f"{prefix} Scraping {char_name} ...")
        try:
            data = scrape_character_page(url)
            # Filename is derived from the INDEX page name (char_name), which is
            # stable and canonical ("Raiden", "Arataki Itto"). The scraped
            # character_name can be a longer title ("Raiden Shogun") or a page
            # heading ("Arataki Itto Rating and Information") — using it would
            # create duplicate/misnamed files.
            fname = _slugify(char_name)
            out_path = OUTPUT_DIR / f"{fname}.json"
            out_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"         → {out_path}")
            success += 1
        except Exception as exc:
            print(f"         ERROR: {exc}")
            failed.append((char_name, url, str(exc)))

        if i < len(characters):
            time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\n{'='*50}")
    print(f"Done. {success} scraped, {skipped} skipped, {len(failed)} failed.")
    if failed:
        print("\nFailed:")
        for name, url, err in failed:
            print(f"  {name}: {err}")
            print(f"    {url}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk-scrape all Genshin character build pages from game8."
    )
    parser.add_argument("--force",   action="store_true", help="Re-scrape even if JSON exists today.")
    parser.add_argument("--dry-run", action="store_true", help="Print URLs without scraping.")
    parser.add_argument("--limit",   type=int,            help="Only scrape first N characters.")
    args = parser.parse_args()
    scrape_all(force=args.force, dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
