"""
src.scraper.page_parser
=======================
Parses a single game8.co character build page into a structured dict.

The page structure uses <h2>/<h3>/<h4> headings as section anchors with
<table> elements immediately following each heading carrying the data.
Content is delivered as minified single-line HTML, so all extraction is
done through BeautifulSoup rather than line-based parsing.

Output schema (one dict per character):
    character_name, source_url, last_updated
    overview       – identity, tier, role, strengths/weaknesses
    artifacts      – sets, stat priority, goal stats
    weapons        – ranked list with explanations
    talents        – priority order, passives (labeled combat/utility)
    teams          – mono-element comps, flex teammates
    constellations – C1–C6 with ratings and pull recommendation
    materials      – ascension + talent items grouped by type

Excluded by design: voice actors, release date / how-to-obtain.
"""

from __future__ import annotations

import re
import requests
from bs4 import BeautifulSoup, Tag
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# HTTP config
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "JoellianePersonalProject/1.0 "
        "(non-commercial, personal use; contact: joelliane@gmail.com)"
    )
}

# ---------------------------------------------------------------------------
# DOM helpers
# ---------------------------------------------------------------------------

def _siblings_until_heading(tag: Tag, stop: tuple[str, ...] = ("h2", "h3", "h4")):
    """Yield sibling elements after *tag* until the next heading."""
    for sib in tag.find_next_siblings():
        if sib.name in stop:
            break
        yield sib


def _table_pipe(table: Tag) -> str:
    """Flatten a table's cells into a pipe-separated string."""
    return table.get_text(separator="|", strip=True)


def _find_heading(
    soup: BeautifulSoup,
    *keywords: str,
    tags: tuple[str, ...] = ("h2", "h3", "h4"),
) -> Tag | None:
    """Return the first heading whose lowercased text contains ALL keywords."""
    for tag in tags:
        for h in soup.find_all(tag):
            txt = h.get_text(strip=True).lower()
            if all(kw.lower() in txt for kw in keywords):
                return h
    return None


def _first_table(heading: Tag, stop: tuple[str, ...] = ("h2", "h3", "h4")) -> Tag | None:
    """Return the first <table> sibling after *heading*."""
    for sib in _siblings_until_heading(heading, stop):
        if sib.name == "table":
            return sib
    return None


def _paras(heading: Tag, stop: tuple[str, ...] = ("h2", "h3", "h4")) -> list[str]:
    """Return non-empty text from <p> siblings after *heading*."""
    return [
        sib.get_text(separator=" ", strip=True)
        for sib in _siblings_until_heading(heading, stop)
        if sib.name == "p" and sib.get_text(strip=True)
    ]


def _clean(text: str) -> str:
    """Remove pipe characters used as inline emphasis markers in game8 <p> tags."""
    return re.sub(r"\s*\|\s*", " ", text).strip()


# ---------------------------------------------------------------------------
# Material type classifier
# ---------------------------------------------------------------------------

_MATERIAL_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("mora",          re.compile(r"^Mora$")),
    ("crown",         re.compile(r"Crown of Insight", re.I)),
    # Boss/enemy drops before ascension gems — some names contain gem-like words
    ("weekly_boss",   re.compile(
        r"(Ascended Sample|Shard of|Tears of|Dream Solvent|Bloodjade"
        r"|Mirror of|Worldspan|Omnivorous)", re.I)),
    ("enemy_drop",    re.compile(
        r"(Drive Shaft|Severed Tail|Fang|Claw|Bone|Wick|Arrowhead"
        r"|Mask|Scroll|Slime|Insignia|Fatui|Recruit|Sergeant|Lieutenant"
        r"|Agent|Chaos|Mist|Golden Raven|Prismatic)", re.I)),
    ("ascension_gem", re.compile(
        r"(Sliver|Fragment|Chunk|Gemstone|Turquoise|Amethyst|Topaz"
        r"|Lapis|Beryl|Jade|Prism|Coral|Quartz|Carnelian)", re.I)),
    ("talent_book",   re.compile(r"(Teachings of|Guide to|Philosophies of)", re.I)),
    ("local_specialty", re.compile(
        r"(hook|berry|mint|lotus|qingxin|violetgrass|windwheel|calla"
        r"|dandelion|cecilia|philanemo|wolfhook|valberry|crimson|onikabuto"
        r"|naku|sango|sakura|sea ganoderma|dendrobium|rukkhashava"
        r"|sand grease|nilotpala|kalpalata|rainbow rose|lumitoile"
        r"|lakelight|romaritime|subdetection|clearwater|dewdrop)", re.I)),
]


def _classify_material(name: str) -> str:
    for type_key, pattern in _MATERIAL_RULES:
        if pattern.search(name):
            return type_key
    return "other"


# ---------------------------------------------------------------------------
# Generic table parsers
# ---------------------------------------------------------------------------

def _parse_material_list(raw: str) -> list[dict]:
    """``"ItemName | ×N | ..."`` → ``[{"name": str, "quantity": int | None}]``"""
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    items: list[dict] = []
    i = 0
    while i < len(parts):
        cell = parts[i]
        if re.match(r"all .* (materials|needed)", cell, re.I):
            i += 1
            continue
        qty: int | None = None
        if (i + 1) < len(parts) and re.match(r"×[\d,.\s\u00a0]+", parts[i + 1]):
            qty = int(re.sub(r"[×,.\s\u00a0]", "", parts[i + 1]))
            i += 2
        else:
            i += 1
        if cell and not re.match(r"×", cell) and len(cell) < 60 and not cell[0].islower():
            items.append({"name": cell, "quantity": qty})
    return items


def _parse_ranked_weapon_items(raw: str) -> list[dict]:
    """Parse the ranked weapon table into ``[{tier, name, explanation}]``."""
    tier_pat = re.compile(r"^(Best-in-Slot|Best 4-Star|F2P|Statstick)$", re.I)
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    # Drop leading section header ("Best Weapons for X")
    if parts and re.search(r"\bfor\b", parts[0], re.I):
        parts = parts[1:]

    items: list[dict] = []
    tier = name = None
    desc: list[str] = []

    def _flush():
        if name:
            items.append({"tier": tier, "name": name, "explanation": " ".join(desc).strip()})

    for part in parts:
        if tier_pat.match(part):
            _flush(); tier = part; name = None; desc = []
        elif tier and name is None and len(part) < 80:
            name = part
        elif name is not None:
            desc.append(part)

    _flush()
    return items


def _parse_artifact_sets(raw: str) -> list[dict]:
    """
    Parse artifact sets table into::

        [{tier, pieces: [{set, count}], explanation}]

    Handles both single 4-pc sets and 2-pc + 2-pc combos.
    """
    tier_pat = re.compile(r"^(Best-in-Slot|Element-Focused|General)$", re.I)
    count_pat = re.compile(r"^x?(\d)$", re.I)
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    if parts and re.search(r"\bfor\b", parts[0], re.I):
        parts = parts[1:]

    sets: list[dict] = []
    cur_tier: str | None = None
    cur_pieces: list[dict] = []
    cur_desc: list[str] = []

    def _flush():
        if cur_tier and cur_pieces:
            sets.append({
                "tier": cur_tier,
                "pieces": list(cur_pieces),
                "explanation": " ".join(cur_desc).strip(),
            })

    i = 0
    while i < len(parts):
        part = parts[i]
        if tier_pat.match(part):
            _flush()
            cur_tier = part; cur_pieces = []; cur_desc = []
            i += 1
        elif cur_tier is not None:
            next_part = parts[i + 1] if (i + 1) < len(parts) else ""
            count_m = count_pat.match(next_part)
            if count_m:
                cur_pieces.append({"set": part, "count": int(count_m.group(1))})
                i += 2
            elif len(part) < 70 and not part[0].islower() and not cur_pieces:
                cur_pieces.append({"set": part, "count": 4})
                i += 1
            elif (len(part) < 70 and not part[0].islower()
                  and len(cur_pieces) == 1 and cur_pieces[0]["count"] == 4):
                cur_pieces[0]["count"] = 2
                cur_pieces.append({"set": part, "count": 2})
                i += 1
            else:
                cur_desc.append(part); i += 1
        else:
            i += 1

    _flush()
    return sets


def _parse_stat_priority(raw: str) -> dict:
    """
    Parse ``"Stat Priority | ATK% | DMG% | CRIT | Substats | : | ... | Summary | ..."``
    into ``{sands, goblet, circlet, substats: [], notes: []}``.
    """
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    result = {"sands": None, "goblet": None, "circlet": None, "substats": [], "notes": []}
    stat_cells: list[str] = []
    in_stats = in_sub = in_notes = False
    substat_str: str | None = None

    for part in parts:
        lo = part.lower()
        if lo == "stat priority":     in_stats, in_sub, in_notes = True, False, False; continue
        if lo in ("substats","substat"): in_stats, in_sub, in_notes = False, True, False; continue
        if lo == "summary":           in_stats, in_sub, in_notes = False, False, True; continue
        if part == ":":               continue
        if in_stats and len(stat_cells) < 3: stat_cells.append(part)
        elif in_sub and substat_str is None: substat_str = part
        elif in_notes and part.startswith("•"):
            result["notes"].append(re.sub(r"^[•\s]+", "", part))

    for i, slot in enumerate(("sands", "goblet", "circlet")):
        result[slot] = stat_cells[i] if i < len(stat_cells) else None

    if substat_str:
        result["substats"] = [s.strip() for s in re.split(r"\s*>\s*", substat_str) if s.strip()]

    return result


def _parse_goal_stats(raw: str) -> dict:
    """``"Stat | Goal Value | ATK | 2500+ | ..."`` → ``{"ATK": "2500+", ...}``"""
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    parts = [p for p in parts if p.lower() not in ("stat", "goal value")]
    result: dict = {}
    i = 0
    while i < len(parts) - 1:
        key, val = parts[i], parts[i + 1]
        if len(key) < 30 and (re.search(r"[\d%]", val) or val.lower().startswith("only")):
            result[key] = val; i += 2
        else:
            i += 1
    return result


def _parse_tier_table(table: Tag) -> dict:
    """
    Tier rating values are rendered as ``<img alt="S">`` (lazy-loaded).
    Plain ``-`` text cells mean "not recommended". Empty cells → None.
    """
    role_labels = {"Main DPS", "Sub-DPS", "Support", "Exploration"}
    keys = ["main_dps", "sub_dps", "support", "exploration"]
    all_cells = table.find_all(["td", "th"])
    value_cells = [c for c in all_cells if c.get_text(strip=True) not in role_labels]

    ratings: dict = {}
    for i, key in enumerate(keys):
        if i >= len(value_cells):
            ratings[key] = None; continue
        cell = value_cells[i]
        img = cell.find("img")
        if img and img.get("alt", "").strip():
            ratings[key] = img["alt"].strip()
        else:
            raw = cell.get_text(strip=True)
            ratings[key] = raw if raw else None

    return ratings


def _parse_team_table(raw: str, comp_name: str) -> dict:
    """Parse one mono-element team table into ``{name, element, members, summary, alternates}``."""
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    members_raw: list[str] = []
    summary: list[str] = []
    alts: list[str] = []
    state = "members"

    for part in parts:
        lo = part.lower()
        if lo.startswith("team summary"):     state = "summary";   continue
        if lo.startswith("alternate"):        state = "alts";      continue
        if state == "members":
            if re.search(r"\b(mono|team)\b", part, re.I): continue
            if part and len(part) < 40 and part[0].isupper() and len(part.split()) <= 3:
                members_raw.append(part)
        elif state == "summary":  summary.append(part)
        elif state == "alts":     alts.append(part)

    summary_text = " ".join(summary).strip()
    alts_text    = " ".join(alts).strip()
    members = _assign_roles(members_raw, summary_text + " " + alts_text)

    element_m = re.search(r"Mono[-\s]?(Pyro|Hydro|Electro|Cryo|Anemo|Geo|Dendro)", comp_name, re.I)
    return {
        "name":      comp_name,
        "element":   element_m.group(1).capitalize() if element_m else None,
        "members":   members,
        "summary":   summary_text,
        "alternates": alts_text,
    }


def _assign_roles(names: list[str], context: str) -> list[dict]:
    """Best-effort role assignment from context text."""
    role_hints = {
        "main dps": "Main DPS", "on-field": "Main DPS", "carry": "Main DPS",
        "sub-dps": "Sub-DPS",   "off-field": "Sub-DPS", "secondary dps": "Sub-DPS",
        "grouping": "Crowd Control", "crowd control": "Crowd Control",
        "heal": "Healer",       "sustain": "Healer",
        "shield": "Shielder",   "buff": "Buffer",
        "res shred": "Buffer",  "support": "Support",
        "battery": "Battery",   "resonance": "Resonance Enabler",
    }
    ctx_lower = context.lower()
    result: list[dict] = []
    for name in names:
        m = re.search(re.escape(name), ctx_lower, re.I)
        role: str | None = None
        if m:
            window = ctx_lower[m.start(): m.start() + 120]
            for hint, label in role_hints.items():
                if hint in window:
                    role = label; break
        result.append({"name": name, "role": role})
    return result


def _parse_flex_teammates(raw: str) -> list[dict]:
    """Parse the Other Teammates table into ``[{name, role, explanation}]``."""
    role_pat = re.compile(r"^(Support|Sub-DPS|DPS|Healer|Buffer|Shielder)$", re.I)
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    while parts and parts[0].lower() in ("character", "explanation"):
        parts.pop(0)

    teammates: list[dict] = []
    i = 0
    while i < len(parts):
        name = parts[i]
        if (name and len(name) < 30 and name[0].isupper()
                and (i + 1) < len(parts) and role_pat.match(parts[i + 1])):
            teammates.append({
                "name": name,
                "role": parts[i + 1],
                "explanation": parts[i + 2].strip() if (i + 2) < len(parts) else "",
            })
            i += 3
        else:
            i += 1
    return teammates


def _parse_constellation_effects(raw: str) -> list[dict]:
    """Split the full constellation table on C1–C6 markers."""
    segments = re.split(r"\|C([1-6])\|", raw)
    constellations: list[dict] = []
    i = 1
    while i < len(segments) - 1:
        level_num, body = segments[i], segments[i + 1]
        i += 2
        parts = [p.strip() for p in body.split("|") if p.strip()]
        name = None
        effect_parts: list[str] = []
        for j, part in enumerate(parts):
            if name is None:
                name = part.strip('"')
            elif part == ".":
                continue
            else:
                effect_parts.append(part)
        constellations.append({
            "level":  f"C{level_num}",
            "name":   name,
            "effect": " ".join(effect_parts).strip(),
        })
    return constellations


def _parse_constellation_ratings(raw: str) -> dict:
    """Parse ``"C1 | ★★☆ | • merit ..."`` into ``{"C1": {rating, worth_pulling, merits}}``. """
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    parts = [p for p in parts if p.lower() not in ("rating", "constellation effect / merits")]
    result: dict = {}
    i = 0
    while i < len(parts):
        if re.match(r"C[1-6]$", parts[i]):
            level = parts[i]
            stars = parts[i + 1] if (i + 1) < len(parts) else None
            merits: list[str] = []
            j = i + 2
            while j < len(parts) and not re.match(r"C[1-6]$", parts[j]):
                if parts[j].startswith("•"):
                    merits.append(re.sub(r"^[•\s]+", "", parts[j]))
                j += 1
            worth = stars.count("★") >= 3 if stars else None
            result[level] = {"rating": stars, "worth_pulling": worth, "merits": merits}
            i = j
        else:
            i += 1
    return result


# ---------------------------------------------------------------------------
# Section extractors
# ---------------------------------------------------------------------------

def _extract_last_updated(soup: BeautifulSoup) -> str:
    img = soup.find("img", src=re.compile(r"archive_revision=\d+"))
    if img:
        m = re.search(r"archive_revision=(\d{8})", img["src"])
        if m:
            r = m.group(1)
            return f"{r[:4]}-{r[4:6]}-{r[6:8]}"
    return datetime.utcnow().strftime("%Y-%m-%d")


def _extract_overview(soup: BeautifulSoup) -> dict:
    name = element = weapon_type = role = None
    rarity: int | None = None

    # Name from "X Rating and Info" h2
    for h2 in soup.find_all("h2"):
        txt = h2.get_text(strip=True)
        if "rating and info" in txt.lower():
            name = re.sub(r"\s*rating and info\s*$", "", txt, flags=re.I).strip()
            break

    # Profile table
    profile_h = _find_heading(soup, "character profile")
    if profile_h:
        table = _first_table(profile_h, stop=("h2", "h3"))
        if table:
            raw = _table_pipe(table)
            voice_idx = raw.lower().find("voice actor")
            if voice_idx != -1:
                raw = raw[:voice_idx]
            if m := re.search(r"(★+)", raw):
                rarity = len(m.group(1))
            if m := re.search(r"Element\s*\|?\s*([A-Z][a-z]+)", raw):
                element = m.group(1)
            if m := re.search(r"Weapon\s*\|?\s*([A-Za-z ]+?)(?:\s*\|.*)?$", raw):
                weapon_type = m.group(1).strip()

    # Tier ratings
    tier_ratings: dict = {k: None for k in ["main_dps", "sub_dps", "support", "exploration"]}
    tier_h = _find_heading(soup, "tier list ranking")
    if tier_h:
        table = _first_table(tier_h, stop=("h2", "h3", "h4"))
        if table:
            tier_ratings = _parse_tier_table(table)

    # Tier explanation
    tier_explanation: str | None = None
    rating_h = _find_heading(soup, "tier rating")
    if rating_h:
        table = _first_table(rating_h, stop=("h2", "h3", "h4"))
        if table:
            tier_explanation = re.sub(r"\s+", " ", _table_pipe(table).replace("|", " ")).strip()
        if tier_explanation:
            if m := re.match(r"As an? ([A-Za-z ]+?)\b", tier_explanation):
                role = m.group(1).strip()

    # Strengths / weaknesses
    strengths: list[str] = []
    weaknesses: list[str] = []
    sw_h = _find_heading(soup, "strengths and weaknesses")
    if sw_h:
        table = _first_table(sw_h, stop=("h2", "h3"))
        if table:
            in_s = in_w = False
            for part in _table_pipe(table).split("|"):
                part = part.strip()
                if part.lower() == "strengths":   in_s, in_w = True, False
                elif part.lower() == "weaknesses": in_s, in_w = False, True
                elif in_s and part.startswith("•"): strengths.append(re.sub(r"^[•\s]+", "", part))
                elif in_w and part.startswith("•"): weaknesses.append(re.sub(r"^[•\s]+", "", part))

    return {
        "character_name":  name,
        "element":         element,
        "rarity":          rarity,
        "weapon_type":     weapon_type,
        "role":            role,
        "tier_ratings":    tier_ratings,
        "tier_explanation": tier_explanation,
        "strengths":       strengths,
        "weaknesses":      weaknesses,
    }


def _extract_artifacts(soup: BeautifulSoup) -> dict:
    sets: list[dict] = []
    four_star: dict | None = None
    stat_priority: dict = {"sands": None, "goblet": None, "circlet": None, "substats": [], "notes": []}
    goal_stats: dict = {}
    goal_stat_notes: list[str] = []

    sets_h = _find_heading(soup, "best artifacts for") or _find_heading(soup, "best artifact")
    if sets_h:
        if table := _first_table(sets_h, stop=("h2", "h3")):
            sets = _parse_artifact_sets(_table_pipe(table))

    four_h = _find_heading(soup, "4-star artifact")
    if four_h:
        if table := _first_table(four_h, stop=("h2", "h3", "h4")):
            raw = _table_pipe(table)
            parts = [p.strip() for p in raw.split("|") if p.strip()
                     and p.strip().lower() not in ("artifact", "artifact bonuses")]
            result: dict = {"name": None, "two_pc_bonus": None, "four_pc_bonus": None}
            i = 0
            while i < len(parts):
                part = parts[i]
                if result["name"] is None and len(part) < 40:
                    result["name"] = part
                elif re.match(r"2-?pc", part, re.I):
                    j = i + 1
                    while j < len(parts) and parts[j] == ":": j += 1
                    if j < len(parts): result["two_pc_bonus"] = parts[j]; i = j + 1; continue
                elif re.match(r"4-?pc", part, re.I):
                    j = i + 1
                    while j < len(parts) and parts[j] == ":": j += 1
                    if j < len(parts): result["four_pc_bonus"] = parts[j]; i = j + 1; continue
                i += 1
            four_star = result if result["name"] else None

    stat_h = _find_heading(soup, "artifact stat priority")
    if stat_h:
        if table := _first_table(stat_h, stop=("h2", "h3", "h4")):
            stat_priority = _parse_stat_priority(_table_pipe(table))

    goal_h = _find_heading(soup, "goal stat")
    if goal_h:
        if table := _first_table(goal_h, stop=("h2", "h3", "h4")):
            goal_stats = _parse_goal_stats(_table_pipe(table))
        goal_stat_notes = [_clean(p) for p in _paras(goal_h, stop=("h2", "h3", "h4")) if p]

    return {
        "sets":            sets,
        "four_star_option": four_star,
        "stat_priority":   stat_priority,
        "goal_stats":      goal_stats,
        "goal_stat_notes": goal_stat_notes,
    }


def _extract_weapons(soup: BeautifulSoup) -> dict:
    h = _find_heading(soup, "best claymores") or _find_heading(soup, "best weapons")
    ranked: list[dict] = []
    if h:
        if table := _first_table(h, stop=("h2", "h3")):
            ranked = _parse_ranked_weapon_items(_table_pipe(table))

    alternatives: list[str] = []
    dps_h = _find_heading(soup, "dps build")
    if dps_h:
        if table := _first_table(dps_h, stop=("h2", "h3")):
            raw = _table_pipe(table)
            parts = [p.strip() for p in raw.split("|") if p.strip()]
            skip = {"best", "weapon", "alternative", "weapons", "artifact", "artifact stats",
                    "alternate", "artifacts", "general", "team comps"}
            collecting = False
            for part in parts:
                lo = part.lower()
                if lo == "alternative": collecting = True; continue
                if collecting:
                    if lo in ("artifact", "best") or lo.startswith(("sands", "goblet")): break
                    if lo not in skip and len(part) < 50 and not part[0].islower():
                        alternatives.append(part)

    return {"ranked": ranked, "alternatives_from_build_card": alternatives}


def _extract_talents(soup: BeautifulSoup) -> dict:
    priority_order: list[dict] = []
    priority_rationale: str | None = None
    passives: list[dict] = []

    talent_h = _find_heading(soup, "talent priority")
    if talent_h:
        if table := _first_table(talent_h, stop=("h2", "h3", "h4")):
            raw = _table_pipe(table)
            parts = [p.strip() for p in raw.split("|") if p.strip()]
            ranks    = [p for p in parts if re.match(r"\d+(?:st|nd|rd|th)$", p)]
            abilities = [p for p in parts if p not in ranks and len(p) < 30]
            priority_order = [{"rank": r, "ability": a} for r, a in zip(ranks, abilities)]
        rationale_parts = [_clean(p) for p in _paras(talent_h, stop=("h2", "h3")) if p]
        priority_rationale = " ".join(rationale_parts).strip() or None

    utility_re = re.compile(
        r"(does not trigger in|exploration|open world|outside of combat"
        r"|Domains|Spiral Abyss|Trounce Domain)", re.I)

    for i in range(1, 5):
        h = _find_heading(soup, f"passive talent {i}")
        if not h:
            continue
        if table := _first_table(h, stop=("h2", "h3")):
            raw = table.get_text(separator=" ", strip=True)
            trigger = re.search(
                r"\b(When|For every|After|If |Upon |Increases|Enhances|Deals)\b", raw)
            passive_name = raw[: trigger.start()].strip() if trigger else None
            description  = raw[trigger.start():].strip() if trigger else raw
            passives.append({
                "slot":        f"Passive {i}",
                "name":        passive_name or None,
                "type":        "utility" if utility_re.search(description or "") else "combat",
                "description": description or None,
            })

    return {
        "priority_order":    priority_order,
        "priority_rationale": priority_rationale,
        "passives":          passives,
    }


def _extract_teams(soup: BeautifulSoup) -> dict:
    teams_h = _find_heading(soup, "best team comp")
    if not teams_h:
        return {"comps": [], "flex_teammates": []}

    comps: list[dict] = []
    flex_teammates: list[dict] = []
    mono_kws = {"mono-pyro", "mono-hydro", "mono-electro", "mono-cryo",
                "mono pyro", "mono hydro", "mono electro", "mono cryo"}

    for sib in _siblings_until_heading(teams_h, stop=(teams_h.name,)):
        if sib.name not in ("h3", "h4"):
            continue
        sub_title = sib.get_text(strip=True)
        sub_lower = sub_title.lower()

        if any(kw in sub_lower for kw in mono_kws):
            if table := _first_table(sib, stop=("h2", "h3", "h4")):
                comps.append(_parse_team_table(_table_pipe(table), sub_title))
        elif "other teammate" in sub_lower:
            if table := _first_table(sib, stop=("h2", "h3", "h4")):
                flex_teammates = _parse_flex_teammates(_table_pipe(table))

    return {"comps": comps, "flex_teammates": flex_teammates}


def _extract_constellations(soup: BeautifulSoup) -> dict:
    levels: list[dict] = []
    ratings_map: dict = {}

    effects_h = _find_heading(soup, "constellation and effect")
    if effects_h:
        if table := _first_table(effects_h, stop=("h2", "h3")):
            levels = _parse_constellation_effects(_table_pipe(table))

    rating_h = _find_heading(soup, "best constellation rating")
    if rating_h:
        if table := _first_table(rating_h, stop=("h2", "h3")):
            ratings_map = _parse_constellation_ratings(_table_pipe(table))

    for c in levels:
        lvl = c["level"]
        if lvl in ratings_map:
            c.update(ratings_map[lvl])
        else:
            c.update({"rating": None, "worth_pulling": None, "merits": []})

    # Pull recommendation note (first h3 + p after the ratings table)
    pull_note: str | None = None
    if rating_h:
        parts: list[str] = []
        for sib in _siblings_until_heading(rating_h, stop=("h2",)):
            if sib.name == "h3":
                parts.append(sib.get_text(strip=True))
            elif sib.name == "p":
                t = _clean(sib.get_text(separator=" ", strip=True))
                if t: parts.append(t)
        pull_note = " — ".join(parts).strip() or None

    return {"levels": levels, "pull_recommendation": pull_note}


def _extract_materials(soup: BeautifulSoup) -> dict:
    empty = {
        "ascension": {k: [] for k in ["local_specialty", "ascension_gems", "enemy_drops", "mora", "other"]},
        "talents":   {k: [] for k in ["talent_books", "weekly_boss_drop", "enemy_drops", "crown", "mora", "other"]},
    }

    h = _find_heading(soup, "ascension and talent material")
    if not h:
        return empty
    table = _first_table(h, stop=("h2", "h3", "h4"))
    if not table:
        return empty

    raw = _table_pipe(table)
    split_kw = "all talent materials needed"
    lower = raw.lower()

    if split_kw in lower:
        idx = lower.index(split_kw)
        asc_items = _parse_material_list(raw[:idx])
        tal_items = _parse_material_list(raw[idx:])
    else:
        asc_items = _parse_material_list(raw)
        tal_items = []

    def _group(items: list[dict]) -> dict[str, list[dict]]:
        groups: dict[str, list[dict]] = {}
        for item in items:
            groups.setdefault(_classify_material(item["name"]), []).append(item)
        return groups

    ag = _group(asc_items)
    tg = _group(tal_items)

    return {
        "ascension": {
            "local_specialty": ag.get("local_specialty", []),
            "ascension_gems":  ag.get("ascension_gem", []),
            "enemy_drops":     ag.get("enemy_drop", []),
            "mora":            ag.get("mora", []),
            "other":           ag.get("other", []),
        },
        "talents": {
            "talent_books":    tg.get("talent_book", []),
            "weekly_boss_drop": tg.get("weekly_boss", []),
            "enemy_drops":     tg.get("enemy_drop", []),
            "crown":           tg.get("crown", []),
            "mora":            tg.get("mora", []),
            "other":           tg.get("other", []),
        },
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def _build_output(soup: BeautifulSoup, url: str) -> dict:
    overview = _extract_overview(soup)
    return {
        "character_name": overview["character_name"],
        "source_url":     url,
        "last_updated":   _extract_last_updated(soup),
        "overview":       overview,
        "artifacts":      _extract_artifacts(soup),
        "weapons":        _extract_weapons(soup),
        "talents":        _extract_talents(soup),
        "teams":          _extract_teams(soup),
        "constellations": _extract_constellations(soup),
        "materials":      _extract_materials(soup),
    }


def scrape_character_page(url: str) -> dict:
    """Fetch a live game8.co character build page and return the parsed dict."""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return _build_output(BeautifulSoup(resp.text, "html.parser"), url)


def scrape_from_local_file(path: str) -> dict:
    """Parse a saved local HTML file (dev helper — no network request)."""
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    return _build_output(BeautifulSoup(html, "html.parser"), f"file://{path}")
