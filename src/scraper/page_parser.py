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


def _find_heading_any(
    soup: BeautifulSoup,
    keyword_sets: list[tuple[str, ...]],
    tags: tuple[str, ...] = ("h2", "h3", "h4"),
) -> Tag | None:
    """
    Try each keyword set in order; return the first heading that matches any set.
    Each set is a tuple of keywords that ALL must appear in the heading text.

    Example::

        _find_heading_any(soup, [
            ("character", "profile"),
            ("character", "information"),
        ])
    """
    for tag in tags:
        for h in soup.find_all(tag):
            txt = h.get_text(strip=True).lower()
            for kws in keyword_sets:
                if all(kw.lower() in txt for kw in kws):
                    return h
    return None


def _first_table(heading: Tag, stop: tuple[str, ...] = ("h2", "h3", "h4")) -> Tag | None:
    """
    Return the first <table> sibling after *heading*, stopping at the next heading
    in *stop*.  If no table is found at the heading's own level, search deeper
    through all sub-headings until the next heading of the same or higher level.
    """
    # Direct search at heading's own level
    for sib in _siblings_until_heading(heading, stop):
        if sib.name == "table":
            return sib

    # Deeper search: look through ALL content (including nested sub-headings)
    # until the next heading of the same or higher level than *heading*.
    heading_level = heading.name
    if heading_level == "h2":
        deep_stop = ("h2",)
    elif heading_level == "h3":
        deep_stop = ("h2", "h3")
    else:
        return None

    for sib in _siblings_until_heading(heading, stop=deep_stop):
        if sib.name == "table":
            return sib
        # Search inside any sub-heading at any depth
        if sib.name in ("h3", "h4") and sib.name != heading_level:
            # Look at direct siblings of this sub-heading
            sub_stop = (heading_level, "h2", "h3") if heading_level == "h2" else ("h2", "h3", sib.name)
            for sub_sib in _siblings_until_heading(sib, stop=sub_stop):
                if sub_sib.name == "table":
                    return sub_sib
                # Go one more level deep (h4 under h3 under h2)
                if sub_sib.name == "h4" and heading_level == "h2":
                    for deep_sib in _siblings_until_heading(sub_sib, stop=("h2", "h3", "h4")):
                        if deep_sib.name == "table":
                            return deep_sib

    return None


def _all_tables(heading: Tag, stop: tuple[str, ...] = ("h2", "h3", "h4")) -> list[Tag]:
    """Return all <table> siblings after *heading* before the next heading."""
    return [
        sib for sib in _siblings_until_heading(heading, stop=stop)
        if sib.name == "table"
    ]


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
        r"|Agent|Chaos|Mist|Golden Raven|Prismatic"
        r"|Meshing Gear|Mechanical Spur Gear|Artificed Dynamic Gear"  # Skirk/Fontaine
        r"|Ensnaring Gaze|Clockwork Meka"                              # new enemies
        r"|Gear|Spur|Clockwork|Meka|Cogs|Sprocket)", re.I)),
    ("ascension_gem", re.compile(
        r"(Sliver|Fragment|Chunk|Gemstone|Turquoise|Amethyst|Topaz"
        r"|Lapis|Beryl|Jade|Prism|Coral|Quartz|Carnelian)", re.I)),
    ("talent_book",   re.compile(r"(Teachings of|Guide to|Philosophies of)", re.I)),
    ("local_specialty", re.compile(
        r"(hook|berry|mint|lotus|qingxin|violetgrass|windwheel|calla"
        r"|dandelion|cecilia|philanemo|wolfhook|valberry|crimson|onikabuto"
        r"|naku|sango|sakura|sea ganoderma|dendrobium|rukkhashava"
        r"|sand grease|nilotpala|kalpalata|rainbow rose|lumitoile"
        r"|lakelight|romaritime|subdetection|clearwater|dewdrop"
        r"|skysplit gembloom|gembloom|brightcrown|evenstar"         # Nod-Krai specialties
        r"|snowstrider|glacialheart|springtime|clearpool)", re.I)),
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
        # Skip level-range rows in the per-level table format (e.g. "Lv.20→Lv.40")
        if re.match(r"Lv\.\d+", cell):
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
    """
    Parse the ranked weapon table into ``[{tier, name, explanation}]``.

    Handles two game8 table formats:
    1. Tier labels: "Best-in-Slot", "Statstick", "Best 4-Star", "F2P"
    2. Rank labels: "1st", "2nd", "3rd" (used by newer/reworked pages)
    """
    # Tier-based format
    tier_pat = re.compile(r"^(Best-in-Slot|Best 4-Star|F2P|Statstick)$", re.I)
    # Rank-based format
    rank_pat = re.compile(r"^(1st|2nd|3rd|4th|5th)$", re.I)

    parts = [p.strip() for p in raw.split("|") if p.strip()]
    # Drop leading section header ("Best Weapons for X" or "Weapon|Weapon Information")
    while parts and (re.search(r"\bfor\b", parts[0], re.I)
                     or parts[0].lower() in ("weapon", "weapon information",
                                             "all good weapons", "all recommended weapons")):
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
        elif rank_pat.match(part):
            _flush()
            tier = {"1st": "Best-in-Slot", "2nd": "Alternative",
                    "3rd": "Alternative", "4th": "Alternative",
                    "5th": "Alternative"}.get(part, part)
            name = None; desc = []
        elif part.lower() in ("base atk", "bonus stat", "skill effect"):
            continue  # skip stat labels in rank-based format
        elif tier and name is None and len(part) < 80 and not part.startswith(":"):
            name = part
        elif name is not None and not part.startswith(":"):
            desc.append(part)
        elif part.startswith(":"):
            continue  # skip colon separators

    # Handle F2P tables: no rank/tier labels, just weapon names directly
    # If no tier has been seen yet and a name-like cell appears, treat it as F2P
    if not items and not tier and parts:
        for part in parts:
            if part.lower() in ("base atk", "bonus stat", "skill effect"):
                continue
            if part.startswith(":"):
                continue
            if len(part) < 80 and not part[0].islower():
                items.append({"tier": "F2P", "name": part,
                              "explanation": " ".join(p for p in parts[parts.index(part)+1:]
                                if p.lower() not in ("base atk", "bonus stat", "skill effect")
                                and not p.startswith(":")).strip()[:200]})
                break

    _flush()
    return items


def _parse_artifact_sets(raw: str) -> list[dict]:
    """
    Parse artifact sets table into::

        [{tier, pieces: [{set, count}], explanation}]

    Handles two game8 table formats:
    1. Tier labels: "Best-in-Slot", "Element-Focused", "General"
    2. Rank labels: "1st", "2nd", "3rd" with separate 2-PC/4-PC descriptions
    """
    tier_pat = re.compile(r"^(Best-in-Slot|Best 4-Star|F2P|Statstick|Best)$", re.I)
    rank_pat = re.compile(r"^(1st|2nd|3rd|4th|5th)$", re.I)
    count_pat = re.compile(r"^x?(\d)$", re.I)

    # Comprehensive tier-label vocabulary. game8 uses many variants across
    # characters: "Best", "Best Option", "Alt. Best Option", "Alternative",
    # "Element-Focused", "General", "Temporary/F2P", etc.
    tier_label_pat = re.compile(
        r"^(Best-in-Slot|Best 4-Star|Best Option|Alt\.? Best Option"
        r"|Alternative|Alternatives|Element-Focused|General|Statstick"
        r"|Temporary/?F?2?P?|Temporary|F2P|Budget|Niche|Situational|Best)$",
        re.I,
    )

    def _is_tier(cell: str) -> bool:
        c = cell.rstrip("/").strip()
        if tier_label_pat.match(c):
            return True
        # Prefix forms like "Best for Stellar-Swirl", "Best for Vaporize"
        if re.match(r"^Best for [A-Za-z\- ]+$", c, re.I) and len(c) < 40:
            return True
        return False

    parts = [p.strip() for p in raw.split("|") if p.strip()]
    # Drop leading section header ("Best Artifact Sets for X", "Artifact | Artifact Bonuses")
    while parts and (re.search(r"\bfor\b", parts[0], re.I)
                     or parts[0].lower() in ("artifact", "artifact bonuses",
                                             "artifact set", "artifact sets")):
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

        if _is_tier(part):
            _flush()
            cur_tier = part.rstrip("/")
            cur_pieces = []; cur_desc = []
            i += 1
        elif rank_pat.match(part):
            _flush()
            cur_tier = {"1st": "Best-in-Slot", "2nd": "Alternative",
                        "3rd": "Alternative", "4th": "Alternative",
                        "5th": "Alternative"}.get(part, part)
            cur_pieces = []; cur_desc = []
            i += 1
        elif cur_tier is not None:
            # Rank-format "N-PC" labels: a set name immediately precedes a
            # "2-PC"/"4-PC" cell. We've already captured the set name as a
            # piece, so here we just record the piece-count on the last piece
            # and append its description.
            pc_m = re.match(r"([24])-?pc", part, re.I)
            if pc_m:
                count = int(pc_m.group(1))
                # Apply the real count to the most recent piece
                if cur_pieces:
                    cur_pieces[-1]["count"] = count
                # Capture the description that follows
                j = i + 1
                while j < len(parts) and parts[j] == ":": j += 1
                if j < len(parts):
                    cur_desc.append(parts[j])
                    i = j + 1
                else:
                    i += 1
                continue

            next_part = parts[i + 1] if (i + 1) < len(parts) else ""
            count_m = count_pat.match(next_part)

            looks_like_set = (
                len(part) < 70
                and not part[0].islower()
                and not _is_tier(part)
                and len(part.split()) <= 6
            )

            # In rank format a set name is followed by a "2-PC"/"4-PC" cell.
            next_is_pc = bool(re.match(r"[24]-?pc", next_part, re.I))

            if count_m and looks_like_set:
                cur_pieces.append({"set": part, "count": int(count_m.group(1))})
                i += 2
            elif next_is_pc and looks_like_set:
                # Set name followed by a piece-count label — count filled in
                # when we process the N-PC cell on the next iteration.
                cur_pieces.append({"set": part, "count": 4})
                i += 1
            elif looks_like_set and not cur_pieces:
                cur_pieces.append({"set": part, "count": 4})
                i += 1
            elif (looks_like_set and len(cur_pieces) == 1
                  and cur_pieces[0]["count"] == 4
                  and count_pat.match(next_part)):
                cur_pieces[0]["count"] = 2
                cur_pieces.append({"set": part, "count": int(count_m.group(1)) if count_m else 2})
                i += 2 if count_m else 1
            else:
                cur_desc.append(part)
                i += 1
        else:
            i += 1

    _flush()
    return sets


def _parse_build_card_stats(soup: BeautifulSoup) -> dict:
    """
    Extract stat priority from the build card when there's no dedicated
    "Artifact Stat Priority" table.

    The build card table has the shape::

        ... | Artifact Main Stats | Sands: | ATK% | Goblet: | Pyro DMG Bonus
            | Circlet: | CRIT Rate or CRIT DMG | Artifact Sub Stats
            | CRIT DMG, CRIT Rate, ATK%, Elemental Mastery | Sample Teams | ...

    Returns ``{sands, goblet, circlet, substats: [], notes: []}``.
    """
    result = {"sands": None, "goblet": None, "circlet": None, "substats": [], "notes": []}

    # Find the build card — the first build sub-section under "Best Builds"
    build_h = _find_heading_any(soup, [
        ("dps", "build"), ("support", "build"), ("build", "summary"),
    ])
    if not build_h:
        build_h = _find_heading_any(soup, [("best", "build"), ("build", "for")], tags=("h2",))
    if not build_h:
        return result

    table = _first_table(build_h, stop=("h2",))
    if not table:
        return result

    parts = [p.strip() for p in _table_pipe(table).split("|") if p.strip()]

    # Walk the cells, capturing values after "Sands:", "Goblet:", "Circlet:"
    # and the substats blob after "Artifact Sub Stats".
    i = 0
    in_substats = False
    while i < len(parts):
        cell = parts[i]
        lo = cell.lower().rstrip(":")

        if lo == "sands" and (i + 1) < len(parts):
            result["sands"] = _next_value(parts, i)
        elif lo == "goblet" and (i + 1) < len(parts):
            result["goblet"] = _next_value(parts, i)
        elif lo == "circlet" and (i + 1) < len(parts):
            result["circlet"] = _next_value(parts, i)
        elif "sub stat" in lo or "substat" in lo:
            in_substats = True
            i += 1
            continue
        elif in_substats:
            # First non-empty cell after "Artifact Sub Stats" is the substat list.
            # Stop at the next section marker (e.g. "Sample Teams").
            if lo in ("sample teams", "team comps", "notable teammates"):
                in_substats = False
            else:
                # Substats are comma-separated (sometimes with stray periods)
                subs = re.split(r"[,.]", cell)
                result["substats"] = [s.strip() for s in subs if s.strip() and len(s.strip()) > 1]
                in_substats = False
        i += 1

    return result


def _next_value(parts: list[str], idx: int) -> str | None:
    """Return the next cell after *idx* that isn't a bare colon."""
    j = idx + 1
    while j < len(parts) and parts[j] == ":":
        j += 1
    return parts[j] if j < len(parts) else None


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
    """
    Parse one team table into ``{name, element, members, summary, alternates}``.

    Handles two formats:
    1. Roster format: members listed then "Team Summary | explanation"
    2. Column-header format: "Main-DPS|Sub-DPS|Support|..." followed by names in rows
    """
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    members: list[dict] = []
    members_raw: list[str] = []
    summary: list[str] = []
    alts: list[str] = []

    # Detect column-header format: first row contains role labels, not names
    role_header_pat = re.compile(
        r"^(Main[-\s]?DPS|Sub[-\s]?DPS/?|On[-\s]?Field|Off[-\s]?Field"
        r"|Support/?|Healer|Buffer|Shielder|DPS|Flex)$", re.I)
    header_count = sum(1 for p in parts[:8] if role_header_pat.match(p))

    if header_count >= 2:
        # Column-header format: a run of role headers, then members in rows.
        # Roles map positionally — member[k] has role headers[k % len(headers)].
        headers: list[str] = []
        member_cells: list[str] = []
        for part in parts:
            if role_header_pat.match(part):
                # Still collecting the header row (only while no members seen yet)
                if not member_cells:
                    headers.append(_normalize_role(part))
                    continue
            if part and len(part) < 40 and part[0].isupper() and len(part.split()) <= 3:
                # Skip stray role words that appear mid-table
                if role_header_pat.match(part):
                    continue
                member_cells.append(part)

        # Assign roles positionally, cycling through the header row per team
        for idx, name in enumerate(member_cells):
            role = headers[idx % len(headers)] if headers else None
            if not any(m["name"] == name for m in members):
                members.append({"name": name, "role": role})
    else:
        # Roster format
        state = "members"
        for part in parts:
            lo = part.lower()
            if lo.startswith("team summary"):     state = "summary"; continue
            if lo.startswith("alternate"):         state = "alts";    continue
            if state == "members":
                if re.search(r"\b(mono|team)\b", part, re.I): continue
                if part and len(part) < 40 and part[0].isupper() and len(part.split()) <= 3:
                    members_raw.append(part)
            elif state == "summary": summary.append(part)
            elif state == "alts":    alts.append(part)

    summary_text = " ".join(summary).strip()
    alts_text    = " ".join(alts).strip()

    # Roster format: assign roles only where the summary prose states one
    if not members:
        members = _assign_roles(members_raw, summary_text + " " + alts_text)

    element_m = re.search(r"Mono[-\s]?(Pyro|Hydro|Electro|Cryo|Anemo|Geo|Dendro)", comp_name, re.I)
    return {
        "name":       comp_name,
        "element":    element_m.group(1).capitalize() if element_m else None,
        "members":    members,
        "summary":    summary_text,
        "alternates": alts_text,
    }


def _normalize_role(raw: str) -> str:
    """Normalize a role header cell to a clean label (strips trailing slashes)."""
    r = raw.strip().rstrip("/").strip()
    mapping = {
        "main-dps": "Main DPS", "main dps": "Main DPS",
        "sub-dps":  "Sub-DPS",  "sub dps":  "Sub-DPS",
        "on-field": "Main DPS", "off-field": "Sub-DPS",
        "support":  "Support",  "healer":   "Healer",
        "buffer":   "Buffer",   "shielder": "Shielder",
        "dps":      "DPS",      "flex":     "Flex",
    }
    return mapping.get(r.lower(), r)


def _assign_roles(names: list[str], context: str) -> list[dict]:
    """
    Assign a role to each roster member ONLY when the surrounding prose
    explicitly states one. No defaults, no guesses — if the site doesn't
    say the role, it stays ``None``.

    (Column-header team tables carry roles directly in the header row and
    are handled separately; this fallback is only for roster-format tables
    where roles may be mentioned in the summary text.)
    """
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
        role: str | None = None
        m = re.search(re.escape(name), ctx_lower, re.I)
        if m:
            window = ctx_lower[m.start(): m.start() + 120]
            for hint, label in role_hints.items():
                if hint in window:
                    role = label
                    break
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

    # Name from "X Rating and Info" or "X Rating and Basic Info" h2
    for h2 in soup.find_all("h2"):
        txt = h2.get_text(strip=True)
        lower = txt.lower()
        if "rating and info" in lower or "rating and basic info" in lower:
            name = re.sub(r"\s*rating and (?:basic )?info\s*$", "", txt, flags=re.I).strip()
            break

    # ── Identity (element / rarity / weapon) ──
    # Primary source: the meta description, which reliably reads
    # "X is a N-star Element Weapon character in Genshin Impact..."
    # for every character regardless of page layout.
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        desc = meta["content"]
        if m := re.search(r"(\d)-star", desc, re.I):
            rarity = int(m.group(1))
        if m := re.search(
            r"\b(Anemo|Hydro|Electro|Cryo|Pyro|Geo|Dendro)\b", desc):
            element = m.group(1)
        if m := re.search(
            r"\b(Sword|Claymore|Polearm|Bow|Catalyst)\b", desc, re.I):
            weapon_type = m.group(1).capitalize()

    # Fallback: the identity table (first table under "X Rating and Info" h2,
    # or under a "Character Profile"/"Character Information" heading).
    if not (element and rarity and weapon_type):
        raw = ""
        # Try the first table directly under the "Rating and Info" h2
        for h2 in soup.find_all("h2"):
            lo = h2.get_text(strip=True).lower()
            if "rating and info" in lo or "rating and basic info" in lo:
                t = _first_table(h2, stop=("h2",))
                if t:
                    raw = _table_pipe(t)
                break
        # Or a dedicated profile heading
        if not raw:
            profile_h = _find_heading_any(soup, [
                ("character", "profile"),
                ("character", "information"),
            ])
            if profile_h:
                t = _first_table(profile_h, stop=("h2", "h3"))
                if t:
                    raw = _table_pipe(t)

        if raw:
            voice_idx = raw.lower().find("voice actor")
            if voice_idx != -1:
                raw = raw[:voice_idx]
            if rarity is None and (m := re.search(r"(★+)", raw)):
                rarity = len(m.group(1))
            # Element may be labelled "Element" or "Vision"
            if element is None:
                if m := re.search(r"(?:Element|Vision)\s*\|?\s*([A-Z][a-z]+)", raw):
                    element = m.group(1)
            if weapon_type is None:
                if m := re.search(r"Weapon\s*\|?\s*([A-Za-z ]+?)(?:\s*\|.*)?$", raw):
                    weapon_type = m.group(1).strip()

    # Tier ratings
    tier_ratings: dict = {k: None for k in ["main_dps", "sub_dps", "support", "exploration"]}
    tier_h = _find_heading(soup, "tier list ranking")
    if tier_h:
        table = _first_table(tier_h, stop=("h2", "h3", "h4"))
        if table:
            tier_ratings = _parse_tier_table(table)

    # Tier explanation — heading is "Tier Rating" but sometimes absent
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
            raw = _table_pipe(table)
            parts = [p.strip() for p in raw.split("|") if p.strip()]
            in_s = in_w = False
            for part in parts:
                lo = part.lower()
                if lo == "strengths":
                    in_s, in_w = True, False
                elif lo == "weaknesses":
                    in_s, in_w = False, True
                elif in_s and part.startswith("•"):
                    strengths.append(re.sub(r"^[•\s]+", "", part))
                elif in_w and part.startswith("•"):
                    weaknesses.append(re.sub(r"^[•\s]+", "", part))

            # Fallback: if no explicit labels, split on the heading text
            # Some tables embed "Strengths" and "Weaknesses" as part of a
            # longer cell like "Raiden Shogun's Strengths" rather than a standalone cell
            if not strengths and not weaknesses:
                in_s = in_w = False
                for part in parts:
                    lo = part.lower()
                    if "strength" in lo and "weakness" not in lo:
                        in_s, in_w = True, False
                        # If bullets are in the same cell, extract them
                        if "•" in part:
                            for bullet in part.split("•")[1:]:
                                if bullet.strip():
                                    strengths.append(bullet.strip())
                    elif "weakness" in lo:
                        in_s, in_w = False, True
                        if "•" in part:
                            for bullet in part.split("•")[1:]:
                                if bullet.strip():
                                    weaknesses.append(bullet.strip())
                    elif in_s and "•" in part:
                        for bullet in part.split("•")[1:]:
                            if bullet.strip():
                                strengths.append(bullet.strip())
                    elif in_w and "•" in part:
                        for bullet in part.split("•")[1:]:
                            if bullet.strip():
                                weaknesses.append(bullet.strip())

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

    # Try multiple heading patterns — some pages use h2, some h3
    sets_h = _find_heading_any(soup, [
        ("best", "artifact", "for"),     # "Best Artifacts for X"
        ("artifact", "ranking"),          # "X Artifact Rankings"
        ("artifact", "ranked"),           # "Best Artifacts Ranked"
        ("best", "artifact"),             # "X Best Artifacts"
    ])
    if sets_h:
        # Direct table under the heading
        table = _first_table(sets_h, stop=("h2",))
        if table:
            sets = _parse_artifact_sets(_table_pipe(table))

        # If none found (or empty), scan all h3 sub-sections under this h2
        if not sets and sets_h.name == "h2":
            for sib in _siblings_until_heading(sets_h, stop=("h2",)):
                if sib.name in ("h3", "h4"):
                    sub_lo = sib.get_text(strip=True).lower()
                    # Prefer artifact-set sub-headings, skip stat priority / 4-star
                    if "stat priority" in sub_lo or "4-star" in sub_lo or "4 star" in sub_lo:
                        continue
                    sub_table = _first_table(sib, stop=("h2", "h3", "h4"))
                    if sub_table:
                        items = _parse_artifact_sets(_table_pipe(sub_table))
                        if items:
                            sets = items
                            break

    four_h = _find_heading_any(soup, [
        ("4-star", "artifact"),
        ("4 star", "artifact"),
        ("alternative", "artifact"),
    ])
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

    stat_h = _find_heading(soup, "artifact", "stat", "priority")
    if stat_h:
        if table := _first_table(stat_h, stop=("h2", "h3", "h4")):
            stat_priority = _parse_stat_priority(_table_pipe(table))

    # Fallback: most characters have no dedicated "Artifact Stat Priority" table.
    # Instead the stat priority lives inside the build card under
    # "Artifact Main Stats" (Sands/Goblet/Circlet) and "Artifact Sub Stats".
    if not stat_priority.get("sands"):
        build_card = _parse_build_card_stats(soup)
        if build_card.get("sands"):
            stat_priority = build_card

    goal_h = _find_heading(soup, "goal", "stat")
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
    # Try weapon-type-specific headings first, then generic patterns.
    # Weapon type isn't always parsed yet (profile may have failed), so try all types.
    weapon_types = ["sword", "claymore", "polearm", "bow", "catalyst"]
    h: Tag | None = None
    for wt in weapon_types:
        h = _find_heading(soup, "best", wt)
        if h:
            break
    if not h:
        h = _find_heading_any(soup, [
            ("weapon", "ranking"),
            ("best", "weapon"),
            ("recommended", "weapon"),
            ("all", "weapon"),
        ])

    ranked: list[dict] = []
    if h:
        # The weapon table may be directly after the heading or under a sub-heading.
        table = _first_table(h, stop=("h2",))
        if table:
            ranked = _parse_ranked_weapon_items(_table_pipe(table))

        # If no table found at h2 level, try each h3 sub-heading
        if not ranked and h.name == "h2":
            for sib in _siblings_until_heading(h, stop=("h2",)):
                if sib.name == "h3":
                    sub_table = _first_table(sib, stop=("h2", "h3"))
                    if sub_table:
                        items = _parse_ranked_weapon_items(_table_pipe(sub_table))
                        if items:
                            ranked = items
                            break

    alternatives: list[str] = []
    dps_h = _find_heading_any(soup, [
        ("dps", "build"),
        ("support", "build"),
        ("build", "summary"),
    ])
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

    # Fallback: newer characters use descriptive headings instead of
    # "Passive Talent N". Look for the ascension-passive table under the
    # "Talents and How to Use" section (passives have a Name|Effect table
    # describing A1/A4 ascension passives).
    if not passives:
        passives = _extract_passives_fallback(soup, utility_re)

    return {
        "priority_order":    priority_order,
        "priority_rationale": priority_rationale,
        "passives":          passives,
    }


def _extract_passives_fallback(soup: BeautifulSoup, utility_re: re.Pattern) -> list[dict]:
    """
    Extract passives when there are no "Passive Talent N" headings.

    Strategy: find the "Talents and How to Use" (or "How to Play") section
    and look for tables whose content describes passive talents — these have
    rows like "Passive Name | effect description". We identify them by the
    presence of ascension-passive keywords.
    """
    passives: list[dict] = []

    talents_section = _find_heading_any(soup, [
        ("talents", "how to use"),
        ("how to play",),
        ("talents", "and", "how"),
    ], tags=("h2",))
    if not talents_section:
        return passives

    # Collect tables in this section that look like passive descriptions
    passive_kw = re.compile(
        r"(ascension|passive|when|after|increases|every|party member"
        r"|elemental|reaction|utility)", re.I)

    idx = 0
    for sib in _siblings_until_heading(talents_section, stop=("h2",)):
        if sib.name != "table":
            continue
        raw = sib.get_text(separator=" ", strip=True)
        # Heuristic: passive tables are prose-heavy and mention ascension/passive mechanics
        if len(raw) > 60 and passive_kw.search(raw):
            # Skip tables that are clearly normal/skill/burst (those have
            # "Lv.1 | Lv.2" damage scaling columns)
            if re.search(r"Lv\.?\s?\d", raw):
                continue
            idx += 1
            trigger = re.search(
                r"\b(When|For every|After|If |Upon |Increases|Enhances|Deals)\b", raw)
            name = raw[: trigger.start()].strip() if trigger else None
            desc = raw[trigger.start():].strip() if trigger else raw
            # Cap name length — descriptive headings can be long
            if name and len(name) > 60:
                name = None
            passives.append({
                "slot":        f"Passive {idx}",
                "name":        name,
                "type":        "utility" if utility_re.search(desc or "") else "combat",
                "description": desc or None,
            })
            if idx >= 4:
                break

    return passives


def _extract_teams(soup: BeautifulSoup) -> dict:
    # Match "Best Team Comps", "Best Teams", "Best Team Comp" (singular), etc.
    teams_h = _find_heading_any(soup, [
        ("best", "team", "comp"),
        ("best", "team"),       # "Best Teams", "Best Team"
    ])
    if not teams_h:
        return {"comps": [], "flex_teammates": []}

    comps: list[dict] = []
    flex_teammates: list[dict] = []
    # Team comp sub-sections: mono-element teams, named team archetypes,
    # or "Other Teammates" / "Notable Teammates"
    mono_kws = {"mono-pyro", "mono-hydro", "mono-electro", "mono-cryo",
                "mono pyro", "mono hydro", "mono electro", "mono cryo"}

    for sib in _siblings_until_heading(teams_h, stop=(teams_h.name,)):
        if sib.name not in ("h3", "h4"):
            continue
        sub_title = sib.get_text(strip=True)
        sub_lower = sub_title.lower()

        # Check for mono-element team or any named team comp
        # (broader than before — not just mono-element)
        is_team_comp = (
            any(kw in sub_lower for kw in mono_kws)
            or "team" in sub_lower
            or "dps" in sub_lower
            or "support" in sub_lower
            or "burst" in sub_lower
            or "carry" in sub_lower
            or "overload" in sub_lower
            or "hyperbloom" in sub_lower
            or "freeze" in sub_lower
            or "melt" in sub_lower
            or "comp" in sub_lower
        )

        # Check for flex teammates section — any heading containing "teammate"
        is_flex = "teammate" in sub_lower

        if is_flex:
            if table := _first_table(sib, stop=("h2", "h3", "h4")):
                flex = _parse_flex_teammates(_table_pipe(table))
                if flex:
                    flex_teammates.extend(flex)
        elif is_team_comp:
            if table := _first_table(sib, stop=("h2", "h3", "h4")):
                raw = _table_pipe(table)
                parts = [p.strip() for p in raw.split("|") if p.strip()]
                # Detect flex table: starts with "Character" header cell
                if parts and parts[0].lower() == "character":
                    flex = _parse_flex_teammates(raw)
                    if flex:
                        flex_teammates.extend(flex)
                else:
                    comps.append(_parse_team_table(raw, sub_title))

    return {"comps": comps, "flex_teammates": flex_teammates}


def _extract_constellations(soup: BeautifulSoup) -> dict:
    levels: list[dict] = []
    ratings_map: dict = {}

    effects_h = _find_heading(soup, "constellation", "effect")
    if effects_h:
        if table := _first_table(effects_h, stop=("h2", "h3")):
            levels = _parse_constellation_effects(_table_pipe(table))

    # "Best Constellation Rating" or "Best Constellations Rating" (plural)
    rating_h = _find_heading_any(soup, [
        ("constellation", "rating"),
        ("constellations", "rating"),
    ])
    if rating_h:
        if table := _first_table(rating_h, stop=("h2", "h3")):
            ratings_map = _parse_constellation_ratings(_table_pipe(table))

    for c in levels:
        lvl = c["level"]
        if lvl in ratings_map:
            c.update(ratings_map[lvl])
        else:
            c.update({"rating": None, "worth_pulling": None, "merits": []})

    # Pull recommendation note
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

    # Try the combined summary heading first (h4)
    # Then fall back to separate h2/h3 sections
    # Search h4 first (summary with totals), then fall back to h2/h3 (level-split table).
    h = _find_heading_any(soup, [("ascension", "talent", "material")], tags=("h4",))
    if not h:
        h = _find_heading_any(soup, [
            ("ascension", "talent", "material"),
            ("ascension", "material"),
        ], tags=("h2", "h3"))
    if not h:
        return empty

    table = _first_table(h, stop=("h2", "h3", "h4"))
    if table:
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
    else:
        asc_items = []
        tal_items = []

    # If talent items are empty (common when pages use separate sections),
    # look for the talent-specific section independently.
    # Use "level-up" as the discriminator — it only appears in the talent
    # materials heading, not the combined "Ascension and Talent Materials" h2.
    if not tal_items:
        tal_h = _find_heading_any(soup, [
            ("talent", "level-up", "material"),
            ("talent", "level-up"),
            ("level-up", "material"),
        ], tags=("h3", "h4"))  # only search h3/h4 — the combined h2 is already handled
        if tal_h:
            for t in _all_tables(tal_h, stop=("h2", "h3")):
                tal_items.extend(_parse_material_list(_table_pipe(t)))

    # If ascension items are empty, try the ascension-specific section
    if not asc_items:
        asc_h = _find_heading_any(soup, [
            ("ascension", "material"),
        ])
        if asc_h:
            for t in _all_tables(asc_h, stop=("h2", "h3")):
                asc_items.extend(_parse_material_list(_table_pipe(t)))

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
