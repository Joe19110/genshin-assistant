"""
src.pipeline.flatteners
=======================
Converts each section of a character JSON into a plain-English passage
suitable for semantic embedding and retrieval.

One flattener per JSON section key.  The passage should be self-contained:
a retrieval model should be able to answer a question about that section
from the passage alone, without needing to look at the raw JSON.

Each function signature is ``(data: dict) -> str``.
``data`` is the full character dict so flatteners can reference the
``character_name`` top-level key without it being duplicated in every section.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(items: list[dict]) -> str:
    """``[{name, quantity}]`` → ``"Wolfhook ×168, Mora ×420000"``"""
    parts = []
    for item in items:
        qty = item.get("quantity")
        parts.append(f"{item['name']} ×{qty}" if qty else item["name"])
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Flatteners
# ---------------------------------------------------------------------------

def flatten_overview(data: dict) -> str:
    ov = data["overview"]
    lines: list[str] = [
        f"{ov['character_name']} is a {ov['rarity']}-star "
        f"{ov['element']} {ov['weapon_type']} character.",
    ]

    if ov.get("role"):
        lines.append(f"Primary role: {ov['role']}.")

    rated = {k: v for k, v in ov.get("tier_ratings", {}).items() if v and v != "-"}
    if rated:
        tier_str = ", ".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in rated.items())
        lines.append(f"Tier ratings — {tier_str}.")

    if ov.get("tier_explanation"):
        lines.append(ov["tier_explanation"])

    if ov.get("strengths"):
        lines.append("Strengths: " + " | ".join(ov["strengths"]))
    if ov.get("weaknesses"):
        lines.append("Weaknesses: " + " | ".join(ov["weaknesses"]))

    return "\n".join(lines)


def flatten_artifacts(data: dict) -> str:
    art = data["artifacts"]
    char = data["character_name"]
    lines: list[str] = [f"Artifact builds for {char}:"]

    for s in art.get("sets", []):
        pieces = " + ".join(f"{p['count']}-pc {p['set']}" for p in s.get("pieces", []))
        lines.append(f"[{s['tier']}] {pieces}: {s.get('explanation', '')}")

    four = art.get("four_star_option")
    if four and four.get("name"):
        lines.append(
            f"Best 4-star artifact: {four['name']}. "
            f"2-pc: {four.get('two_pc_bonus', '')}. "
            f"4-pc: {four.get('four_pc_bonus', '')}."
        )

    sp = art.get("stat_priority", {})
    if sp.get("sands"):
        lines.append(
            f"Artifact stat priority — Sands: {sp['sands']} | "
            f"Goblet: {sp.get('goblet', '?')} | Circlet: {sp.get('circlet', '?')}."
        )
    if sp.get("substats"):
        lines.append("Substat priority: " + " > ".join(sp["substats"]) + ".")
    for note in sp.get("notes", []):
        lines.append(note)

    gs = art.get("goal_stats", {})
    if gs:
        lines.append("Target stat values: " + " | ".join(f"{k}: {v}" for k, v in gs.items()) + ".")
    for note in art.get("goal_stat_notes", []):
        lines.append(note)

    return "\n".join(lines)


def flatten_weapons(data: dict) -> str:
    wep = data["weapons"]
    char = data["character_name"]
    lines: list[str] = [f"Weapon recommendations for {char}:"]

    for w in wep.get("ranked", []):
        lines.append(f"[{w['tier']}] {w['name']}: {w.get('explanation', '')}")

    alts = wep.get("alternatives_from_build_card", [])
    if alts:
        lines.append("Other weapons to consider: " + ", ".join(alts) + ".")

    return "\n".join(lines)


def flatten_talents(data: dict) -> str:
    tal = data["talents"]
    char = data["character_name"]
    lines: list[str] = [f"Talent priority and kit for {char}:"]

    order = tal.get("priority_order", [])
    if order:
        order_str = " > ".join(f"{e['ability']} ({e['rank']})" for e in order)
        lines.append(f"Talent leveling priority: {order_str}.")
    if tal.get("priority_rationale"):
        lines.append(tal["priority_rationale"])

    for p in tal.get("passives", []):
        tag  = f"[{p['type'].upper()}]" if p.get("type") else ""
        name = p.get("name") or p["slot"]
        desc = p.get("description") or ""
        lines.append(f"{p['slot']} — {name} {tag}: {desc}")

    return "\n".join(lines)


def flatten_teams(data: dict) -> str:
    teams = data["teams"]
    char  = data["character_name"]
    lines: list[str] = [f"Team compositions for {char}:"]

    for comp in teams.get("comps", []):
        members_str = ", ".join(
            f"{m['name']}" + (f" ({m['role']})" if m.get("role") else "")
            for m in comp.get("members", [])
        )
        lines.append(f"\n{comp['name']} [{comp.get('element', '')}]:")
        lines.append(f"  Members: {members_str}")
        if comp.get("summary"):
            lines.append(f"  {comp['summary']}")
        if comp.get("alternates"):
            lines.append(f"  Alternates: {comp['alternates']}")

    if teams.get("flex_teammates"):
        lines.append("\nFlex / situational teammates:")
        for t in teams["flex_teammates"]:
            lines.append(f"  {t['name']} ({t['role']}): {t.get('explanation', '')}")

    return "\n".join(lines)


def flatten_constellations(data: dict) -> str:
    cons = data["constellations"]
    char = data["character_name"]
    lines: list[str] = [f"Constellations for {char}:"]

    for c in cons.get("levels", []):
        worth = (
            " ✓ WORTH PULLING" if c.get("worth_pulling") is True
            else " ✗ not priority" if c.get("worth_pulling") is False
            else ""
        )
        rating = f" [{c['rating']}]" if c.get("rating") else ""
        lines.append(f"\n{c['level']} — {c.get('name', '')}{rating}{worth}")
        lines.append(f"  Effect: {c.get('effect', '')}")
        for merit in c.get("merits", []):
            lines.append(f"  • {merit}")

    if cons.get("pull_recommendation"):
        lines.append(f"\nPull note: {cons['pull_recommendation']}")

    return "\n".join(lines)


def flatten_materials(data: dict) -> str:
    mat  = data["materials"]
    char = data["character_name"]
    lines: list[str] = [f"Ascension and talent materials for {char}:"]

    asc = mat.get("ascension", {})
    lines.append("\nAscension materials:")
    if asc.get("local_specialty"): lines.append(f"  Local specialty: {_fmt(asc['local_specialty'])}")
    if asc.get("ascension_gems"):  lines.append(f"  Ascension gems: {_fmt(asc['ascension_gems'])}")
    if asc.get("enemy_drops"):     lines.append(f"  Enemy drops: {_fmt(asc['enemy_drops'])}")
    if asc.get("mora"):            lines.append(f"  Mora: {_fmt(asc['mora'])}")

    tal = mat.get("talents", {})
    lines.append("\nTalent materials (per talent; ×3 total for all talents):")
    if tal.get("talent_books"):     lines.append(f"  Talent books: {_fmt(tal['talent_books'])}")
    if tal.get("weekly_boss_drop"): lines.append(f"  Weekly boss drop: {_fmt(tal['weekly_boss_drop'])}")
    if tal.get("enemy_drops"):      lines.append(f"  Enemy drops: {_fmt(tal['enemy_drops'])}")
    if tal.get("crown"):            lines.append(f"  Crown of Insight: {_fmt(tal['crown'])}")
    if tal.get("mora"):             lines.append(f"  Mora: {_fmt(tal['mora'])}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Maps JSON section key → flattener.  Order also determines chunk order.
FLATTENERS: dict[str, callable] = {
    "overview":       flatten_overview,
    "artifacts":      flatten_artifacts,
    "weapons":        flatten_weapons,
    "talents":        flatten_talents,
    "teams":          flatten_teams,
    "constellations": flatten_constellations,
    "materials":      flatten_materials,
}
