from collections.abc import Sequence
from typing import TypedDict

RISK_TERMS = (
    "careful",
    "warning",
    "caution",
    "do not",
    "avoid",
    "battery",
    "puncture",
    "heat",
    "adhesive",
    "fragile",
    "connector",
    "cable",
    "glass",
)

ACTION_VERBS = (
    "remove",
    "unscrew",
    "disconnect",
    "pry",
    "lift",
    "pull",
    "peel",
    "separate",
    "detach",
    "replace",
)

GUIDE_ANALYSIS_TEMPLATE = """
Complexity: {complexity_label} (score {complexity_score})
Total steps: {num_steps}
Total tools: {num_tools}
Risk indicators: {risk_terms}
Common actions: {common_actions}
"""


class GuideAnalysis(TypedDict):
    num_steps: int
    num_tools: int
    risk_terms: list[str]
    action_counts: dict[str, int]
    complexity_score: int
    complexity_label: str


def analyse_guide(steps: Sequence[str], tools: Sequence[str]) -> GuideAnalysis:
    full_text = " ".join(steps).lower()

    risk_terms = [term for term in RISK_TERMS if term in full_text]
    action_counts = {verb: full_text.count(verb) for verb in ACTION_VERBS}

    complexity_score = (
        len(steps) + len(tools) * 2 + len(risk_terms) * 3 + sum(action_counts.values())
    )

    if complexity_score < 20:
        complexity_label = "low"
    elif complexity_score < 50:
        complexity_label = "medium"
    else:
        complexity_label = "high"

    return {
        "num_steps": len(steps),
        "num_tools": len(tools),
        "risk_terms": risk_terms,
        "action_counts": action_counts,
        "complexity_score": complexity_score,
        "complexity_label": complexity_label,
    }


def format_guide_analysis(analysis: GuideAnalysis) -> str:
    risk_terms = ", ".join(analysis["risk_terms"]) or "none detected"

    common_actions = ", ".join(
        f"{verb} ({count})"
        for verb, count in analysis["action_counts"].items()
        if count > 0
    )

    return GUIDE_ANALYSIS_TEMPLATE.format(
        complexity_label=analysis["complexity_label"],
        complexity_score=analysis["complexity_score"],
        num_steps=analysis["num_steps"],
        num_tools=analysis["num_tools"],
        risk_terms=risk_terms,
        common_actions=common_actions or "none detected",
    ).strip()
