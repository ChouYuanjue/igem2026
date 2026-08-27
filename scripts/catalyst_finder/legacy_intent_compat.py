from __future__ import annotations

import re

from scripts.catalyst_finder.task_contracts import VALID_TASK_HINTS

# Compatibility-only lexical classifier. The production agent does not feed these
# guesses into the semantic planner; deterministic authority is limited to explicit
# structured inputs/IDs and explicit UI task choices.
ROUTE_DESIGN_INTENT_RE = re.compile(
    r"(?:推荐|生成|设计|规划|寻找|找|给我|有哪些|列出|排序|比较)[\s\S]{0,120}(?:候选)?(?:合成|生物合成|代谢|反应)?(?:路线|路径|线路)|(?:候选|合成|生物合成|代谢)(?:路线|路径)[\s\S]{0,80}(?:推荐|排序|比较|设计|规划)|(?:route|pathway)[\s\S]{0,80}(?:design|recommend|rank|generate|search|plan)|retrosynth",
    re.IGNORECASE,
)
ROUTE_ROLE_PAIR_RE = re.compile(
    r"(?:起始前体|路线起点|starting\s+precursor|route\s+start)[\s\S]{0,160}(?:目标产物|路线终点|target\s+product|route\s+target)|(?:目标产物|路线终点|target\s+product|route\s+target)[\s\S]{0,160}(?:起始前体|路线起点|starting\s+precursor|route\s+start)",
    re.IGNORECASE,
)
SINGLE_REACTION_INTENT_RE = re.compile(
    r"(?:目标反应|单步反应|single[- ]?step\s+reaction)|(?:底物|substrate)[\s\S]{0,120}(?:产物|product)|(?:转化为|转变为|催化.{0,16}(?:生成|形成)|convert(?:s|ed|ing)?\s+.{0,80}\s+to)",
    re.IGNORECASE,
)
PATHWAY_INTENT_RE = re.compile(
    r"(?:完整.{0,6}(?:路径|线路)|整条.{0,6}(?:路径|线路)|反应.{0,5}(?:路径|线路)|多步反应|每一步|级联|串联|cascade|one[- ]?pot|一锅|多酶.{0,6}(?:兼容|冲突)|酶.{0,6}(?:兼容|冲突)|条件.{0,6}(?:冲突|兼容)|沉淀|沉降)",
    re.IGNORECASE,
)
PATHWAY_ARROW_RE = re.compile(r"(?:→|->)[\s\S]{0,500}(?:→|->)")
FOLLOWUP_REACTION_ONLY_RE = re.compile(
    r"(?:只看|只要|仅看|只列|只关注).{0,40}(?:潜在反应|可能反应|反应|催化反应)|(?:不要|不需要).{0,20}(?:路线|路径)",
    re.IGNORECASE,
)
FOLLOWUP_ENZYME_ONLY_RE = re.compile(
    r"(?:只看|只要|仅看|只列).{0,40}(?:候选酶|酶|催化剂)",
    re.IGNORECASE,
)


def classify_task_intent(text: str, direction_hint: str = "auto") -> str | None:
    """Historical lexical task classifier retained only for compatibility tooling."""
    value = str(text or "").strip()
    hint = direction_hint if direction_hint in VALID_TASK_HINTS else "auto"
    if FOLLOWUP_REACTION_ONLY_RE.search(value):
        return "enzyme_to_reaction"
    if FOLLOWUP_ENZYME_ONLY_RE.search(value):
        return "reaction_to_enzyme"
    if hint in {"reaction_to_enzyme", "enzyme_to_reaction"}:
        return hint
    if hint == "route_design":
        if PATHWAY_ARROW_RE.search(value):
            return "pathway_compatibility"
        if SINGLE_REACTION_INTENT_RE.search(value) and not (
            ROUTE_DESIGN_INTENT_RE.search(value) or ROUTE_ROLE_PAIR_RE.search(value)
        ):
            return "reaction_to_enzyme"
        return "route_design"
    if hint == "pathway_compatibility":
        if ROUTE_DESIGN_INTENT_RE.search(value) and not (
            PATHWAY_ARROW_RE.search(value) or PATHWAY_INTENT_RE.search(value)
        ):
            return "route_design"
        if SINGLE_REACTION_INTENT_RE.search(value) and not (
            PATHWAY_ARROW_RE.search(value) or PATHWAY_INTENT_RE.search(value)
        ):
            return "reaction_to_enzyme"
        return "pathway_compatibility"
    if PATHWAY_ARROW_RE.search(value):
        return "pathway_compatibility"
    if ROUTE_DESIGN_INTENT_RE.search(value):
        return "route_design"
    if PATHWAY_INTENT_RE.search(value):
        return "pathway_compatibility"
    if ROUTE_ROLE_PAIR_RE.search(value):
        return "route_design"
    if SINGLE_REACTION_INTENT_RE.search(value):
        return "reaction_to_enzyme"
    return None
