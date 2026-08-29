from __future__ import annotations

from typing import Any


CAPABILITY_MANIFEST: dict[str, Any] = {
    "version": "catalyst-capabilities-v7",
    "interaction": {
        "model_led": True,
        "natural_language_first": True,
        "supports_follow_up": True,
        "markdown_responses": True,
        "conversation_hint_en": "Ask a question directly, or ask what the agent can do.",
        "conversation_hint_zh": "直接提问，也可以先问智能体能做什么。",
        "guide_note_en": "Describe the research goal directly. Verified objects stay available in this conversation, so later questions can refer to this enzyme, the second paper, or a previously confirmed target.",
        "guide_note_zh": "直接描述研究目标。已经核对的对象会保留在当前对话中，后续可以继续说“这个酶”“第二篇文献”或“刚才确认的目标”。",
        "structured_inputs": ["RHEA ID", "UniProt accession", "Reaction SMILES", "FASTA", "amino-acid sequence"],
    },
    "groups": [
        {
            "id": "research_workspace",
            "title_en": "Scientific research workspace",
            "title_zh": "科研资料工作区",
            "description_en": "Compose the research modules that matter for the current question: annotations, structures, literature, recorded relationships, model analysis and next-step priorities.",
            "description_zh": "围绕已核对的蛋白或反应，按本轮问题组合注释、结构、文献、已记录关系、模型分析和下一步优先级。",
            "examples": [
                {
                    "title_en": "Choose any research combination",
                    "title_zh": "自由组合资料模块",
                    "description_en": "Ask for any subset, such as literature + structures, recorded relationships + model analysis, or annotations only.",
                    "description_zh": "可任意组合，例如只看文献与结构、只看已记录关系与模型分析，或只查数据库注释。",
                    "prompt_en": "For UniProt P00338, only show literature and structures.",
                    "prompt_zh": "针对 UniProt P00338，只查文献和结构信息。",
                },
                {
                    "title_en": "Research one enzyme end to end",
                    "title_zh": "完整研究一个酶",
                    "description_en": "Collect UniProt/InterPro/literature evidence, recorded reactions, model recovery and the next unrecorded frontier.",
                    "description_zh": "汇集 UniProt、InterPro、文献和已记录反应，同时查看模型对已知关系的回收与下一批新关联候选。",
                    "prompt_en": "Research UniProt P00338 for me: annotations, domains, literature, recorded reactions, and what the model thinks is worth testing next.",
                    "prompt_zh": "完整查一下 UniProt P00338：注释、结构域、文献、已记录反应，以及模型认为下一步最值得验证什么。",
                },
                {
                    "title_en": "Research one reaction",
                    "title_zh": "围绕一个反应做资料检索",
                    "description_en": "Keep recorded enzymes and the model frontier on the same verified reaction target.",
                    "description_zh": "围绕同一个已核对反应同时整理已记录酶、外部资料和模型扩展空间。",
                    "prompt_en": "Give me a research workspace for RHEA:54512, including recorded enzymes and the model frontier.",
                    "prompt_zh": "围绕 RHEA:54512 给我一份科研资料工作区，包含已记录酶和模型扩展空间。",
                },
            ],
        },
        {
            "id": "evidence",
            "title_en": "Database evidence and entity relationships",
            "title_zh": "数据库证据与实体关系",
            "description_en": "Verify reactions, proteins, families and functional classes; inspect verified records; list scope members; query recorded enzyme–reaction relationships; and compare verified entities.",
            "description_zh": "核对反应、蛋白、家族与功能类别，查看实体详情和范围成员，查询数据库已记录的酶–反应关系，并比较多个已核对实体。",
            "examples": [
                {
                    "title_en": "Reaction → recorded enzymes",
                    "title_zh": "查询反应的已记录酶",
                    "description_en": "Retrieve database-recorded enzymes for a verified Rhea reaction and combine protein-family or organism constraints when useful.",
                    "description_zh": "查询已核对 Rhea 反应的数据库记录酶，并可结合蛋白家族或物种条件。",
                    "prompt_en": "Which enzymes are database-recorded for RHEA:54512?",
                    "prompt_zh": "RHEA:54512 在数据库里记录了哪些催化酶？",
                },
                {
                    "title_en": "Protein → recorded reactions",
                    "title_zh": "查询蛋白的已记录反应",
                    "description_en": "Read the reverse evidence index for one concrete protein.",
                    "description_zh": "从反向证据索引查询一个具体蛋白已经记录的反应。",
                    "prompt_en": "Which reactions are database-recorded for UniProt P00330?",
                    "prompt_zh": "UniProt P00330 已记录能催化哪些反应？",
                },
                {
                    "title_en": "Family or functional-class evidence",
                    "title_zh": "家族与功能类证据",
                    "description_en": "Summarize recorded reactions across an auditable family or a search-derived functional-class cohort.",
                    "description_zh": "按可审计家族或检索得到的功能类成员集合汇总已记录反应。",
                    "prompt_en": "What reactions are recorded for cytochrome P450 enzymes in the current evidence base?",
                    "prompt_zh": "当前证据库里，细胞色素 P450 已记录能催化哪些反应？",
                },
                {
                    "title_en": "List concrete scope members",
                    "title_zh": "查看具体成员",
                    "description_en": "List proteins included in a verified family or functional-class scope and show current model-candidate coverage.",
                    "description_zh": "列出已核对家族或功能类范围中的具体蛋白，并显示当前模型候选库覆盖情况。",
                    "prompt_en": "Give me ten concrete members from this P450 scope and show model-candidate coverage.",
                    "prompt_zh": "列出这个 P450 范围里的 10 个具体成员，并显示模型候选库覆盖情况。",
                },
                {
                    "title_en": "Inspect a verified record",
                    "title_zh": "查看实体详情",
                    "description_en": "Read the verified record for a Rhea reaction, UniProt protein, protein scope or resolved compound.",
                    "description_zh": "查看已核对的 Rhea 反应、UniProt 蛋白、蛋白范围或化合物记录。",
                    "prompt_en": "What exactly is RHEA:23444? Show its verified record details.",
                    "prompt_zh": "RHEA:23444 具体是什么反应？给我记录详情。",
                },
                {
                    "title_en": "Compare verified entities",
                    "title_zh": "比较已核对实体",
                    "description_en": "Compare structured fields from multiple verified reactions, proteins or compounds.",
                    "description_zh": "并排比较多个已核对反应、蛋白或化合物的结构化字段。",
                    "prompt_en": "Compare RHEA:23444 and RHEA:54512 using their verified records.",
                    "prompt_zh": "基于已核对记录比较 RHEA:23444 和 RHEA:54512。",
                },
            ],
        },
        {
            "id": "compound_identity",
            "title_en": "Compound identity and ChEBI resolution",
            "title_zh": "化合物身份与 ChEBI 核对",
            "description_en": "Resolve biochemical names and naming variants against the local Rhea/ChEBI index, then reuse verified compounds in later questions and route tasks.",
            "description_zh": "用本地 Rhea/ChEBI 索引核对生化名称及命名变体，并在后续问题和路线任务中复用已核对化合物。",
            "examples": [
                {
                    "title_en": "Resolve a biochemical name",
                    "title_zh": "核对生化名称",
                    "description_en": "Find the matching ChEBI record and local structure information.",
                    "description_zh": "查询对应的 ChEBI 记录与本地结构信息。",
                    "prompt_en": "Which ChEBI record corresponds to p-coumaric acid?",
                    "prompt_zh": "对香豆酸对应哪个 ChEBI 记录？",
                },
                {
                    "title_en": "Compare compounds",
                    "title_zh": "比较化合物",
                    "description_en": "Resolve several compounds and compare their verified names and structures.",
                    "description_zh": "核对多个化合物并比较名称与结构信息。",
                    "prompt_en": "Resolve p-coumaric acid and caffeic acid, then compare their structures.",
                    "prompt_zh": "分别核对对香豆酸和咖啡酸，再比较它们的结构。",
                },
            ],
        },
        {
            "id": "candidate_retrieval",
            "title_en": "Candidate enzyme and reaction discovery",
            "title_zh": "模型扩展与实验优先级",
            "description_en": "Move from the evidence-backed research workspace into deeper model ranking: prioritize unrecorded associations, explore sequence-diverse candidates, and turn database gaps into testable shortlists.",
            "description_zh": "从资料与证据继续向前，用模型排序尚未记录的关联、寻找序列更远的候选，并把数据库空白转成可直接验证的实验短名单。",
            "examples": [
                {
                    "title_en": "Reaction → candidate enzymes",
                    "title_zh": "从反应寻找候选酶",
                    "description_en": "Use a Rhea reaction, reaction description or Reaction SMILES as the query.",
                    "description_zh": "可使用 Rhea 反应、自然语言反应描述或 Reaction SMILES 作为查询。",
                    "prompt_en": "For RHEA:54512, show recorded enzymes and rank ten additional candidate associations.",
                    "prompt_zh": "针对 RHEA:54512，展示已记录酶并排序 10 个其他候选关联。",
                },
                {
                    "title_en": "Protein → candidate reactions",
                    "title_zh": "从蛋白探索候选反应",
                    "description_en": "Use a UniProt protein, FASTA or amino-acid sequence as the query.",
                    "description_zh": "可使用 UniProt 蛋白、FASTA 或氨基酸序列作为查询。",
                    "prompt_en": "For UniProt P00330, show recorded reactions and rank ten additional reaction candidates.",
                    "prompt_zh": "对于 UniProt P00330，展示已记录反应并排序 10 个其他反应候选。",
                },
                {
                    "title_en": "Open-world structured input",
                    "title_zh": "开放世界结构化输入",
                    "description_en": "Paste Reaction SMILES, FASTA or an amino-acid sequence directly into the conversation.",
                    "description_zh": "可直接在对话中粘贴 Reaction SMILES、FASTA 或氨基酸序列。",
                    "prompt_en": "I will paste a Reaction SMILES or FASTA sequence. Analyze it and continue with the relevant scientific task.",
                    "prompt_zh": "我会粘贴 Reaction SMILES 或 FASTA 序列，请分析并继续完成相关科学任务。",
                },
            ],
        },
        {
            "id": "route_design",
            "title_en": "Biosynthetic route design",
            "title_zh": "生物合成路线设计",
            "description_en": "Search the Rhea reaction graph, then add only the analyses requested for this turn. Thermodynamic MDF, E. coli host flux and predicted-transformation exploration run on demand rather than as mandatory route-search steps.",
            "description_zh": "先搜索 Rhea 反应图，再按本轮目标加入热力学 MDF、E. coli 宿主通量或预测转化扩展。",
            "examples": [
                {
                    "title_en": "Search the Rhea route graph",
                    "title_zh": "只搜索路线",
                    "description_en": "Search and rank Rhea-supported routes as the base route-design step; add MDF or host-flux analysis when useful.",
                    "description_zh": "以 Rhea 已记录反应图搜索和排序候选路线，后续可按需加入 MDF 或宿主通量分析。",
                    "prompt_en": "Give me five Rhea-supported routes from L-tyrosine to caffeate, ranked by route length first.",
                    "prompt_zh": "给我 5 条从 L-酪氨酸到咖啡酸的 Rhea 已记录路线，先按路线长度排序。",
                },
                {
                    "title_en": "Design and rank routes",
                    "title_zh": "设计并比较候选路线",
                    "description_en": "State the source, target and ranking priorities in the request.",
                    "description_zh": "在请求中说明起点、终点和路线偏好。",
                    "prompt_en": "Design biosynthetic routes from L-tyrosine to caffeate, prioritizing recorded reactions and thermodynamic feasibility.",
                    "prompt_zh": "设计从 L-酪氨酸到咖啡酸的生物合成路线，优先已记录反应和热力学可行性。",
                },
            ],
        },
        {
            "id": "pathway",
            "title_en": "Whole-pathway enzyme compatibility",
            "title_zh": "整条路径的多酶兼容性",
            "description_en": "Jointly select enzymes for a specified multi-step pathway, using only the compatibility dimensions requested for this turn. pH, temperature, cofactors, localization and cross-step activity can be combined freely; model-only joint selection skips condition lookups.",
            "description_zh": "为指定多步路径联合选择酶，并按本轮目标组合 pH、温度、辅因子、定位和跨步活性等兼容性证据；也可先按模型完成整路联合选择。",
            "examples": [
                {
                    "title_en": "Choose compatibility dimensions",
                    "title_zh": "选择兼容性维度",
                    "description_en": "Ask for any subset of pH, temperature, cofactors, localization or cross-step activity, or request model-only joint enzyme selection.",
                    "description_zh": "可只看 pH、温度、辅因子、定位、跨步活性中的任意组合，也可以只按模型联合选酶。",
                    "prompt_en": "For this pathway, only compare cofactor and temperature compatibility across the enzyme combination.",
                    "prompt_zh": "这条路径只比较酶组合的辅因子和温度兼容性。",
                },
                {
                    "title_en": "Evaluate a multi-step pathway",
                    "title_zh": "评估多步路径",
                    "description_en": "Provide the pathway and any fixed enzymes or target conditions.",
                    "description_zh": "提供路径以及已固定的酶或目标条件。",
                    "prompt_en": "Evaluate L-tyrosine → p-coumarate → caffeate and identify a compatible enzyme combination for 30 °C operation.",
                    "prompt_zh": "评估 L-酪氨酸 → 对香豆酸 → 咖啡酸，并寻找适合 30 °C 条件的酶组合。",
                },
            ],
        },
    ],
}


# Keep the folded capability map comprehensive without turning the first screen into a menu.
# These examples are all implemented by the same model-led controller and scientific tools.
_EXTRA_CAPABILITY_EXAMPLES: dict[str, list[dict[str, str]]] = {
    "research_workspace": [
        {
            "title_en": "Protein structures",
            "title_zh": "实验结构与预测结构",
            "description_en": "Collect available PDB experimental structures and AlphaFold models for one verified protein.",
            "description_zh": "汇集一个已核对蛋白可用的 PDB 实验结构和 AlphaFold 预测结构。",
            "prompt_en": "For this protein, only show the available experimental structures and AlphaFold models.",
            "prompt_zh": "只查这个蛋白可用的实验结构和 AlphaFold 模型。",
        },
        {
            "title_en": "Curated and broader literature",
            "title_zh": "关联文献与扩展检索",
            "description_en": "Start from references linked by UniProt or Rhea when available, then inspect the returned papers in later turns.",
            "description_zh": "优先查看 UniProt 或 Rhea 关联文献，并可在后续继续打开某一篇记录。",
            "prompt_en": "For this target, show the linked literature and the most useful surrounding papers.",
            "prompt_zh": "查这个目标的关联文献，并补充最有用的相关研究。",
        },
        {
            "title_en": "Annotations and cross-database links",
            "title_zh": "注释与跨数据库入口",
            "description_en": "Combine UniProt and InterPro annotations with available cross-references such as domains, pathways, structures and biochemical resources.",
            "description_zh": "整合 UniProt、InterPro 注释，以及可用的结构域、通路、结构和生化数据库交叉入口。",
            "prompt_en": "Summarize the annotations and cross-database records for this protein.",
            "prompt_zh": "整理这个蛋白的数据库注释和可用的交叉数据库记录。",
        },
        {
            "title_en": "Follow a returned paper",
            "title_zh": "继续追一篇文献",
            "description_en": "A paper returned by the workspace remains a verified session object for later inspection.",
            "description_zh": "工作区返回的文献会作为已核对对象保留，可继续查看某一篇的记录。",
            "prompt_en": "Open the second paper from the previous result and show its verified record.",
            "prompt_zh": "打开刚才结果里的第二篇文献，给我看它的记录。",
        },
    ],
    "evidence": [
        {
            "title_en": "Database-only evidence",
            "title_zh": "只看数据库事实",
            "description_en": "Return the recorded relationship layer alone when a strict evidence-only answer is useful.",
            "description_zh": "需要严格事实层时，可只返回数据库已记录关系。",
            "prompt_en": "For this target, only show database-recorded relationships and their sources.",
            "prompt_zh": "这个目标只看数据库已记录关系和来源。",
        },
        {
            "title_en": "Relation with family constraints",
            "title_zh": "结合家族条件查关系",
            "description_en": "Intersect a verified reaction with a protein family, functional class or organism constraint.",
            "description_zh": "把已核对反应与蛋白家族、功能类别或物种条件结合查询。",
            "prompt_en": "For this reaction, which database-recorded enzymes also fall in the UbiA family?",
            "prompt_zh": "这个反应的已记录催化酶里，哪些同时属于 UbiA 家族？",
        },
    ],
    "candidate_retrieval": [
        {
            "title_en": "Known-relation recovery + frontier",
            "title_zh": "先回看已知，再看模型前沿",
            "description_en": "On a project-aligned target, inspect whether official recorded relationships rank near the top before using the unrecorded frontier for experiments.",
            "description_zh": "对领域内目标先看模型能否把官方已知关系排到前面，再用未记录前沿安排实验优先级。",
            "prompt_en": "Show the recorded relationships, the model recovery check, and the top unrecorded frontier for this target.",
            "prompt_zh": "把这个目标的已记录关系、模型回看和最优先的未记录前沿放在一起看。",
        },
        {
            "title_en": "Add biological constraints",
            "title_zh": "加入物种或家族条件",
            "description_en": "Apply biological constraints while ranking new associations instead of filtering them manually afterwards.",
            "description_zh": "在模型排序阶段直接加入物种、家族等生物学条件。",
            "prompt_en": "Rank ten candidate enzymes for this reaction, focusing on fungal proteins and excluding recorded associations.",
            "prompt_zh": "给这个反应排序 10 个候选酶，优先真菌蛋白，并排除已记录关联。",
        },
    ],
    "route_design": [
        {
            "title_en": "Thermodynamic route analysis",
            "title_zh": "路线热力学分析",
            "description_en": "Run eQuilibrator MDF only when thermodynamic driving force is relevant to the current route decision.",
            "description_zh": "需要比较路线驱动力时，可按需运行 eQuilibrator MDF。",
            "prompt_en": "For the top routes, calculate MDF and compare their thermodynamic driving force.",
            "prompt_zh": "对前几条路线计算 MDF，并比较热力学驱动力。",
        },
        {
            "title_en": "E. coli host-flux check",
            "title_zh": "大肠杆菌宿主通量检查",
            "description_en": "Use the iML1515 host model to flag routes with no supported route flux under the requested growth constraint.",
            "description_zh": "用 iML1515 宿主模型检查候选路线在目标生长约束下的可承载通量。",
            "prompt_en": "Check the candidate routes with E. coli iML1515 and remove zero-flux routes.",
            "prompt_zh": "用大肠杆菌 iML1515 检查这些候选路线，并过滤零通量路线。",
        },
        {
            "title_en": "Explore missing transformations",
            "title_zh": "探索数据库之外的转化桥接",
            "description_en": "When the Rhea graph is insufficient, explore rule-predicted MINE/Pickaxe bridges and keep them visibly separate from recorded Rhea steps.",
            "description_zh": "Rhea 已知图不足时，可探索 MINE/Pickaxe 规则预测的桥接转化，并与已记录 Rhea 步骤分开呈现。",
            "prompt_en": "If Rhea alone cannot connect the route, also explore predicted transformation bridges.",
            "prompt_zh": "如果 Rhea 已知反应无法连通，也探索预测转化桥接。",
        },
    ],
    "pathway": [
        {
            "title_en": "Model-only joint enzyme selection",
            "title_zh": "只按模型联合选酶",
            "description_en": "Jointly choose enzymes across all steps from model priorities first, with condition evidence available as a later layer.",
            "description_zh": "可先按模型优先级跨步骤联合选酶，再按需要补充条件证据。",
            "prompt_en": "For this pathway, first choose the enzyme combination jointly from model priorities.",
            "prompt_zh": "这条路径先按模型优先级联合选择整组酶。",
        },
        {
            "title_en": "Fix some steps, optimize the rest",
            "title_zh": "固定部分酶，再联合选择其余步骤",
            "description_en": "Keep user-specified enzymes fixed and optimize candidates only for the unresolved pathway steps.",
            "description_zh": "保留用户已经指定的酶，只为其余步骤联合选择候选。",
            "prompt_en": "Keep the enzyme I specified for step 1, and jointly select the remaining pathway enzymes.",
            "prompt_zh": "固定我指定的第 1 步酶，再联合选择其余步骤的酶。",
        },
        {
            "title_en": "Condition and localization compatibility",
            "title_zh": "条件与定位兼容性",
            "description_en": "Combine any requested subset of pH, temperature, cofactors, localization and cross-step activity evidence.",
            "description_zh": "可组合检查 pH、温度、辅因子、亚细胞定位和跨步活性中的任意维度。",
            "prompt_en": "For this enzyme combination, check temperature, cofactors and localization compatibility only.",
            "prompt_zh": "这组酶只检查温度、辅因子和亚细胞定位兼容性。",
        },
    ],
}

for _group in CAPABILITY_MANIFEST["groups"]:
    _group["examples"].extend(_EXTRA_CAPABILITY_EXAMPLES.get(str(_group.get("id") or ""), []))


def public_capabilities() -> dict[str, Any]:
    """Return a JSON-safe copy used by both the controller and the frontend."""
    import copy

    return copy.deepcopy(CAPABILITY_MANIFEST)
