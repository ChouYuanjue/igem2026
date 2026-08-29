from __future__ import annotations

from typing import Any


CAPABILITY_MANIFEST: dict[str, Any] = {
    "version": "catalyst-capabilities-v5",
    "interaction": {
        "model_led": True,
        "natural_language_first": True,
        "supports_follow_up": True,
        "markdown_responses": True,
        "conversation_hint_en": "Ask a question directly, or ask what the agent can do.",
        "conversation_hint_zh": "直接提问，也可以先问智能体能做什么。",
        "structured_inputs": ["RHEA ID", "UniProt accession", "Reaction SMILES", "FASTA", "amino-acid sequence"],
    },
    "groups": [
        {
            "id": "research_workspace",
            "title_en": "Scientific research workspace",
            "title_zh": "科研资料工作区",
            "description_en": "Start from a verified protein or reaction, then bring together current database annotations, literature, recorded relationships and a model lens in one workflow. External sources are queried on demand rather than mirrored locally.",
            "description_zh": "从已核对的蛋白或反应出发，把实时数据库注释、文献、已记录关系和模型视角放进同一条工作流；外部资料按需查询，不依赖本地整库镜像。",
            "examples": [
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
            "description_en": "Search the Rhea reaction graph and rank routes using step count, enzyme availability, thermodynamics, project-model coverage and E. coli host-flux evidence. Predicted transformations can be explored as a separate evidence layer.",
            "description_zh": "搜索 Rhea 反应图，并结合步骤数、酶可获得性、热力学、项目模型覆盖和 E. coli 宿主通量排序路线。预测转化作为独立证据层展示。",
            "examples": [
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
            "description_en": "Evaluate a specified multi-step pathway, jointly select missing enzymes, and summarize compatibility evidence for pH, temperature, cofactors and operating conditions.",
            "description_zh": "评估指定的多步路径，联合选择缺失酶，并整理 pH、温度、辅因子与运行条件的兼容性证据。",
            "examples": [
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


def public_capabilities() -> dict[str, Any]:
    """Return a JSON-safe copy used by both the controller and the frontend."""
    import copy

    return copy.deepcopy(CAPABILITY_MANIFEST)
