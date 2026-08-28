from __future__ import annotations

from typing import Any


CAPABILITY_MANIFEST: dict[str, Any] = {
    "version": "catalyst-capabilities-v3",
    "interaction": {
        "model_led": True,
        "natural_language_first": True,
        "supports_follow_up": True,
        "markdown_responses": True,
        "structured_inputs": ["RHEA ID", "UniProt accession", "Reaction SMILES", "FASTA", "amino-acid sequence"],
    },
    "groups": [
        {
            "id": "conversation",
            "title_en": "Ask and refine naturally",
            "title_zh": "自然提问与连续追问",
            "description_en": "Ask what the system can do, discuss a scientific goal, or refine a previous result. The model decides whether to answer, ask a concrete question, or use scientific tools.",
            "description_zh": "可以询问系统能力、讨论科研目标，也可以围绕上一轮结果自然追问。模型会自行决定直接回答、提出具体澄清问题，还是调用科学工具。",
            "examples": [
                {
                    "title_en": "What can you do?",
                    "title_zh": "你有哪些功能？",
                    "description_en": "Get a current capability overview from the same manifest used by the interface.",
                    "description_zh": "直接询问当前能力，回答与页面展示使用同一份能力清单。",
                    "prompt_en": "What can you do, and what kinds of biochemical research questions can I ask you?",
                    "prompt_zh": "你有哪些功能？我可以怎样自然地向你提出生化研究问题？",
                },
                {
                    "title_en": "Continue from the last result",
                    "title_zh": "围绕上一轮继续追问",
                    "description_en": "Verified entities from the current session can be reused without restating every identifier.",
                    "description_zh": "当前会话里已经核对过的实体可以被后续自然引用，不必重复所有编号。",
                    "prompt_en": "Among those, which ones belong to the enzyme family I mentioned earlier?",
                    "prompt_zh": "刚才这些里面，哪些属于我前面提到的那个酶家族？",
                },
            ],
        },
        {
            "id": "evidence",
            "title_en": "Database evidence and relationship queries",
            "title_zh": "数据库证据与关系查询",
            "description_en": "Resolve and inspect reactions, proteins, families or functional classes, list auditable members, and query recorded enzyme–reaction relationships without turning a family into one representative protein.",
            "description_zh": "核对并查看反应、蛋白、家族或功能类别详情，列出可审计成员，并查询数据库已经记录的酶–反应关系；家族问题不会再被压缩成某一个代表蛋白。",
            "examples": [
                {
                    "title_en": "Which enzyme is recorded for this reaction?",
                    "title_zh": "这个反应具体由哪个酶催化？",
                    "description_en": "Combine reaction and protein/family constraints in one relationship query.",
                    "description_zh": "可以把反应、蛋白家族、物种等条件组合成关系查询。",
                    "prompt_en": "Which recorded enzyme in the UbiA-type family catalyzes RHEA:74587?",
                    "prompt_zh": "催化 RHEA:74587 的 UbiA 型萜环化酶具体是哪个？",
                },
                {
                    "title_en": "Summarize a functional class",
                    "title_zh": "汇总一个酶功能类别",
                    "description_en": "Aggregate recorded reactions across an auditable family or search-derived functional-class scope.",
                    "description_zh": "按可审计家族或检索得到的功能类成员集合汇总已记录反应。",
                    "prompt_en": "What reactions are recorded for cytochrome P450 enzymes in the current evidence base?",
                    "prompt_zh": "当前证据库里，细胞色素 P450 这一类酶已经记录能催化哪些反应？",
                },
                {
                    "title_en": "Inspect a verified record",
                    "title_zh": "查看已核对实体详情",
                    "description_en": "Inspect the record itself—such as a Rhea reaction, a concrete UniProt protein, a family scope or a resolved compound—without launching an unrelated relation or prediction workflow.",
                    "description_zh": "直接查看已经核对的 Rhea 反应、具体 UniProt 蛋白、家族范围或化合物本身，不必为了看详情再启动无关的关系查询或候选预测。",
                    "prompt_en": "What exactly is RHEA:23444? Show the verified record details.",
                    "prompt_zh": "RHEA:23444 具体是什么反应？给我已经核对的记录详情。",
                },
                {
                    "title_en": "Inspect one protein's recorded reactions",
                    "title_zh": "查询具体蛋白的已记录反应",
                    "description_en": "Use the reverse evidence index for a concrete protein without routing it through a family workflow.",
                    "description_zh": "具体蛋白直接查询反向证据索引，不再绕到家族汇总。",
                    "prompt_en": "Which reactions are database-recorded for UniProt P00330?",
                    "prompt_zh": "UniProt P00330 在数据库里已经记录能催化哪些反应？",
                },
                {
                    "title_en": "List concrete members of a scope",
                    "title_zh": "查看家族或功能类的具体成员",
                    "description_en": "Inspect the auditable member subset behind a family or functional-class result.",
                    "description_zh": "查看家族或功能类结果背后真正被纳入当前可审计范围的具体蛋白。",
                    "prompt_en": "Give me five concrete members from the P450 scope you just used.",
                    "prompt_zh": "把刚才 P450 范围里的 5 个具体蛋白成员列给我。",
                },
            ],
        },
        {
            "id": "compound_identity",
            "title_en": "Compound identity and ChEBI resolution",
            "title_zh": "化合物身份与 ChEBI 核对",
            "description_en": "Resolve biochemical names and common naming variants against the local Rhea/ChEBI index. The model may propose search synonyms, but only the index assigns database IDs.",
            "description_zh": "用本地 Rhea/ChEBI 索引核对生化名称及常见命名变体。模型可以提出检索同义词，但数据库编号只由索引确定。",
            "examples": [
                {
                    "title_en": "Resolve a compound",
                    "title_zh": "核对一个化合物",
                    "description_en": "Useful for ChEBI identity questions and for clarifying route endpoints before route search.",
                    "description_zh": "适合查询 ChEBI 身份，也可以在路线搜索前先核对起点或终点化合物。",
                    "prompt_en": "Which ChEBI record corresponds to p-coumaric acid?",
                    "prompt_zh": "对香豆酸对应哪个 ChEBI 记录？",
                },
            ],
        },
        {
            "id": "candidate_retrieval",
            "title_en": "Discover candidate enzymes or reactions",
            "title_zh": "发现候选酶或候选反应",
            "description_en": "Use the general merged candidate universe by default, separate recorded evidence from model-ranked unrecorded associations, and accept natural constraints such as shortlist size, taxonomy, homology distance or an explicitly requested specialized library.",
            "description_zh": "默认使用整合通用候选库，把数据库已知证据和模型筛选的新关联分开；可以直接用自然语言指定候选数量、物种范围、同源距离，或明确要求专用候选库。",
            "examples": [
                {
                    "title_en": "Reaction → candidate enzymes",
                    "title_zh": "从反应寻找候选酶",
                    "description_en": "Recorded associations and model-ranked unrecorded candidates can be requested together or separately.",
                    "description_zh": "可以同时查看已知关联和新关联候选，也可以自然语言只要求其中一种。",
                    "prompt_en": "For RHEA:32883, show the recorded enzymes and then rank 10 unrecorded candidate associations outside the closest homolog family.",
                    "prompt_zh": "针对 RHEA:32883，先展示数据库已知酶，再给 10 个尚未记录、并避开最近同源家族的候选。",
                },
                {
                    "title_en": "Enzyme → possible reactions",
                    "title_zh": "从酶探索可能反应",
                    "description_en": "Use a UniProt protein, a provided sequence, or a verified known activity as the query/reference.",
                    "description_zh": "可使用 UniProt 蛋白、直接提供的序列或已经核对的已知活性作为查询与参考。",
                    "prompt_en": "For UniProt P00338, show recorded reactions and then 10 possible unrecorded reaction associations.",
                    "prompt_zh": "对于 UniProt P00338，先展示数据库已经记录的反应，再给 10 个可能尚未记录的新关联候选。",
                },
                {
                    "title_en": "Paste Reaction SMILES or FASTA",
                    "title_zh": "直接粘贴 Reaction SMILES 或 FASTA",
                    "description_en": "Open-world inputs are handled through the same model-led agent loop rather than a separate front-door mode.",
                    "description_zh": "开放世界结构化输入也经过同一个模型主导的智能体循环，不再走独立的前置模式。",
                    "prompt_en": "I will paste a Reaction SMILES or FASTA sequence here; infer what scientific workflow is appropriate and continue.",
                    "prompt_zh": "我会直接粘贴 Reaction SMILES 或 FASTA 序列，请你自己判断应该采用什么科学工作流并继续。",
                },
            ],
        },
        {
            "id": "route_design",
            "title_en": "Biosynthetic route design",
            "title_zh": "生物合成路线设计",
            "description_en": "Search the Rhea reaction graph and rank routes using step count, enzyme availability, thermodynamics, project-model coverage or E. coli host-flux evidence; predicted transformations can be explored separately when requested.",
            "description_zh": "搜索 Rhea 反应图，可综合步骤数、酶可获得性、热力学、项目模型覆盖和 E. coli 宿主通量进行路线排序；需要时可单独探索预测转化。",
            "examples": [
                {
                    "title_en": "Design and rank routes",
                    "title_zh": "设计并比较候选路线",
                    "description_en": "State the source, target and priorities naturally instead of selecting a route mode.",
                    "description_zh": "直接说明起点、终点和偏好，不需要先选择任何路线模式。",
                    "prompt_en": "Design biosynthetic routes from L-tyrosine to caffeate, prefer recorded reactions and thermodynamically feasible routes, and explain the trade-offs.",
                    "prompt_zh": "设计从 L-酪氨酸到咖啡酸的生物合成路线，优先已知反应和热力学更可行的路线，并说明取舍。",
                },
            ],
        },
        {
            "id": "pathway",
            "title_en": "Whole-pathway enzyme compatibility",
            "title_zh": "整条路径的多酶兼容性",
            "description_en": "Evaluate an already specified multi-step pathway, jointly select missing enzymes, and consider one-pot, staged or in-vivo operation together with pH, temperature and cofactors.",
            "description_zh": "评估已经给出的多步路径，联合选择缺失酶，并结合一锅、分步或细胞内运行方式以及 pH、温度和辅因子条件判断兼容性。",
            "examples": [
                {
                    "title_en": "Evaluate a multi-step pathway",
                    "title_zh": "评估多步路径",
                    "description_en": "Some steps may already have fixed enzymes while others remain open for joint selection.",
                    "description_zh": "可以固定部分步骤的酶，其余步骤交给智能体联合选择。",
                    "prompt_en": "Evaluate L-tyrosine → p-coumarate → caffeate as a two-step pathway and identify a compatible enzyme combination for 30 °C operation.",
                    "prompt_zh": "评估 L-酪氨酸 → 对香豆酸 → 咖啡酸这条两步路径，并寻找适合 30 °C 条件的兼容酶组合。",
                },
            ],
        },
    ],
}


def public_capabilities() -> dict[str, Any]:
    """Return a JSON-safe copy used by both the controller and the frontend."""
    import copy

    return copy.deepcopy(CAPABILITY_MANIFEST)
