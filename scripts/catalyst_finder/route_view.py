from __future__ import annotations

from typing import Any

from scripts.catalyst_finder.route_catalog import build_route_catalog


R2E_MODULES: dict[str, dict[str, str]] = {
    "r2e-query": {
        "title": "确认目标反应",
        "subtitle": "Rhea ID / 定向反应",
        "kind": "input",
        "detail": "先把自然语言映射到真实 Rhea 记录；模型不负责发明反应 ID。",
    },
    "r2e-shot": {
        "title": "已知正例引导",
        "subtitle": "Zero-shot / Few-shot",
        "kind": "decision",
        "detail": "有已核对阳性酶时默认作为 Few-shot 上下文；明确要求 Zero-shot 或回顾性混排时关闭。这里的锚点位于蛋白表示空间。",
    },
    "r2e-scope": {
        "title": "判断反应范围",
        "subtitle": "库内 / 外部反应",
        "kind": "decision",
        "detail": "库内反应直接读取已部署表示；外部反应通过 Rhea Reaction SMILES 现场编码。",
    },
    "r2e-encoder": {
        "title": "反应表示",
        "subtitle": "DRFP + 化学上下文",
        "kind": "encode",
        "detail": "把反应转成生产模型使用的反应表示；外部 Top-10/20 路线还使用 exact-residual 反应表示。",
    },
    "r2e-universe": {
        "title": "构建候选酶空间",
        "subtitle": "当前所选候选库",
        "kind": "universe",
        "detail": "候选数量和来源由本次实际 candidate universe 决定；通用库、TPS 专用库和临时扩展必须分别记录 provenance，不能沿用历史固定规模说明。",
    },
    "r2e-taxonomy": {
        "title": "物种范围筛选",
        "subtitle": "全部 / 真核 / 原核",
        "kind": "filter",
        "detail": "如果指定真核或原核，候选矩阵在模型打分之前先按本地审计的 taxonomy scope 缩小。",
    },
    "r2e-known-mask": {
        "title": "排除数据库已记录酶",
        "subtitle": "known association mask",
        "kind": "novelty",
        "detail": "只改变返回范围：已记录酶不作为新候选返回。Few-shot 上下文与输出过滤彼此独立，因此有已核对阳性时仍可用其引导未记录候选检索。",
    },
    "r2e-known-only": {
        "title": "仅对已记录酶评分",
        "subtitle": "verified-known candidate subset",
        "kind": "filter",
        "detail": "把已核对的数据库关联直接作为候选子集，并用同一 Zero-shot 模型分数在子集内部排序。待评分的已知酶不会同时作为 Few-shot 锚点。",
    },
    "r2e-mixed-ranking": {
        "title": "已知与未知统一 Zero-shot 排名",
        "subtitle": "retrospective mixed ranking",
        "kind": "evaluation",
        "detail": "显式要求时关闭阳性 seed，并让数据库已记录酶与未记录候选进入同一 Zero-shot 排名。已知关联若自然进入前列，可作为模型恢复既有关系的回顾性证据。",
    },
    "r2e-router": {
        "title": "生产路由器",
        "subtitle": "范围 × Few-shot × 预算",
        "kind": "router",
        "detail": "真正的生产路线由 query 是否库内、是否有 seed、Top-K 优化目标以及候选空间修饰共同决定。",
    },
    "r2e-shared": {
        "title": "Shared R2E PU Ensemble",
        "subtitle": "库内反应直接检索",
        "kind": "model",
        "detail": "三个生产模型成员在共享反应—蛋白空间中比较目标反应与所有候选酶。",
    },
    "r2e-loss075": {
        "title": "External R2E Top-3",
        "subtitle": "新反应 · focused model",
        "kind": "model",
        "detail": "外部新反应、短名单目标时使用 reaction-loss-weight 0.75 的专用三模型集成。",
    },
    "r2e-residual": {
        "title": "External R2E Exact Residual",
        "subtitle": "新反应 · Top-10 / Top-20",
        "kind": "model",
        "detail": "外部反应的较深列表使用 exact reaction representation 加 learned residual 的专用生产模型。",
    },
    "r2e-seed": {
        "title": "已知阳性 Few-shot",
        "subtitle": "protein-space positive anchors",
        "kind": "seed",
        "detail": "把数据库阳性和用户核对的额外阳性酶作为蛋白空间锚点，对候选计算到正例集合的最大表示相似度。这里会切换到 Few-shot retrieval，不是给 direct score 简单加权。",
    },
    "r2e-seed-mask": {
        "title": "移除已知阳性",
        "subtitle": "seed 不返回",
        "kind": "filter",
        "detail": "用于引导检索的已知阳性酶本身不会重新出现在新候选列表里。",
    },
    "r2e-cross-cluster": {
        "title": "远缘 / 跨簇筛选",
        "subtitle": "排除 50% identity 同簇蛋白",
        "kind": "novelty",
        "detail": "按项目 protein-cluster-cold 使用的 MMseqs2 家族边界，排除与锚点处于同一 50% sequence-identity cluster 的候选；coverage 阈值为 80%。这是新增的显式 novelty overlay，不伪装成原生产 manifest 的 route suffix。",
    },
    "r2e-cage": {
        "title": "CAGE 结构证据救援",
        "subtitle": "仅特定库内 Top-20",
        "kind": "rescue",
        "detail": "当独立结构证据存在时，可在 broad Top-20 中加入少量结构支持候选。是否出现由真实 selection_source 决定。",
    },
    "r2e-rank": {
        "title": "锁定候选排序",
        "subtitle": "Top-K shortlist",
        "kind": "rank",
        "detail": "在所有已启用的候选范围、seed mask、novelty filter 或 rescue 规则之后形成最终候选顺序。",
    },
    "r2e-trust": {
        "title": "证据与适用性解释",
        "subtitle": "Evidence Passport / reliability",
        "kind": "trust",
        "detail": "对适用的路线展示最近参考、ensemble 稳定性、经验可靠性和 conformal review depth；这些都不是活性概率。",
    },
    "r2e-output": {
        "title": "候选酶",
        "subtitle": "进入实验评估",
        "kind": "output",
        "detail": "返回排序后的候选酶，并保留 UniProt 入口、候选来源和路线 provenance。",
    },
}


E2R_MODULES: dict[str, dict[str, str]] = {
    "e2r-query": {
        "title": "确认目标酶",
        "subtitle": "UniProt / 本地 ID / 蛋白序列",
        "kind": "input",
        "detail": "自然语言先映射到可核对的蛋白记录；库内 ID 使用预计算表示，外部 UniProt 条目读取真实序列后编码。",
    },
    "e2r-shot": {
        "title": "已知正例引导",
        "subtitle": "Zero-shot / Few-shot",
        "kind": "decision",
        "detail": "有已核对反应活性时默认作为 Few-shot 上下文；明确要求 Zero-shot 或回顾性混排时关闭。这里的锚点位于学习到的反应表示空间。",
    },
    "e2r-scope": {
        "title": "判断蛋白范围",
        "subtitle": "库内 / 外部蛋白",
        "kind": "decision",
        "detail": "库内蛋白直接读取 ESM-C 表示；外部 UniProt 条目使用真实氨基酸序列现场编码。",
    },
    "e2r-encoder": {
        "title": "蛋白表示",
        "subtitle": "ESM-C 600M · mean embedding",
        "kind": "encode",
        "detail": "把氨基酸序列编码成生产 E2R 模型使用的蛋白表示。",
    },
    "e2r-universe": {
        "title": "候选反应空间",
        "subtitle": "当前所选候选库",
        "kind": "universe",
        "detail": "候选数量和来源由本次实际 candidate universe 决定；路线视图直接读取运行时 metadata，不再写死历史候选规模。",
    },
    "e2r-router": {
        "title": "生产路由器",
        "subtitle": "范围 × Few-shot × 预算",
        "kind": "router",
        "detail": "Top-3、Top-10、Top-20 的外部蛋白路线并不相同；路由器根据查询范围和预算选择真实部署。",
    },
    "e2r-current": {
        "title": "Dedicated E2R model",
        "subtitle": "库内酶直接反应排序",
        "kind": "model",
        "detail": "对已在参考空间中的酶，使用专用 E2R 神经模型直接排列候选反应。",
    },
    "e2r-neighbor": {
        "title": "Direct + related proteins",
        "subtitle": "外部蛋白 primary view",
        "kind": "model",
        "detail": "外部蛋白先获得直接预测，再结合最多 5 个已注释近邻蛋白的活性迁移证据。",
    },
    "e2r-hardneg": {
        "title": "Hard-negative secondary model",
        "subtitle": "外部 Top-10 第二排序视角",
        "kind": "model",
        "detail": "Top-10 额外调用针对困难替代反应训练的第二神经模型，产生独立排名。",
    },
    "e2r-dualkernel": {
        "title": "Dual-kernel graph support",
        "subtitle": "外部 Top-20 图证据",
        "kind": "model",
        "detail": "Top-20 额外结合蛋白相似、反应相似和已知关联图形成图结构证据。",
    },
    "e2r-rrf10": {
        "title": "Top-10 RRF",
        "subtitle": "35% primary · 65% secondary",
        "kind": "fusion",
        "detail": "使用 reciprocal-rank fusion 合并两条神经排序，不直接混合尺度不可比的原始分数。",
    },
    "e2r-rrf20": {
        "title": "Top-20 RRF",
        "subtitle": "70% neural · 30% dual-kernel",
        "kind": "fusion",
        "detail": "把神经路线与双核图路线按排名位置融合，服务于更深的 promiscuity 探索。",
    },
    "e2r-seed": {
        "title": "已知活性 Few-shot",
        "subtitle": "reaction-space positive anchors",
        "kind": "seed",
        "detail": "把数据库已记录反应和用户核对的额外已知活性作为反应空间锚点，在学习到的反应表示中寻找相似候选活性。它与 R2E Few-shot 对称，但锚点对象是反应而不是蛋白。",
    },
    "e2r-seed-mask": {
        "title": "移除 seed 反应",
        "subtitle": "只返回扩展候选",
        "kind": "filter",
        "detail": "作为 Few-shot seed 的已知反应会从最终结果中移除。",
    },
    "e2r-mask-only": {
        "title": "排除数据库已记录反应",
        "subtitle": "known association mask",
        "kind": "novelty",
        "detail": "只改变返回范围：已记录反应不作为新候选返回。Few-shot 上下文与输出过滤彼此独立，因此有已核对活性时仍可用其引导未记录反应检索。",
    },
    "e2r-known-only": {
        "title": "仅对已记录反应评分",
        "subtitle": "verified-known candidate subset",
        "kind": "filter",
        "detail": "把已核对的数据库反应直接作为候选子集，并用同一 Zero-shot 模型分数在子集内部排序。待评分的已知反应不会同时作为 Few-shot 锚点。",
    },

    "e2r-mixed-ranking": {
        "title": "已知与未知统一 Zero-shot 排名",
        "subtitle": "retrospective mixed ranking",
        "kind": "evaluation",
        "detail": "显式要求时不使用已知反应作为 seed 或 mask，让已记录反应与未记录候选接受同一 Zero-shot 模型评分。已知反应自然排到前列时，可用于回顾模型恢复能力。",
    },
    "e2r-rank": {
        "title": "锁定反应排序",
        "subtitle": "Top-K reaction shortlist",
        "kind": "rank",
        "detail": "在模型路线、Few-shot 或 mask-only 规则之后形成最终反应顺序。",
    },
    "e2r-trust": {
        "title": "证据与适用性解释",
        "subtitle": "Evidence Passport / reliability",
        "kind": "trust",
        "detail": "展示查询熟悉度、ensemble 稳定性和适用的可靠性 / conformal 信息；它们不是催化概率。",
    },
    "e2r-output": {
        "title": "候选反应",
        "subtitle": "Rhea-linked activity hypotheses",
        "kind": "output",
        "detail": "返回按优先级排列的反应，并尽可能提供 Rhea 页面作为人工核对入口。",
    },
}


BASE_ROUTE_LABELS = {
    "r2e-current-top3-v1": "库内反应 · focused Top-3",
    "r2e-current-top10-v1": "库内反应 · balanced Top-10",
    "r2e-current-top20-v1": "库内反应 · broad Top-20",
    "r2e-external-top3-v1": "外部反应 · focused Top-3",
    "r2e-external-top10-v1": "外部反应 · exact-residual Top-10",
    "r2e-external-top20-v1": "外部反应 · exact-residual Top-20",
    "e2r-current-top3-v1": "库内酶 · Top-3 reaction annotation",
    "e2r-current-top10-v1": "库内酶 · Top-10 activity profile",
    "e2r-current-top20-v1": "库内酶 · Top-20 promiscuity map",
    "e2r-external-top3-neighbor-v1": "外部酶 · direct + related proteins Top-3",
    "e2r-external-top10-neural-rrf-v1": "外部酶 · dual neural RRF Top-10",
    "e2r-external-top20-dual-kernel-rrf-v1": "外部酶 · neural + graph RRF Top-20",
}

BASE_ROUTE_LABELS_EN = {
    "r2e-current-top3-v1": "R2E · current · Top 3",
    "r2e-current-top10-v1": "R2E · current · Top 10",
    "r2e-current-top20-v1": "R2E · current · Top 20",
    "r2e-external-top3-v1": "R2E · external · Top 3",
    "r2e-external-top10-v1": "R2E · external · Top 10",
    "r2e-external-top20-v1": "R2E · external · Top 20",
    "e2r-current-top3-v1": "E2R · current · Top 3",
    "e2r-current-top10-v1": "E2R · current · Top 10",
    "e2r-current-top20-v1": "E2R · current · Top 20",
    "e2r-external-top3-neighbor-v1": "E2R · external · Top 3",
    "e2r-external-top10-neural-rrf-v1": "E2R · external · Top 10",
    "e2r-external-top20-dual-kernel-rrf-v1": "E2R · external · Top 20",
}


OVERLAY_LABELS = {
    "r2e-fewshot-seed": "R2E · Few-shot",
    "e2r-fewshot-seed": "E2R · Few-shot",
    "r2e-known-association-mask-overlay": "R2E 批量发现：屏蔽已知关联",
    "e2r-zero-shot-mask-overlay": "E2R · 仅新关联",
    "r2e-temporary-universe-overlay": "R2E 临时候选酶扩展",
    "e2r-temporary-universe-overlay": "E2R 临时候选反应扩展",
    "r2e-manual-override-overlay": "R2E 研究人员手工路线覆盖",
    "e2r-manual-override-overlay": "E2R 研究人员手工路线覆盖",
    "r2e-eukaryote-only-overlay": "R2E · 仅真核",
    "r2e-prokaryote-only-overlay": "R2E · 仅原核",
    "r2e-cage-rescue-overlay": "R2E CAGE 结构证据救援",
}

DOWNSTREAM_WORKFLOWS = [
    {
        "key": "route-design-rhea-known-v1",
        "title": "候选路线设计",
        "title_en": "Route design",
        "availability": "catalyst_finder",
        "description": "从自然语言中的起始前体/宿主和目标产物出发，在官方 Rhea 全量已知生化反应图中枚举候选路线；随后恢复完整 Rhea 化学计量，用 eQuilibrator MDF 评价热力学，E. coli 任务再用 iML1515 route-supported FBA 过滤整路零通量候选。语言模型不生成反应。",
        "flow": [
            {"id": "route-design-parse", "title": "解析路线目标", "subtitle": "natural language → source / target / host", "kind": "input", "detail": "只规范化用户明确给出的前体、目标、宿主和排序偏好，不产生中间反应或数据库 ID。"},
            {"id": "route-design-rhea", "title": "构建全量 Rhea 已知反应图", "subtitle": "directed reaction SMILES · ChEBI · Swiss-Prot", "kind": "universe", "detail": "缓存 Rhea 官方定向 reaction SMILES、ChEBI 结构、方向和 Swiss-Prot 映射，形成远大于本地项目关系表的可审计已知生化网络。"},
            {"id": "route-design-transform", "title": "筛选主底物到主产物", "subtitle": "currency exclusion · structure continuity", "kind": "filter", "detail": "排除高频辅因子捷径，用结构连续性保留主转化连接，同时保留完整 Rhea ID 供复核。"},
            {"id": "route-design-kpaths", "title": "先扩展较大的候选路线池", "subtitle": "NetworkX shortest_simple_paths", "kind": "model", "detail": "复用 NetworkX K-shortest simple paths；最终 Top-K 之前保留更多预候选，避免旧图分提前截断后续真实可行性更好的路线。"},
            {"id": "route-design-stoichiometry", "title": "恢复完整 Rhea 化学计量", "subtitle": "exact directed reaction participants", "kind": "trust", "detail": "热力学和 FBA 不使用主链投影，而是重新从官方定向 reaction SMILES 恢复全部底物、产物、辅因子并精确映射到 ChEBI。"},
            {"id": "route-design-thermo", "title": "计算 Max-min Driving Force", "subtitle": "eQuilibrator · equilibrator-pathway", "kind": "trust", "detail": "对证据完整且平衡的 Rhea 路线计算逐步 ΔG′ 和整路 MDF。未覆盖或失败保持未知；MDF 不是酶活性或成功率。"},
            {"id": "route-design-fba", "title": "E. coli 宿主通量门控", "subtitle": "COBRApy · iML1515", "kind": "filter", "detail": "仅 E. coli 任务启用：要求候选每一步与目标输出同时承载共同路线通量，并保持至少 10%/50% 野生型生长；已完成 FBA 且整路通量为 0 的候选被过滤。"},
            {"id": "route-design-rank", "title": "合并可解释证据排序", "subtitle": "base route · MDF · host flux", "kind": "rank", "detail": "基础路线分、MDF 和适用时的宿主 route-supported flux 进入最终相对排序。FBA 是化学计量容量，不是滴度或动力学预测。"},
            {"id": "route-design-handoff", "title": "把选中路线交给多酶评估", "subtitle": "route → pathway-compatibility-v1", "kind": "output", "detail": "选中候选路线后继续复用现有逐步 R2E、UniProt 条件证据与多酶全局兼容性重排。"},
        ],
    },
    {
        "key": "pathway-compatibility-v1",
        "title": "整路多酶兼容性",
        "title_en": "Pathway enzyme compatibility",
        "availability": "catalyst_finder",
        "description": "把自然语言中的多步反应拆成已核对步骤，复用每一步的生产 R2E 候选排序，并用 UniProt 条件证据对整组酶做全局兼容性重排。缺失条件不会被当作兼容证据。",
        "flow": [
            {"id": "pathway-parse", "title": "解析并核对整条路径", "subtitle": "natural language → Rhea steps", "kind": "input", "detail": "把多步反应顺序拆开，并逐步映射到真实 Rhea 记录；用户仍可以在确认卡里改选。"},
            {"id": "pathway-r2e", "title": "逐步生成候选酶", "subtitle": "reuse production R2E", "kind": "model", "detail": "每一步继续使用 Catalyst Finder 已部署的反应→酶排序，而不是另造一套未经验证的候选模型。"},
            {"id": "pathway-uniprot-conditions", "title": "汇集实验条件证据", "subtitle": "UniProtKB annotations", "kind": "trust", "detail": "读取可获得的 pH、温度、辅因子、活性调控和亚细胞定位注释；没有记录时明确标为未知。"},
            {"id": "pathway-global-rerank", "title": "联合选择整组酶", "subtitle": "global combination rerank", "kind": "fusion", "detail": "以各步模型排名为主信号，再考虑已知条件兼容性，在候选组合中寻找更适合整条路径的一组酶。"},
            {"id": "pathway-conflict-audit", "title": "审计条件冲突", "subtitle": "pH · temperature · cofactor · localization", "kind": "filter", "detail": "显式列出共享条件不足、辅因子/调控风险和体内定位差异；不会把未知数据解释为没有冲突。"},
            {"id": "pathway-output", "title": "给出实验策略", "subtitle": "one-pot / staged / compartmentalized", "kind": "output", "detail": "输出逐步酶选择、冲突证据与共同条件窗口，并在需要时建议分步、换酶或区室化。"},
        ],
    },
]


def _catalog_flow(module_ids: list[str]) -> list[dict[str, str]]:
    flow: list[dict[str, str]] = []
    for module_id in module_ids:
        source = R2E_MODULES.get(module_id) or E2R_MODULES.get(module_id)
        if source is None:
            source = {
                "title": module_id.replace("-", " "),
                "subtitle": "repository module",
                "kind": "control",
                "detail": "该模块来自仓库中的生产路线定义。",
            }
        flow.append({"id": module_id, **dict(source)})
    return flow



def system_route_catalog() -> dict[str, Any]:
    catalog = build_route_catalog()
    bases = []
    for entry in catalog["routes"]:
        bases.append({
            "key": entry["route_id"],
            "label": BASE_ROUTE_LABELS.get(entry["route_id"], entry["route_id"]),
            "label_en": BASE_ROUTE_LABELS_EN.get(entry["route_id"], entry["route_id"]),
            "direction": entry["direction"],
            "scope": entry["scope"],
            "objective": entry["objective"],
            "retrieval": entry["retrieval"],
            "modules": entry["modules"],
            "flow": _catalog_flow(list(entry["modules"])),
            "description": entry["description"],
            "use_case": entry["use_case"],
            "availability": entry["availability"],
        })
    overlays = []
    for entry in catalog["overlays"]:
        if str(entry.get("availability") or "") != "portal":
            continue
        overlays.append({
            "key": entry["key"],
            "label": OVERLAY_LABELS.get(entry["key"], entry["key"]),
            "label_en": {
                "r2e-fewshot-seed": "R2E · Few-shot",
                "e2r-fewshot-seed": "E2R · Few-shot",
                "e2r-zero-shot-mask-overlay": "E2R · unrecorded only",
                "r2e-eukaryote-only-overlay": "R2E · eukaryotes only",
                "r2e-prokaryote-only-overlay": "R2E · prokaryotes only",
            }.get(entry["key"], entry["key"]),
            "direction": entry["direction"],
            "scope": entry["scope"],
            "objective": entry["objective"],
            "retrieval": entry["retrieval"],
            "modules": entry["modules"],
            "flow": _catalog_flow(list(entry["modules"])),
            "description": entry["description"],
            "use_case": entry["use_case"],
            "availability": entry["availability"],
        })
    overlays.append({
        "key": "r2e-discovery-known-mask-v1",
        "label": "R2E · 仅新关联",
        "label_en": "R2E · unrecorded only",
        "direction": "reaction_to_enzyme",
        "scope": "any",
        "objective": "top3|top5|top10|top20",
        "retrieval": "post_score_known_association_filter",
        "modules": ["r2e-known-mask"],
        "flow": _catalog_flow(["r2e-known-mask"]),
        "description": "当用户明确要求“排除已知”时，把当前知识库中已经记录为可催化该反应的酶从返回列表中移除，聚焦尚未收录的潜在新反应–酶关联。普通排序默认保留这些已记录酶。",
        "use_case": "Optional discovery policy for reaction-to-enzyme search. It changes the returned candidate set without turning known enzymes into positive seeds.",
        "availability": "catalyst_finder",
    })
    overlays.append({
        "key": "r2e-known-only-filter-v1",
        "label": "R2E · 仅已知",
        "label_en": "R2E · recorded only",
        "direction": "reaction_to_enzyme",
        "scope": "any",
        "objective": "top3|top5|top10|top20",
        "retrieval": "full_score_then_recorded_association_filter",
        "modules": ["r2e-known-only"],
        "flow": _catalog_flow(["r2e-known-only"]),
        "description": "当用户只想查看当前知识库已有反应–酶关联时，先完成候选空间排序，再仅保留已记录催化酶并在其中取 Top-K。",
        "use_case": "Reference and positive-control view without changing the underlying model score ordering.",
        "availability": "catalyst_finder",
    })
    overlays.append({
        "key": "e2r-known-only-filter-v1",
        "label": "E2R · 仅已知",
        "label_en": "E2R · recorded only",
        "direction": "enzyme_to_reaction",
        "scope": "any",
        "objective": "top3|top5|top10|top20",
        "retrieval": "full_score_then_recorded_association_filter",
        "modules": ["e2r-known-only"],
        "flow": _catalog_flow(["e2r-known-only"]),
        "description": "当用户只想查看当前知识库已经记录的酶–反应关联时，先完成反应空间排序，再仅保留已记录反应并在其中取 Top-K。",
        "use_case": "Reference-activity view for checking known catalog associations in model score order.",
        "availability": "catalyst_finder",
    })
    overlays.append({
        "key": "r2e-cross-cluster-filter-v1",
        "label": "R2E · 远缘候选",
        "label_en": "R2E · remote candidates",
        "direction": "reaction_to_enzyme",
        "scope": "any",
        "objective": "top3|top5|top10|top20",
        "retrieval": "post_score_candidate_novelty_filter",
        "modules": ["r2e-cross-cluster"],
        "flow": _catalog_flow(["r2e-cross-cluster"]),
        "description": "Uses the repository's protein-cluster-cold family boundary: MMseqs2 50% identity with 80% coverage. Candidate enzymes in the same cluster as selected positive anchors are excluded before the final shortlist is returned.",
        "use_case": "Use only when the scientific goal is remote-family discovery rather than the default high-success homolog expansion objective.",
        "availability": "catalyst_finder",
    })
    overlays.extend([
        {
            "key": "r2e-mixed-zero-shot",
            "label": "R2E · Zero-shot 混排",
            "label_en": "R2E · zero-shot mixed ranking",
            "direction": "reaction_to_enzyme",
            "scope": "any",
            "objective": "top3|top5|top10|top20",
            "retrieval": "zero_shot_rank_with_recorded",
            "modules": ["r2e-mixed-ranking"],
            "flow": _catalog_flow(["r2e-mixed-ranking"]),
            "description": "显式回顾性模式：关闭阳性 seed 与已知关联 mask，让已记录和未记录酶接受同一套 Zero-shot 排名。",
            "use_case": "Use only when the user explicitly wants one retrospective zero-shot ranking containing both recorded and unrecorded enzymes.",
            "availability": "explicit",
        },
        {
            "key": "e2r-mixed-zero-shot",
            "label": "E2R · Zero-shot 混排",
            "label_en": "E2R · zero-shot mixed ranking",
            "direction": "enzyme_to_reaction",
            "scope": "any",
            "objective": "top3|top5|top10|top20",
            "retrieval": "zero_shot_rank_with_recorded",
            "modules": ["e2r-mixed-ranking"],
            "flow": _catalog_flow(["e2r-mixed-ranking"]),
            "description": "显式回顾性模式：不使用已知反应作为 seed 或 mask，让已记录和未记录反应进入同一 Zero-shot 排名。",
            "use_case": "Use only when the user explicitly wants one retrospective zero-shot ranking containing both recorded and unrecorded reactions.",
            "availability": "explicit",
        },
        {
            "key": "r2e-tps-specialized",
            "label": "R2E · TPS 专用库",
            "label_en": "R2E · TPS specialist",
            "direction": "reaction_to_enzyme",
            "scope": "any",
            "objective": "top3|top5|top10|top20",
            "retrieval": "tps_specialized_universe",
            "modules": ["r2e-universe"],
            "flow": _catalog_flow(["r2e-universe"]),
            "description": "显式把候选空间限制到 TPS 专用库；相关资产针对 TPS 领域训练和评测，不因普通萜类语境自动启用。",
            "use_case": "Explicit TPS-specialist retrieval only; scores stay within this specialist scope and are not compared with general-universe scores.",
            "availability": "explicit",
        },
        {
            "key": "e2r-tps-specialized",
            "label": "E2R · TPS 专用库",
            "label_en": "E2R · TPS specialist",
            "direction": "enzyme_to_reaction",
            "scope": "any",
            "objective": "top3|top5|top10|top20",
            "retrieval": "tps_specialized_universe",
            "modules": ["e2r-universe"],
            "flow": _catalog_flow(["e2r-universe"]),
            "description": "显式把反应候选空间限制到 TPS 专用库；使用 TPS 领域特化训练/评测资产，不作为通用默认。",
            "use_case": "Explicit TPS-specialist retrieval only; scores stay within this specialist scope and are not compared with general-universe scores.",
            "availability": "explicit",
        },
    ])
    overlays.append({
        "key": "route-design-pickaxe-isolated",
        "label": "路线 · 预测转化扩展",
        "label_en": "Routes · predicted transformations",
        "direction": "route_design",
        "scope": "external",
        "objective": "novel_route_exploration",
        "retrieval": "isolated_rule_based_prediction",
        "modules": [],
        "flow": [
            {"id": "pickaxe-isolated", "title": "隔离运行 MINE/Pickaxe", "subtitle": "pinned external worker", "kind": "universe", "detail": "使用固定 upstream commit 和独立运行时依赖；不修改 vendored 源码，也不把旧依赖写入 Catalyst Finder 主环境。"},
            {"id": "pickaxe-metadata", "title": "应用 MetaCyc generalized rules", "subtitle": "predicted transformations", "kind": "model", "detail": "预测步骤必须保留规则来源并标记为预测，不与 Rhea 已收录反应共用证据标签。"},
            {"id": "pickaxe-verify", "title": "与已知数据库分层核对", "subtitle": "predicted ≠ Rhea-known", "kind": "trust", "detail": "只有明确要求探索时才启用；输出层必须区分已知与预测步骤，再进入热力学和酶可获得性复核。"},
        ],
        "description": "可选的规则扩展层，用于本地/Rhea 已知网络找不到足够路线时探索潜在生化转化。它与主服务、原 terpene portal 和 Rhea 已知路线严格隔离。",
        "use_case": "Explicit predicted-route exploration only; never silently mixed into known Rhea routes.",
        "availability": "isolated_optional",
    })
    return {
        "base_routes": bases,
        "overlays": overlays,
        "downstream_workflows": DOWNSTREAM_WORKFLOWS,
        "counts": {
            "manifest_routes": len(bases),
            "public_overlays": len(overlays),
            "hidden_internal_overlays": len(catalog["overlays"]) - sum(1 for row in catalog["overlays"] if str(row.get("availability") or "") == "portal"),
            "specialist_overlays": sum(1 for row in overlays if str(row.get("availability") or "") in {"explicit", "isolated_optional"}),
        },
        "coverage": catalog["coverage"],
    }


def _module(module_id: str, *, metric: str = "", note: str = "", detail: str = "", state: str = "active") -> dict[str, str]:
    base = dict(R2E_MODULES[module_id])
    base.update({"id": module_id, "metric": metric, "note": note, "state": state})
    if detail:
        base["detail"] = detail
    return base


def build_r2e_route_view(
    *,
    reaction: dict[str, Any],
    query: dict[str, Any],
    routing: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    route_id = str(query.get("route_id") or routing.get("actual_route_id") or routing.get("planned_route_id") or "")
    base_route = route_id.split("+", 1)[0]
    shot_mode = str(query.get("shot_mode") or routing.get("shot_mode") or "zero_shot")
    scope = str(query.get("scope") or routing.get("scope") or "external")
    objective = str(query.get("ranking_objective") or routing.get("ranking_objective") or "top10")
    taxonomy = str(query.get("enzyme_taxonomy_scope") or routing.get("enzyme_taxonomy_scope") or "all")
    selected_top_k = int(routing.get("top_k") or len(candidates) or 10)
    seed_ids = list(routing.get("known_enzyme_ids") or [])
    seed_source = str(routing.get("seed_source") or ("user" if seed_ids else "none"))
    novelty = routing.get("homology_filter") or {}
    novelty_applied = bool(novelty.get("applied"))
    discovery = routing.get("discovery_filter") or {}
    result_mode = str(
        discovery.get("result_mode")
        or ("novel_association_discovery" if discovery.get("applied") else "full_ranking")
    )
    discovery_applied = result_mode == "novel_association_discovery"
    known_only_applied = result_mode in {"known_associations_only", "known_associations_model_ranked"}
    mixed_applied = result_mode == "mixed_zero_shot_ranking"
    cage_count = sum(1 for row in candidates if row.get("selection_source") == "cage_rescue")

    pre_tax = query.get("candidate_universe_pre_taxonomy_size")
    post_tax = query.get("candidate_universe_post_taxonomy_size") or query.get("candidate_universe_size")
    nodes = [
        _module("r2e-query", metric=str(reaction.get("rhea_id") or "Rhea verified"), note=str(reaction.get("equation") or "")),
        _module(
            "r2e-shot",
            metric="Few-shot" if shot_mode == "few_shot" else "Zero-shot",
            note=(f"{len(seed_ids)} 个蛋白正例锚点 · {seed_source}" if seed_ids else "本轮使用 Zero-shot，不以已知阳性引导评分"),
        ),
        _module(
            "r2e-scope",
            metric="库内反应" if scope == "current" else "外部反应",
            note="读取部署中的反应表示" if scope == "current" else "Rhea 定向 Reaction SMILES 现场编码",
        ),
        _module(
            "r2e-encoder",
            metric="precomputed" if scope == "current" else "external encoding",
            note="生产模型反应表示",
        ),
        _module(
            "r2e-universe",
            metric=f"{pre_tax or query.get('candidate_universe_size') or '—'} proteins",
            note=str(query.get("candidate_universe_description") or routing.get("candidate_universe") or "当前生产候选空间"),
            detail=str(query.get("candidate_universe_description") or routing.get("candidate_universe") or R2E_MODULES["r2e-universe"]["detail"]),
        ),
        _module(
            "r2e-taxonomy",
            metric={"all": "全部候选", "eukaryote": "仅真核", "prokaryote": "仅原核"}.get(taxonomy, taxonomy),
            note=(f"{pre_tax} → {post_tax}" if pre_tax and post_tax and pre_tax != post_tax else "不改变候选空间"),
        ),
        _module(
            "r2e-router",
            metric=f"{scope} · {shot_mode} · {objective}",
            note=f"route family: {base_route}",
        ),
    ]

    if shot_mode == "few_shot" or "+fewshot" in route_id:
        nodes.append(_module("r2e-seed", metric=f"{len(seed_ids)} positive anchor(s)", note="候选到蛋白空间正例锚点的最大 ESM-C 表示相似度"))
        nodes.append(_module("r2e-seed-mask", metric=f"mask {len(seed_ids)} seed(s)", note="seed 只提供检索证据，不重新作为候选返回"))
    elif base_route.startswith("r2e-current-"):
        nodes.append(_module("r2e-shared", metric="3-member ensemble", note=str(query.get("score_source") or "direct")))
    elif base_route == "r2e-external-top3-v1":
        nodes.append(_module("r2e-loss075", metric="reaction loss 0.75", note=str(query.get("score_source") or "direct")))
    else:
        nodes.append(_module("r2e-residual", metric="exact residual", note=str(query.get("score_source") or "direct")))

    if novelty_applied:
        nodes.append(_module(
            "r2e-cross-cluster",
            metric=f"排除 {novelty.get('excluded_count', 0)} 个同簇候选",
            note=f"锚点 {novelty.get('anchor_count', 0)} · {novelty.get('definition', '50% identity cluster')}",
        ))

    if mixed_applied:
        known_in_top = sum(1 for row in candidates if row.get("known_association"))
        nodes.append(_module(
            "r2e-mixed-ranking",
            metric=f"Top {selected_top_k} 中 {known_in_top} 个已记录关联",
            note="同一 Zero-shot 分数下混排；不使用已知阳性作为 seed",
        ))
    elif discovery_applied:
        nodes.append(_module(
            "r2e-known-mask",
            metric=f"屏蔽 {discovery.get('recorded_association_count', 0)} 条已记录关联",
            note=f"本次评分序列实际移除 {discovery.get('excluded_count', 0)} 个已知酶候选",
        ))
    elif known_only_applied:
        nodes.append(_module(
            "r2e-known-only",
            metric=f"保留 {discovery.get('retained_count', len(candidates))} 个已记录候选",
            note=f"对已核对关联子集做 Zero-shot 评分 · 子集大小 {discovery.get('candidate_universe_recorded_association_count', discovery.get('recorded_association_count', 0))}",
        ))

    if cage_count:
        nodes.append(_module("r2e-cage", metric=f"{cage_count} rescue", note="由实际 selection_source=cage_rescue 触发"))

    nodes.extend([
        _module("r2e-rank", metric=f"Top {selected_top_k}", note=f"最终返回 {len(candidates)} 个候选"),
        _module(
            "r2e-trust",
            metric=str(query.get("empirical_reliability_tier") or query.get("empirical_reliability_status") or "scope-aware"),
            note=str((query.get("evidence_passport") or {}).get("applicability_tier") or "Evidence Passport"),
        ),
        _module("r2e-output", metric=f"{len(candidates)} candidates", note="rank + provenance + UniProt"),
    ])

    edges = [{"from": nodes[index]["id"], "to": nodes[index + 1]["id"]} for index in range(len(nodes) - 1)]
    active_overlays = []
    if shot_mode == "few_shot":
        active_overlays.append("r2e-fewshot-seed")
    if taxonomy == "eukaryote":
        active_overlays.append("r2e-eukaryote-only-overlay")
    elif taxonomy == "prokaryote":
        active_overlays.append("r2e-prokaryote-only-overlay")
    if novelty_applied:
        active_overlays.append("r2e-cross-cluster-filter-v1")
    if mixed_applied:
        active_overlays.append("r2e-mixed-zero-shot")
    elif discovery_applied:
        active_overlays.append("r2e-discovery-known-mask-v1")
    elif known_only_applied:
        active_overlays.append("r2e-known-only-filter-v1")
    if str(query.get("candidate_universe") or routing.get("candidate_universe") or "") == "tps_specialized":
        active_overlays.append("r2e-tps-specialized")
    if cage_count:
        active_overlays.append("r2e-cage-rescue-overlay")

    return {
        "direction": "reaction_to_enzyme",
        "route_id": route_id,
        "base_route_id": base_route,
        "active_overlays": active_overlays,
        "title": (
            BASE_ROUTE_LABELS.get(base_route, base_route)
            + (" · Zero-shot 混排" if mixed_applied else " · 新关联发现" if discovery_applied else " · 仅已记录" if known_only_applied else "")
        ),
        "decision": {
            "scope": scope,
            "shot_mode": shot_mode,
            "objective": objective,
            "top_k": selected_top_k,
            "taxonomy": taxonomy,
            "homology_policy": routing.get("homology_policy", "allow"),
            "known_association_policy": (
                "rank_with_known" if mixed_applied else "exclude_known" if discovery_applied else "known_only" if known_only_applied else "separate_known"
            ),
        },
        "nodes": nodes,
        "edges": edges,
        "summary": routing.get("reason") or "按生产路由规则执行。",
    }


def _e2r_module(module_id: str, *, metric: str = "", note: str = "", detail: str = "", state: str = "active") -> dict[str, str]:
    base = dict(E2R_MODULES[module_id])
    base.update({"id": module_id, "metric": metric, "note": note, "state": state})
    if detail:
        base["detail"] = detail
    return base


def build_e2r_route_view(
    *,
    protein: dict[str, Any],
    query: dict[str, Any],
    routing: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    route_id = str(query.get("route_id") or routing.get("actual_route_id") or routing.get("planned_route_id") or "")
    base_route = route_id.split("+", 1)[0]
    scope = str(query.get("scope") or routing.get("scope") or "external")
    shot_mode = str(query.get("shot_mode") or routing.get("shot_mode") or "zero_shot")
    objective = str(query.get("ranking_objective") or routing.get("ranking_objective") or "top10")
    top_k = int(routing.get("top_k") or len(candidates) or 10)
    seed_ids = list(routing.get("known_reaction_ids") or [])
    mask_ids = list(routing.get("mask_reaction_ids") or [])
    discovery = routing.get("discovery_filter") or {}
    result_mode = str(
        discovery.get("result_mode")
        or ("novel_association_discovery" if discovery.get("applied") else "full_ranking")
    )
    known_only_applied = result_mode in {"known_associations_only", "known_associations_model_ranked"}
    mixed_applied = result_mode == "mixed_zero_shot_ranking"
    nodes = [
        _e2r_module("e2r-query", metric=str(protein.get("id") or protein.get("accession") or "verified protein"), note=f"{protein.get('name') or ''} · {protein.get('organism') or ''}".strip(" ·")),
        _e2r_module(
            "e2r-shot",
            metric="Few-shot" if seed_ids else "Zero-shot",
            note=(f"{len(seed_ids)} 个反应正例锚点" if seed_ids else "本轮使用 Zero-shot；输出过滤另行处理"),
        ),
        _e2r_module("e2r-scope", metric="库内蛋白" if scope == "current" else "外部蛋白", note="读取预计算蛋白表示" if scope == "current" else "从 UniProt 序列现场编码"),
        _e2r_module("e2r-encoder", metric="precomputed" if scope == "current" else "ESM-C external encoding", note=str((query.get("input_audit") or {}).get("protein_input_status") or "protein representation")),
        _e2r_module(
            "e2r-universe",
            metric=f"{query.get('candidate_universe_size') or '—'} reactions",
            note=str(query.get("candidate_universe_description") or routing.get("candidate_universe") or "当前生产候选空间"),
            detail=str(query.get("candidate_universe_description") or routing.get("candidate_universe") or E2R_MODULES["e2r-universe"]["detail"]),
        ),
        _e2r_module("e2r-router", metric=f"{scope} · {shot_mode} · {objective}", note=f"route family: {base_route}"),
    ]
    if seed_ids or "+fewshot" in route_id:
        nodes.extend([
            _e2r_module("e2r-seed", metric=f"{len(seed_ids)} positive anchor(s)", note="learned reaction-space positive-anchor similarity"),
            _e2r_module("e2r-seed-mask", metric=f"mask {len(seed_ids)} seed(s)", note="seed 本身不作为新发现返回"),
        ])
    elif base_route.startswith("e2r-current-"):
        nodes.append(_e2r_module("e2r-current", metric="3-member ensemble", note=str(query.get("score_source") or "direct")))
    elif base_route == "e2r-external-top3-neighbor-v1":
        nodes.append(_e2r_module("e2r-neighbor", metric="direct + 5 neighbours", note=str(query.get("score_source") or "neighbor_hybrid")))
    elif base_route == "e2r-external-top10-neural-rrf-v1":
        nodes.extend([
            _e2r_module("e2r-neighbor", metric="primary ranking", note="direct + related proteins"),
            _e2r_module("e2r-hardneg", metric="secondary ranking", note="hard-negative model + neighbours"),
            _e2r_module("e2r-rrf10", metric="35 / 65 · c=60", note=str(query.get("score_source") or "RRF")),
        ])
    elif base_route == "e2r-external-top20-dual-kernel-rrf-v1":
        nodes.extend([
            _e2r_module("e2r-neighbor", metric="primary neural ranking", note="direct + related proteins"),
            _e2r_module("e2r-dualkernel", metric="graph support", note="protein × reaction × association"),
            _e2r_module("e2r-rrf20", metric="70 / 30 · c=60", note=str(query.get("score_source") or "RRF")),
        ])
    if mixed_applied:
        known_in_top = sum(1 for row in candidates if row.get("known_association"))
        nodes.append(_e2r_module(
            "e2r-mixed-ranking", metric=f"Top {top_k} 中 {known_in_top} 个已记录关联",
            note="同一 Zero-shot 分数下混排；不使用已知反应作为 seed 或 mask",
        ))
    elif mask_ids:
        nodes.append(_e2r_module("e2r-mask-only", metric=f"屏蔽 {len(mask_ids)} 个已记录反应", note="结果仅保留当前知识库中尚未与该酶记录关联的候选"))
    elif known_only_applied:
        nodes.append(_e2r_module(
            "e2r-known-only",
            metric=f"保留 {discovery.get('retained_count', len(candidates))} 个已记录反应",
            note=f"对已核对反应子集做 Zero-shot 评分 · 子集大小 {discovery.get('candidate_universe_recorded_association_count', discovery.get('recorded_association_count', 0))}",
        ))
    nodes.extend([
        _e2r_module("e2r-rank", metric=f"Top {top_k}", note=f"最终返回 {len(candidates)} 个反应"),
        _e2r_module("e2r-trust", metric=str(query.get("empirical_reliability_tier") or query.get("empirical_reliability_status") or "scope-aware"), note=str((query.get("evidence_passport") or {}).get("applicability_tier") or "Evidence Passport")),
        _e2r_module("e2r-output", metric=f"{len(candidates)} reactions", note="rank + provenance + Rhea"),
    ])
    overlays: list[str] = []
    if seed_ids:
        overlays.append("e2r-fewshot-seed")
    if mixed_applied:
        overlays.append("e2r-mixed-zero-shot")
    elif mask_ids:
        overlays.append("e2r-zero-shot-mask-overlay")
    elif known_only_applied:
        overlays.append("e2r-known-only-filter-v1")
    if str(query.get("candidate_universe") or routing.get("candidate_universe") or "") == "tps_specialized":
        overlays.append("e2r-tps-specialized")
    return {
        "direction": "enzyme_to_reaction",
        "route_id": route_id,
        "base_route_id": base_route,
        "active_overlays": overlays,
        "title": BASE_ROUTE_LABELS.get(base_route, base_route) + (" · Zero-shot 混排" if mixed_applied else " · 新关联发现" if mask_ids else " · 仅已记录" if known_only_applied else ""),
        "decision": {
            "scope": scope,
            "shot_mode": shot_mode,
            "objective": objective,
            "top_k": top_k,
            "use_known_activity_seeds": bool(seed_ids),
            "known_association_policy": "rank_with_known" if mixed_applied else "exclude_known" if mask_ids else "known_only" if known_only_applied else "separate_known",
        },
        "nodes": nodes,
        "edges": [{"from": nodes[i]["id"], "to": nodes[i+1]["id"]} for i in range(len(nodes)-1)],
        "summary": routing.get("reason") or "按生产 E2R 路由规则执行。",
    }
