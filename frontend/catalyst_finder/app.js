(() => {
  const $ = (id) => document.getElementById(id);
  const messages = $("messages");
  const form = $("composerForm");
  const input = $("composerInput");
  const sendButton = $("sendButton");
  const serviceStatus = $("serviceStatus");
  const composerContext = $("composerContext");
  const contextTitle = $("contextTitle");
  const contextSummary = $("contextSummary");
  const contextFacts = $("contextFacts");
  const processList = $("processList");
  const technicalDetails = $("technicalDetails");
  const workspace = document.querySelector(".workspace");
  const runRail = $("runRail");
  const railToggle = $("railToggle");
  const techLanguageModel = $("techLanguageModel");
  const routeTitle = $("routeTitle");
  const routeEmpty = $("routeEmpty");
  const routeScroll = $("routeScroll");
  const routeTimeline = $("routeTimeline");
  const routeId = $("routeId");
  const routeStepCount = $("routeStepCount");
  const routeCatalog = $("routeCatalog");
  const routeCatalogCount = $("routeCatalogCount");
  const routeCatalogStats = $("routeCatalogStats");
  const routeTitleButton = $("routeTitleButton");
  const routeDialog = $("routeDialog");
  const routeDialogClose = $("routeDialogClose");
  const routeDialogType = $("routeDialogType");
  const routeDialogTitle = $("routeDialogTitle");
  const routeDialogKey = $("routeDialogKey");
  const routeDialogDescription = $("routeDialogDescription");
  const routeDialogMeta = $("routeDialogMeta");
  const routeDialogFlow = $("routeDialogFlow");
  const feedbackButton = $("feedbackButton");
  const feedbackDialog = $("feedbackDialog");
  const feedbackForm = $("feedbackForm");
  const feedbackClose = $("feedbackClose");
  const feedbackCancel = $("feedbackCancel");
  const feedbackCategory = $("feedbackCategory");
  const feedbackMessage = $("feedbackMessage");
  const feedbackContact = $("feedbackContact");
  const feedbackStatus = $("feedbackStatus");
  const feedbackSubmit = $("feedbackSubmit");
  const languageToggle = $("languageToggle");
  const i18n = window.CatalystI18n;
  const uiLanguage = i18n?.current?.() || "en";
  const tr = (en, zh) => i18n?.tr?.(en, zh) ?? (uiLanguage === "zh" ? zh : en);
  const containsCjk = (value) => /[\u3400-\u9fff]/.test(String(value || ""));
  function localizedBackendText(value, enFallback, zhFallback = enFallback) {
    const text = String(value || "").trim();
    if (uiLanguage === "zh") return text || zhFallback;
    return text && !containsCjk(text) ? text : enFallback;
  }

  let directionHint = "auto";
  let directionHintOneShot = false;
  let routeMode = "intelligent";
  let busy = false;
  let continuation = null;
  let useContinuation = true;
  let activeVerification = null;
  let serviceSnapshot = null;
  let currentRouteView = null;
  let activeRun = null;
  const routeCatalogIndex = new Map();
  const initialWelcome = messages.firstElementChild.cloneNode(true);

  function newId(prefix) {
    const random = globalThis.crypto?.randomUUID?.() || `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
    return `${prefix}_${random}`;
  }

  function sessionId() {
    try {
      const existing = sessionStorage.getItem(`catalyst_finder_session_id_${uiLanguage}`);
      if (existing) return existing;
      const created = newId("sess");
      sessionStorage.setItem(`catalyst_finder_session_id_${uiLanguage}`, created);
      return created;
    } catch (_) {
      return newId("sess");
    }
  }

  function recordClientEvent(eventType, run, input = null, metadata = {}) {
    if (!run?.run_id) return;
    api("/api/run-events", {
      event_type: eventType,
      session_id: run.session_id,
      run_id: run.run_id,
      step_id: `step_${Date.now().toString(36)}`,
      input,
      metadata,
    }).catch(() => { /* telemetry must not interrupt the model workflow */ });
  }

  function setRailCollapsed(collapsed) {
    const isCollapsed = Boolean(collapsed);
    workspace?.classList.toggle("rail-collapsed", isCollapsed);
    if (runRail) runRail.setAttribute("aria-hidden", isCollapsed ? "true" : "false");
    if (railToggle) {
      const actionLabel = isCollapsed ? tr("Expand task panel", "展开任务侧栏") : tr("Collapse task panel", "收起任务侧栏");
      railToggle.setAttribute("aria-expanded", isCollapsed ? "false" : "true");
      railToggle.setAttribute("aria-label", isCollapsed ? tr("Expand current task panel", "展开本次任务侧栏") : tr("Collapse current task panel", "收起本次任务侧栏"));
      railToggle.dataset.tooltip = actionLabel;
      railToggle.dataset.mobileLabel = actionLabel;
    }
  }

  const directionLabels = {
    reaction_to_enzyme: tr("Find enzymes", "寻找候选酶"),
    enzyme_to_reaction: tr("Find reactions", "预测可能反应"),
    route_design: tr("Design routes", "推荐并排序路线"),
    pathway_compatibility: tr("Evaluate pathway", "评估整条路径"),
  };
  const processOrder = ["understand", "verify", "search", "result"];
  const routeKindLabels = {
    input: "INPUT",
    decision: "GATE",
    encode: "ENCODE",
    universe: "SPACE",
    filter: "FILTER",
    router: "ROUTER",
    model: "MODEL",
    seed: "SEED",
    fusion: "FUSION",
    novelty: "NOVELTY",
    rescue: "RESCUE",
    rank: "RANK",
    trust: "EVIDENCE",
    output: "OUTPUT",
    control: "STEP",
  };
  const routeKindNames = {
    input: tr("Input & verification", "输入与核对"),
    decision: tr("Decision gate", "条件判断"),
    encode: tr("Representation", "特征表示"),
    universe: tr("Candidate universe", "候选空间"),
    filter: tr("Filtering", "候选过滤"),
    router: tr("Routing", "路线选择"),
    model: tr("Model computation", "模型计算"),
    seed: tr("Known-evidence expansion", "已知证据扩展"),
    fusion: tr("Result fusion", "多路结果融合"),
    novelty: tr("Novel association filter", "新关联过滤"),
    rescue: tr("Candidate rescue", "补充候选"),
    rank: tr("Ranking", "候选排序"),
    trust: tr("Evidence interpretation", "证据解释"),
    output: tr("Output", "结果输出"),
    control: tr("Workflow step", "流程步骤"),
  };

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function localizedApiError(data, status) {
    const code = String(data?.error?.code || "");
    if (uiLanguage === "zh") return data?.error?.message || `请求失败 (${status})`;
    const messages = {
      deepseek_key_missing: "Natural-language routing is not configured.",
      deepseek_agent_failed: "The request could not be interpreted. Try a clearer description or an explicit database identifier.",
      agent_direction_unclear: "The task direction is unclear. Specify a reaction, enzyme, route-design goal, or multi-step pathway.",
      protein_no_match: "No verifiable protein record was found.",
      protein_unverified: "The protein could not be verified in UniProt.",
      protein_sequence_missing: "The UniProt record has no sequence available for model retrieval.",
      rhea_no_match: "No verifiable reaction was found in Rhea.",
      rhea_not_found: "The requested Rhea record was not found.",
      rhea_unavailable: "Rhea is temporarily unavailable.",
      model_failed: "Enzyme discovery ranking did not complete.",
      e2r_model_failed: "Reaction discovery ranking did not complete.",
      route_design_target_missing: "No target product was identified for route design.",
      deepseek_route_design_failed: "The route-design goal could not be interpreted.",
      deepseek_pathway_failed: "The multi-step pathway could not be interpreted.",
      feedback_empty: "Choose a rating or write a comment.",
      internal_error: "The service could not complete the request.",
    };
    return messages[code] || (code ? `The request could not be completed (${code}).` : `Request failed (${status}).`);
  }

  function api(path, payload) {
    const requestPayload = payload === undefined ? undefined : { ...payload, ui_language: uiLanguage };
    return fetch(path, {
      method: payload === undefined ? "GET" : "POST",
      headers: payload === undefined ? {} : { "Content-Type": "application/json" },
      body: requestPayload === undefined ? undefined : JSON.stringify(requestPayload),
    }).then(async (response) => {
      let data = null;
      try { data = await response.json(); } catch (_) { /* no-op */ }
      if (!response.ok) {
        const backendMessage = String(data?.error?.message || "");
        const safeMessage = uiLanguage === "zh" || !containsCjk(backendMessage) ? backendMessage : "";
        const error = new Error(safeMessage || tr(`Request failed (${response.status})`, `请求失败 (${response.status})`));
        error.code = data?.error?.code || "request_failed";
        throw error;
      }
      return data;
    });
  }

  function setBusy(value) {
    busy = value;
    sendButton.disabled = value;
    input.disabled = value;
    sendButton.classList.toggle("busy", value);
  }

  function scrollConversation(behavior = "smooth") {
    requestAnimationFrame(() => messages.scrollTo({ top: messages.scrollHeight, behavior }));
  }

  function preserveConversationScroll(callback) {
    const top = messages.scrollTop;
    callback();
    messages.scrollTop = top;
    requestAnimationFrame(() => { messages.scrollTop = top; });
  }

  function messageShell(type = "assistant") {
    const article = el("article", `message ${type === "user" ? "user-message" : "assistant-message"}`);
    if (type === "assistant") article.appendChild(el("div", "assistant-avatar", "CF"));
    const content = el("div", "message-content");
    const meta = el("div", "message-meta");
    meta.append(el("strong", "", type === "user" ? tr("You", "你") : "Catalyst Finder"), el("span", "", tr("now", "刚刚")));
    content.appendChild(meta);
    article.appendChild(content);
    messages.appendChild(article);
    return { article, content };
  }

  function addUserMessage(text, continued) {
    const { content } = messageShell("user");
    const bubble = el("div", "user-bubble");
    bubble.appendChild(el("p", "", text));
    if (continued) bubble.appendChild(el("span", "context-tag", tr("Follow-up", "继续上一轮")));
    content.appendChild(bubble);
    scrollConversation();
  }

  function addError(message, title = tr("This step did not complete", "这一步没有完成")) {
    const { content } = messageShell("assistant");
    const card = el("div", "inline-error");
    card.append(el("span", "error-mark", "!"));
    const copy = el("div");
    copy.append(el("strong", "", title), el("p", "", message));
    card.appendChild(copy);
    content.appendChild(card);
    scrollConversation();
  }

  function addActivity(title) {
    const { content } = messageShell("assistant");
    const card = el("div", "activity-card");
    const dot = el("span", "pulse-dot");
    const copy = el("div", "activity-copy");
    const strong = el("strong", "", title);
    const small = el("small", "", tr("Processing", "正在处理"));
    copy.append(strong, small);
    card.append(dot, copy);
    content.appendChild(card);
    scrollConversation();
    return {
      update(nextTitle, detail = tr("Processing", "正在处理")) {
        strong.textContent = nextTitle;
        small.textContent = detail;
      },
      finish(nextTitle, detail = tr("Completed", "已完成")) {
        strong.textContent = nextTitle;
        small.textContent = detail;
        dot.classList.add("done");
      },
      fail(nextTitle = tr("Did not complete", "没有完成")) {
        strong.textContent = nextTitle;
        small.textContent = tr("Check the input and try again", "请检查输入后重试");
        dot.classList.add("failed");
      },
    };
  }

  const agentToolLabels = {
    resolve_reaction: ["Resolve reaction", "解析反应"],
    resolve_protein_scope: ["Resolve protein scope", "解析蛋白范围"],
    lookup_recorded_associations: ["Read recorded associations", "查询已记录关联"],
    summarize_recorded_relations: ["Summarize recorded reactions", "汇总已记录反应"],
    broaden_protein_scope: ["Broaden annotation scope", "扩大注释范围"],
    prepare_candidate_retrieval: ["Prepare model candidate search", "准备模型候选筛选"],
    prepare_route_design: ["Prepare route design", "准备路线设计"],
    prepare_pathway_compatibility: ["Prepare pathway evaluation", "准备路径评估"],
    legacy_agent_resolution: ["Compatibility resolver", "兼容解析流程"],
  };

  function agentStepLabel(step) {
    if (step.action_kind === "deterministic_fast_path") return tr("Verify structured input", "核对结构化输入");
    if (step.action_kind === "ask_user") return tr("Clarify the task", "确认任务范围");
    if (step.action_kind === "controller_error") return tr("Recover controller", "恢复智能体控制");
    if (step.action_kind === "fallback") return tr("Use compatibility path", "使用兼容流程");
    const labels = agentToolLabels[step.tool];
    return labels ? (uiLanguage === "zh" ? labels[1] : labels[0]) : (step.tool || tr("Scientific step", "科学处理步骤")).replaceAll("_", " ");
  }

  function renderAgentExecution(execution) {
    const steps = Array.isArray(execution?.steps) ? execution.steps : [];
    if (!steps.length) return;
    const { content } = messageShell("assistant");
    const details = el("details", "agent-trace-card");
    details.open = steps.length <= 4;
    const summary = el("summary");
    const title = el("span", "agent-trace-title", tr("Scientific agent", "科学智能体"));
    const meta = el("small", "", tr(`${steps.length} step${steps.length === 1 ? "" : "s"}`, `${steps.length} 个步骤`));
    if (execution.fallback) meta.append(" · ", tr("compatibility recovery", "兼容恢复"));
    summary.append(title, meta);
    details.appendChild(summary);
    const list = el("div", "agent-trace-list");
    steps.forEach((step, index) => {
      const row = el("div", `agent-trace-step status-${step.status || "ok"}`);
      row.append(
        el("span", "agent-trace-index", String(index + 1).padStart(2, "0")),
        el("span", "agent-trace-step-label", agentStepLabel(step)),
        el("small", "agent-trace-status", step.status === "error" || step.status === "rejected"
          ? tr("Recovered", "已恢复")
          : step.status === "needs_input" ? tr("Needs input", "需要确认") : tr("Done", "完成")),
      );
      list.appendChild(row);
    });
    details.appendChild(list);
    content.appendChild(details);
    scrollConversation();
  }

  function resetProcess() {
    processList.querySelectorAll("li").forEach((item) => item.className = "");
  }

  function advanceProcess(stage) {
    const currentIndex = processOrder.indexOf(stage);
    processList.querySelectorAll("li").forEach((item) => {
      const index = processOrder.indexOf(item.dataset.stage);
      item.className = index < currentIndex ? "done" : index === currentIndex ? "active" : "";
    });
  }

  function completeProcess() {
    processList.querySelectorAll("li").forEach((item) => item.className = "done");
  }

  function setDirection(value) {
    const accepted = ["auto", "reaction_to_enzyme", "enzyme_to_reaction", "route_design", "pathway_compatibility"];
    directionHint = accepted.includes(value) ? value : "auto";
    // Route/pathway starter hints are internal disambiguation contracts, not new UI
    // modes. Keep “自动判断” visibly active so the interface stays natural-language-first.
    const visibleDirection = ["route_design", "pathway_compatibility"].includes(directionHint) ? "auto" : directionHint;
    document.querySelectorAll("[data-direction]").forEach((node) => {
      node.classList.toggle("active", node.dataset.direction === visibleDirection);
    });
  }

  function setRouteMode(value) {
    routeMode = value === "default" ? "default" : "intelligent";
    document.querySelectorAll("[data-route-mode]").forEach((node) => {
      node.classList.toggle("active", node.dataset.routeMode === routeMode);
    });
  }

  function externalLink(url, text) {
    const a = el("a", "external-link", text);
    a.href = url;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    return a;
  }

  function sourceBadge(candidate) {
    if (candidate.input_mode === "raw_protein_sequence") return tr("Provided sequence", "用户提供序列");
    if (candidate.input_mode === "general_merged_sequence_match") return tr("Provided sequence · database match", "用户序列 · 数据库匹配");
    if (candidate.source === "model_catalog") return tr("Project model", "项目模型");
    if (candidate.model_ready) return tr("UniProt · project model", "UniProt · 项目模型");
    return tr("UniProt · verified record", "UniProt · 已核对记录");
  }

  function updateGroupToggle(group, mode = "change") {
    const toggle = group?._toggleButton;
    if (!toggle) return;
    const count = Number(group.dataset.optionCount || 0);
    if (mode === "expanded") toggle.textContent = tr("Hide alternatives", "收起其他结果");
    else if (mode === "initial") toggle.textContent = tr(`Show ${Math.max(0, count - 1)} alternatives`, `查看其他 ${Math.max(0, count - 1)} 个结果`);
    else toggle.textContent = tr("Change", "更改");
  }

  function collapseSelectedGroup(group, mode = "change") {
    if (!group) return;
    preserveConversationScroll(() => {
      group.classList.add("selection-collapsed");
      updateGroupToggle(group, mode);
    });
  }

  function expandGroup(group) {
    if (!group) return;
    preserveConversationScroll(() => {
      group.classList.remove("selection-collapsed");
      updateGroupToggle(group, "expanded");
    });
  }

  function prepareCollapsibleGroup(section, group) {
    const options = group.querySelectorAll(".entity-option");
    group.dataset.optionCount = String(options.length);
    if (options.length <= 1) return;
    group.classList.add("selection-collapsed");
    const toggle = el("button", "selection-change", tr(`Show ${options.length - 1} alternatives`, `查看其他 ${options.length - 1} 个结果`));
    toggle.type = "button";
    group._toggleButton = toggle;
    toggle.addEventListener("click", (event) => {
      event.preventDefault();
      if (group.classList.contains("selection-collapsed")) expandGroup(group);
      else collapseSelectedGroup(group, "initial");
    });
    section.querySelector(".verification-section-head")?.appendChild(toggle);
  }

  function triggerActiveVerification() {
    if (busy || !activeVerification || activeVerification.button.disabled) return false;
    activeVerification.button.click();
    return true;
  }

  function bindStableEntitySelection(label, radio) {
    radio.tabIndex = -1;
    const select = () => {
      const group = label.closest(".entity-list");
      preserveConversationScroll(() => {
        radio.checked = true;
        if (group) {
          group.querySelectorAll(".entity-option").forEach((node) => {
            const inputNode = node.querySelector("input[type=radio]");
            const selected = Boolean(inputNode?.checked);
            node.classList.toggle("selected", selected);
            node.setAttribute("aria-checked", selected ? "true" : "false");
          });
          group.classList.add("selection-collapsed");
          updateGroupToggle(group, "change");
        }
      });
      label.blur();
      input.focus({ preventScroll: true });
    };

    label.tabIndex = 0;
    label.setAttribute("role", "radio");
    label.setAttribute("aria-checked", radio.checked ? "true" : "false");
    label.addEventListener("click", (event) => {
      if (event.target.closest("a")) return;
      event.preventDefault();
      select();
    });
    label.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      select();
    });
  }

  function proteinOption(candidate, name, checked) {
    const label = el("label", `entity-option protein-option ${checked ? "selected" : ""}`);
    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = name;
    radio.value = candidate.id;
    radio.checked = checked;
    radio.dataset.inputMode = candidate.input_mode || "";
    if (candidate.sequence) radio.dataset.sequence = candidate.sequence;
    const dot = el("span", "option-radio");
    const main = el("span", "entity-main");
    const top = el("span", "entity-top");
    top.append(el("strong", "", candidate.name || candidate.id), el("em", "", sourceBadge(candidate)));
    const idline = el("span", "entity-idline", candidate.accession ? `${candidate.id} · ${candidate.accession}` : candidate.id);
    const meta = [candidate.organism, candidate.gene_names?.length ? candidate.gene_names.join(", ") : null, candidate.length ? `${candidate.length} aa` : null]
      .filter(Boolean).join(" · ");
    main.append(top, idline, el("small", "", meta || tr("Protein record", "蛋白记录")));
    label.append(radio, dot, main);
    if (candidate.url) label.appendChild(externalLink(candidate.url, "UniProt ↗"));
    bindStableEntitySelection(label, radio);
    return label;
  }

  function reactionOption(candidate, name, checked) {
    const label = el("label", `entity-option reaction-option ${checked ? "selected" : ""}`);
    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = name;
    radio.value = candidate.rhea_id;
    radio.dataset.orientation = candidate.orientation || "forward";
    radio.dataset.inputMode = candidate.input_mode || "";
    if (candidate.reaction_smiles) radio.dataset.reactionSmiles = candidate.reaction_smiles;
    radio.checked = checked;
    const dot = el("span", "option-radio");
    const main = el("span", "entity-main");
    const top = el("span", "entity-top");
    const reactionBadge = candidate.input_mode === "raw_reaction_smiles"
      ? tr("Provided reaction structure", "用户提供反应结构")
      : candidate.model_ready
        ? tr("Rhea · project model", "Rhea · 项目模型")
        : tr("Verified Rhea reaction", "已核对 Rhea 反应");
    top.append(el("strong", "", candidate.rhea_id), el("em", "", reactionBadge));
    main.append(top, el("span", "reaction-equation", candidate.equation || ""));
    const meta = [];
    if (candidate.enzyme_count !== null && candidate.enzyme_count !== undefined) meta.push(tr(`Rhea links ${candidate.enzyme_count} enzyme record(s)`, `Rhea 已关联 ${candidate.enzyme_count} 个酶记录`));
    if (candidate.orientation === "reverse") meta.push(tr("Using reverse orientation", "将按反向反应处理"));
    main.appendChild(el("small", "", meta.join(" · ") || tr("Verified in Rhea", "已由 Rhea 核对")));
    label.append(radio, dot, main);
    if (candidate.url) label.appendChild(externalLink(candidate.url, "Rhea ↗"));
    bindStableEntitySelection(label, radio);
    return label;
  }

  function compoundOption(candidate, name, checked, roleLabel = tr("Compound", "化合物")) {
    const label = el("label", `entity-option compound-option ${checked ? "selected" : ""}`);
    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = name;
    radio.value = candidate.chebi_id || "";
    radio.checked = checked;
    const dot = el("span", "option-radio");
    const main = el("span", "entity-main");
    const top = el("span", "entity-top");
    top.append(el("strong", "", candidate.name || candidate.chebi_id || roleLabel), el("em", "", candidate.chebi_id || "Rhea participant"));
    main.append(top);
    const meta = candidate.smiles ? tr(`Structure verified against the Rhea/ChEBI participant index · ${candidate.smiles.slice(0, 86)}${candidate.smiles.length > 86 ? "…" : ""}`, `结构已与 Rhea/ChEBI 参与物索引核对 · ${candidate.smiles.slice(0, 86)}${candidate.smiles.length > 86 ? "…" : ""}`) : tr("Verified against the Rhea participant index", "已与 Rhea 参与物索引核对");
    main.appendChild(el("small", "", meta));
    const chebiUrl = candidate.chebi_id ? `https://www.ebi.ac.uk/chebi/searchId.do?chebiId=${encodeURIComponent(candidate.chebi_id)}` : "#";
    label.append(radio, dot, main, externalLink(chebiUrl, "ChEBI ↗"));
    bindStableEntitySelection(label, radio);
    return label;
  }

  function verificationSection(title, subtitle) {
    const section = el("section", "verification-section");
    const head = el("div", "verification-section-head");
    const copy = el("div");
    copy.append(el("strong", "", title), el("small", "", subtitle));
    head.appendChild(copy);
    section.appendChild(head);
    return section;
  }

  function taskTargetFromResolution(resolution) {
    if (resolution.direction === "reaction_to_enzyme") {
      return resolution.reaction_resolution?.recommended_id || tr("Pending", "待确认");
    }
    if (resolution.direction === "route_design") {
      const rd = resolution.route_design_resolution || {};
      return rd.recommended_target_id || rd.target_terms?.[0] || tr("Pending target", "待确认目标");
    }
    if (resolution.direction === "pathway_compatibility") {
      const count = resolution.pathway_resolution?.steps?.length || 0;
      return count ? tr(`${count}-step pathway`, `${count} 步路径`) : tr("Pending pathway", "待确认路径");
    }
    return resolution.protein_resolution?.recommended_id || tr("Pending", "待确认");
  }

  function updateContextBeforeRun(resolution) {
    const direction = resolution.direction;
    const taskLabel = directionLabels[direction] || tr("Experimental retrieval", "实验筛选");
    contextTitle.textContent = taskLabel;
    contextSummary.textContent = resolution.summary || tr("Goal understood. Please verify the database records.", "已理解目标，等待你确认数据库记录。");
    const facts = contextFacts.querySelectorAll("span");
    facts[0].querySelector("strong").textContent = taskLabel;
    facts[1].querySelector("strong").textContent = taskTargetFromResolution(resolution);
    facts[2].querySelector("strong").textContent = direction === "pathway_compatibility" ? tr("Joint selection", "联合选择") : direction === "route_design" ? tr("Route ranking", "路线排序") : tr("Known evidence + discovery", "已知证据 + 新关联候选");
  }

  function updateTechnicalLanguage(provenance) {
    if (provenance?.live_verified) {
      techLanguageModel.textContent = `${provenance.provider || "DeepSeek"} · ${provenance.model || "API"} · ${tr("live verified", "已验证调用")}`;
      return;
    }
    if (serviceSnapshot?.deepseek?.live_verified) {
      techLanguageModel.textContent = `${serviceSnapshot.deepseek.provider || "DeepSeek"} · ${serviceSnapshot.deepseek.model || "API"} · ${tr("live verified", "已验证调用")}`;
      return;
    }
    if (serviceSnapshot?.deepseek_configured) {
      techLanguageModel.textContent = `${serviceSnapshot.deepseek_model || "DeepSeek"} · ${tr("configured", "已配置")}`;
      return;
    }
    techLanguageModel.textContent = tr("Not configured", "未配置");
  }

  function maybePrewarmProteinEncoder(resolution) {
    const proteinCandidates = [
      ...(resolution?.protein_resolution?.candidates || []),
      ...(resolution?.positive_enzyme_resolutions || []).flatMap((group) => group?.candidates || []),
    ];
    const needsRawSequenceEncoder = proteinCandidates.some((candidate) =>
      candidate?.input_mode === "raw_protein_sequence" && Boolean(candidate?.sequence));
    if (!needsRawSequenceEncoder) return;
    api("/api/warmup/protein-encoder", {}).catch(() => {
      // Warmup is an optional latency optimization. The confirmed ranking request
      // remains authoritative and can load the same fixed encoder itself.
    });
  }

  function renderVerification(resolution, displayText, effectiveText) {
    updateContextBeforeRun(resolution);
    updateTechnicalLanguage(resolution.llm_provenance);
    advanceProcess("verify");

    const { content } = messageShell("assistant");
    const copy = el("div", "assistant-copy");
    copy.append(el("p", "", resolution.summary || tr("I found database records that can be verified.", "我找到了可核对的数据库记录。")));
    copy.append(el("p", "subtle", tr("Confirm the record that matches your experimental target. The best match is shown first; alternatives can be expanded if needed.", "请确认与实验目标一致的记录。默认只显示最可能的匹配，需要时可以展开其他结果。")));
    content.appendChild(copy);

    const card = el("div", "verification-card");
    const cardHead = el("div", "tool-card-head");
    cardHead.append(el("span", "tool-icon", "✓"), el("div", "", ""));
    cardHead.querySelector("div").append(el("strong", "", tr("Confirm database records", "请确认数据库记录")), el("small", "", tr("Candidate ranking starts after confirmation", "确认后再开始筛选")));
    card.appendChild(cardHead);

    if (resolution.direction === "reaction_to_enzyme") {
      const reaction = resolution.reaction_resolution;
      const rsec = verificationSection(tr("Target reaction", "目标反应"), reaction?.interpreted_reaction || tr("Rhea match", "Rhea 匹配结果"));
      const list = el("div", "entity-list");
      const rname = `reaction-${Math.random().toString(36).slice(2)}`;
      (reaction?.candidates || []).forEach((candidate, index) => {
        const checked = candidate.rhea_id === reaction.recommended_id || (!reaction.recommended_id && index === 0);
        list.appendChild(reactionOption(candidate, rname, checked));
      });
      rsec.appendChild(list);
      card.appendChild(rsec);
      prepareCollapsibleGroup(rsec, list);

      (resolution.positive_enzyme_resolutions || []).forEach((group, groupIndex) => {
        const psec = verificationSection(tr(`Known active enzyme${resolution.positive_enzyme_resolutions.length > 1 ? ` ${groupIndex + 1}` : ""}`, `已知有效酶${resolution.positive_enzyme_resolutions.length > 1 ? ` ${groupIndex + 1}` : ""}`), group.mention || tr("Protein match", "蛋白匹配结果"));
        const plist = el("div", "entity-list");
        const pname = `positive-${groupIndex}-${Math.random().toString(36).slice(2)}`;
        (group.candidates || []).forEach((candidate, index) => {
          const checked = candidate.id === group.recommended_id || (!group.recommended_id && index === 0);
          plist.appendChild(proteinOption(candidate, pname, checked));
        });
        if (!(group.candidates || []).length) {
          psec.appendChild(el("p", "empty-inline", tr("No sufficiently matching protein record was found; this description will not be used as known-active evidence.", "没有找到足够匹配的蛋白记录，这条描述不会作为已知有效酶使用。")));
        } else {
          psec.appendChild(plist);
          prepareCollapsibleGroup(psec, plist);
        }
        card.appendChild(psec);
      });
    } else if (resolution.direction === "route_design") {
      const rd = resolution.route_design_resolution || {};
      if ((rd.source_candidates || []).length) {
        const ssec = verificationSection(tr("Starting precursor", "起始前体"), (rd.source_terms || []).join(" / ") || tr("Rhea / ChEBI match", "Rhea / ChEBI 匹配结果"));
        const slist = el("div", "entity-list route-source-list");
        const sname = `route-source-${Math.random().toString(36).slice(2)}`;
        (rd.source_candidates || []).forEach((candidate, index) => {
          const checked = candidate.chebi_id === rd.recommended_source_id || (!rd.recommended_source_id && index === 0);
          slist.appendChild(compoundOption(candidate, sname, checked, tr("Starting precursor", "起始前体")));
        });
        ssec.appendChild(slist);
        card.appendChild(ssec);
        prepareCollapsibleGroup(ssec, slist);
      } else if (rd.host_pool_supported) {
        const ssec = verificationSection(tr("Route origin", "路线起点"), rd.host || "E. coli");
        ssec.appendChild(el("p", "pathway-auto-enzyme", tr(`Routes will be searched from the ${rd.host || "E. coli"} iML1515 metabolite pool; no precursor will be invented.`, `将从 ${rd.host || "E. coli"} 的 iML1515 代谢物池中寻找可达目标的候选路线，不会凭空指定前体。`)));
        card.appendChild(ssec);
      }
      const tsec = verificationSection(tr("Target product", "目标产物"), (rd.target_terms || []).join(" / ") || tr("Rhea / ChEBI match", "Rhea / ChEBI 匹配结果"));
      const tlist = el("div", "entity-list route-target-list");
      const tname = `route-target-${Math.random().toString(36).slice(2)}`;
      (rd.target_candidates || []).forEach((candidate, index) => {
        const checked = candidate.chebi_id === rd.recommended_target_id || (!rd.recommended_target_id && index === 0);
        tlist.appendChild(compoundOption(candidate, tname, checked, tr("Target product", "目标产物")));
      });
      tsec.appendChild(tlist);
      card.appendChild(tsec);
      prepareCollapsibleGroup(tsec, tlist);
      const policy = ({ short: tr("Prefer shorter routes", "优先短路线"), enzyme_available: tr("Prefer enzyme availability", "优先酶可获得性"), project_covered: tr("Prefer project model coverage", "优先项目模型覆盖"), thermodynamic: tr("Prefer thermodynamic driving force", "优先热力学驱动力"), host_flux: tr("Prefer host-supported flux", "优先宿主可承载通量"), balanced: tr("Balanced feasibility", "综合可实现性") })[rd.priority] || tr("Balanced feasibility", "综合可实现性");
      card.appendChild(el("p", "pathway-auto-enzyme", tr(`${policy} · up to ${rd.max_steps || 6} steps · return ${rd.route_count || 10}. Change preferences directly in natural language.`, `${policy} · 最多 ${rd.max_steps || 6} 步 · 返回 ${rd.route_count || 10} 条。需要改变偏好时，直接在自然语言里说。`)));
    } else if (resolution.direction === "pathway_compatibility") {
      const pathway = resolution.pathway_resolution || {};
      (pathway.steps || []).forEach((step, stepIndex) => {
        const section = verificationSection(tr(`Step ${stepIndex + 1}`, `第 ${stepIndex + 1} 步`), step.mention || tr("Verify this reaction step", "核对这一步反应"));
        section.classList.add("pathway-step-section");
        section.dataset.pathwayStep = String(stepIndex);

        const reaction = step.reaction_resolution || {};
        const list = el("div", "entity-list pathway-reaction-list");
        const rname = `pathway-reaction-${stepIndex}-${Math.random().toString(36).slice(2)}`;
        (reaction.candidates || []).forEach((candidate, index) => {
          const checked = candidate.rhea_id === reaction.recommended_id || (!reaction.recommended_id && index === 0);
          list.appendChild(reactionOption(candidate, rname, checked));
        });
        section.appendChild(list);
        prepareCollapsibleGroup(section, list);

        const enzyme = step.enzyme_resolution || {};
        if (enzyme.specified) {
          const label = el("div", "pathway-enzyme-label");
          label.append(el("strong", "", tr("Specified enzyme", "你指定的酶")), el("small", "", enzyme.interpreted_protein || tr("Verify the protein record", "请核对蛋白记录")));
          section.appendChild(label);
          const plist = el("div", "entity-list pathway-enzyme-list");
          const pname = `pathway-enzyme-${stepIndex}-${Math.random().toString(36).slice(2)}`;
          (enzyme.candidates || []).forEach((candidate, index) => {
            const checked = candidate.id === enzyme.recommended_id || (!enzyme.recommended_id && index === 0);
            plist.appendChild(proteinOption(candidate, pname, checked));
          });
          if ((enzyme.candidates || []).length) {
            section.appendChild(plist);
            prepareCollapsibleGroup(section, plist);
          } else {
            section.appendChild(el("p", "empty-inline", tr("No verifiable protein record was found; revise the enzyme description for this step.", "没有找到可核对的蛋白记录；请修改这一步的酶描述后再试。")));
          }
        } else {
          section.appendChild(el("p", "pathway-auto-enzyme", tr("No enzyme was specified for this step. After pathway confirmation, candidates will be selected jointly across steps.", "这一步未指定酶：确认路径后，系统会从候选中与其他步骤一起联合选择。")));
        }
        card.appendChild(section);
      });
    } else {
      const protein = resolution.protein_resolution;
      if (protein?.mode === "protein_family") {
        const family = protein.family || {};
        const psec = verificationSection(
          tr("Target protein family", "目标蛋白家族"),
          family.label || protein.interpreted_protein || protein.recommended_id || tr("Protein family", "蛋白家族"),
        );
        const scope = el("div", "pathway-auto-enzyme");
        const familyId = family.family_id || protein.recommended_id || "";
        const memberCount = Number(family.member_count || 0);
        scope.appendChild(el(
          "p",
          "",
          tr(
            `${familyId} · ${memberCount} model-candidate member${memberCount === 1 ? "" : "s"} in the current family scope.`,
            `${familyId} · 当前家族范围包含 ${memberCount} 个模型候选成员。`,
          ),
        ));
        const familyScopeNote = uiLanguage === "zh" ? (family.scope_note_zh || family.scope_note) : family.scope_note;
        if (familyScopeNote) {
          scope.appendChild(el("p", "subtle", familyScopeNote));
        }
        const familyCaution = uiLanguage === "zh" ? (family.caution_zh || family.caution) : family.caution;
        if (familyCaution) {
          scope.appendChild(el("p", "subtle", localizedBackendText(
            familyCaution,
            "Family membership defines the query scope; it is not catalytic validation.",
            familyCaution,
          )));
        }
        if ((family.member_ids_sample || []).length) {
          scope.appendChild(el(
            "small",
            "",
            tr(
              `Example members: ${family.member_ids_sample.slice(0, 6).join(", ")}`,
              `部分成员：${family.member_ids_sample.slice(0, 6).join("、")}`,
            ),
          ));
        }
        psec.appendChild(scope);
        card.appendChild(psec);
      } else {
        const psec = verificationSection(tr("Target enzyme", "目标酶"), protein?.interpreted_protein || tr("Protein match", "蛋白匹配结果"));
        const plist = el("div", "entity-list");
        const pname = `query-protein-${Math.random().toString(36).slice(2)}`;
        (protein?.candidates || []).forEach((candidate, index) => {
          const checked = candidate.id === protein.recommended_id || (!protein.recommended_id && index === 0);
          plist.appendChild(proteinOption(candidate, pname, checked));
        });
        psec.appendChild(plist);
        card.appendChild(psec);
        prepareCollapsibleGroup(psec, plist);
      }
    }

    const footer = el("div", "verification-actions");
    const pathwayTask = resolution.direction === "pathway_compatibility";
    const routeDesignTask = resolution.direction === "route_design";
    footer.appendChild(el("p", "", pathwayTask
      ? tr("Review each Rhea step and any enzymes you specified. Unspecified steps will be selected jointly after confirmation.", "逐步核对 Rhea 记录和你已指定的酶；其余步骤会在确认后联合选择。")
      : routeDesignTask
        ? tr("Confirm the source and target, then generate candidate routes from the Rhea graph.", "确认起点和目标后，系统会从 Rhea 反应图生成候选路线。")
        : resolution.protein_resolution?.mode === "protein_family"
          ? tr("Confirm the family scope to summarize database-recorded reactions across its members. Choose a concrete member later for sequence-specific neural predictions.", "确认家族范围后，将汇总成员在数据库中已有记录的反应；如果需要神经模型预测潜在反应，再进一步选择具体成员或提供具体序列。")
          : tr("Open Rhea / UniProt to inspect source records. Press Enter to confirm and continue.", "可以打开 Rhea / UniProt 查看原始记录。按 Enter 也可以确认并继续。")));
    const runText = pathwayTask ? tr("Confirm pathway & evaluate", "确认路径并评估") : routeDesignTask ? tr("Confirm target & design routes", "确认目标并推荐路线") : tr("Confirm & run", "确认并开始筛选");
    const run = el("button", "primary-button", runText);
    run.type = "button";
    run.title = `${runText}（Enter）`;
    run.setAttribute("aria-keyshortcuts", "Enter");
    run.addEventListener("click", () => executeConfirmed(resolution, card, displayText, effectiveText, run));
    footer.appendChild(run);
    card.appendChild(footer);
    content.appendChild(card);
    activeVerification = { card, button: run };
    scrollConversation();
  }

  async function executeConfirmed(resolution, card, displayText, effectiveText, runButton) {
    if (busy) return;
    let payload;
    let selectedTarget = "";

    if (resolution.direction === "route_design") {
      const rd = resolution.route_design_resolution || {};
      const sourceRadio = card.querySelector(".route-source-list .compound-option input:checked");
      const targetRadio = card.querySelector(".route-target-list .compound-option input:checked");
      if ((rd.source_candidates || []).length && !sourceRadio) { addError(tr("Confirm the starting precursor first.", "请先确认起始前体。"), tr("Route origin still needs confirmation", "还需要确认路线起点")); return; }
      if (!targetRadio) { addError(tr("Confirm the target product first.", "请先确认目标产物。"), tr("Route target still needs confirmation", "还需要确认路线目标")); return; }
      selectedTarget = targetRadio.value;
      payload = {
        endpoint: "/api/route/design",
        body: {
          source_chebi_id: sourceRadio?.value || "",
          target_chebi_id: targetRadio.value,
          target_terms: rd.target_terms || [],
          host: rd.host || "",
          max_steps: rd.max_steps || 6,
          route_count: rd.route_count || 10,
          priority: rd.priority || "balanced",
          exploration_policy: rd.exploration_policy || "known_first",
          user_text: effectiveText,
        },
      };
    } else if (resolution.direction === "pathway_compatibility") {
      const stepSections = Array.from(card.querySelectorAll(".pathway-step-section"));
      const steps = [];
      for (const [index, section] of stepSections.entries()) {
        const reactionRadio = section.querySelector(".pathway-reaction-list .reaction-option input:checked");
        if (!reactionRadio) { addError(tr(`Confirm reaction step ${index + 1} first.`, `请先确认第 ${index + 1} 步反应。`), tr("Pathway still needs confirmation", "还需要确认路径")); return; }
        const enzymeRadio = section.querySelector(".pathway-enzyme-list .protein-option input:checked");
        const reactionOptionNode = reactionRadio.closest(".reaction-option");
        const equation = reactionOptionNode?.querySelector(".reaction-equation")?.textContent || "";
        steps.push({
          rhea_id: reactionRadio.value,
          orientation: reactionRadio.dataset.orientation || "forward",
          equation,
          enzyme_id: enzymeRadio?.value || "",
        });
      }
      selectedTarget = tr(`${steps.length}-step pathway`, `${steps.length} 步路径`);
      payload = {
        endpoint: "/api/pathway/analyze",
        body: {
          steps,
          user_text: effectiveText,
          execution_mode: resolution.pathway_resolution?.execution_mode || "auto",
          host: resolution.pathway_resolution?.host || "",
          target_conditions: resolution.pathway_resolution?.target_conditions || {},
        },
      };
    } else if (resolution.direction === "reaction_to_enzyme") {
      const reactionRadio = card.querySelector(".reaction-option input:checked");
      if (!reactionRadio) { addError(tr("Select the target reaction first.", "请先选择目标反应。"), tr("Reaction still needs confirmation", "还需要确认反应")); return; }
      const reactionSmiles = reactionRadio.dataset.reactionSmiles || "";
      const positiveSelections = Array.from(card.querySelectorAll(".protein-option input:checked"));
      const positiveIds = positiveSelections.filter((node) => !node.dataset.sequence).map((node) => node.value);
      const positiveSequenceInputs = positiveSelections.filter((node) => node.dataset.sequence).map((node) => ({
        id: node.value,
        sequence: node.dataset.sequence,
      }));
      selectedTarget = reactionRadio.value;
      payload = {
        endpoint: "/api/rank",
        body: {
          rhea_id: reactionSmiles ? "" : reactionRadio.value,
          reaction_smiles: reactionSmiles,
          query_id: reactionSmiles ? reactionRadio.value : "",
          orientation: reactionRadio.dataset.orientation || "forward",
          user_text: effectiveText,
          route_mode: routeMode,
          conversation_context: {
            previous_direction: continuation?.direction || "",
            previous_result_mode: continuation?.resultMode || "",
            previous_association_policy: continuation?.associationPolicy || "",
            previous_route_id: continuation?.routeId || "",
          },
          confirmed_seed_ids: positiveIds,
          confirmed_seed_inputs: positiveSequenceInputs,
        },
      };
    } else {
      const protein = resolution.protein_resolution || {};
      if (protein.mode === "protein_family") {
        const family = protein.family || {};
        const familyId = family.family_id || protein.recommended_id || "";
        if (!familyId) { addError(tr("Confirm the target protein family first.", "请先确认目标蛋白家族。"), tr("Protein family still needs confirmation", "还需要确认蛋白家族")); return; }
        selectedTarget = family.label || familyId;
        payload = {
          endpoint: "/api/rank-family-reactions",
          body: {
            family_id: familyId,
            user_text: effectiveText,
          },
        };
      } else {
        const proteinRadio = card.querySelector(".protein-option input:checked");
        if (!proteinRadio) { addError(tr("Select the target enzyme first.", "请先选择目标酶。"), tr("Protein still needs confirmation", "还需要确认蛋白")); return; }
        const enzymeSequence = proteinRadio.dataset.sequence || "";
        selectedTarget = proteinRadio.value;
        payload = {
          endpoint: "/api/rank-reactions",
          body: {
            protein_id: enzymeSequence ? "" : proteinRadio.value,
            enzyme_sequence: enzymeSequence,
            query_id: enzymeSequence ? proteinRadio.value : "",
            user_text: effectiveText,
            route_mode: routeMode,
            conversation_context: {
              previous_direction: continuation?.direction || "",
              previous_result_mode: continuation?.resultMode || "",
              previous_association_policy: continuation?.associationPolicy || "",
              previous_route_id: continuation?.routeId || "",
            },
          },
        };
      }
    }

    payload.body.ui_language = uiLanguage;
    payload.body.session_id = activeRun?.session_id || sessionId();
    payload.body.run_id = activeRun?.run_id || newId("run");
    payload.body.step_id = `step_${Date.now().toString(36)}`;
    payload.body.card_id = activeRun?.card_id || "";
    payload.body.card_title = activeRun?.card_title || "";
    payload.body.prompt_template = activeRun?.prompt_template || "";
    payload.body.prompt_source = activeRun?.card_id ? "shortcut_card" : "composer";
    payload.body.edited_after_card_click = Boolean(activeRun?.card_id && effectiveText !== activeRun.prompt_template);

    const facts = contextFacts.querySelectorAll("span strong");
    if (facts[1]) facts[1].textContent = selectedTarget;
    runButton.disabled = true;
    runButton.textContent = resolution.direction === "pathway_compatibility" ? tr("Evaluating jointly…", "正在联合评估…") : resolution.direction === "route_design" ? tr("Generating & ranking routes…", "正在生成并排序路线…") : tr("Running discovery…", "正在筛选…");
    activeVerification = null;
    card.querySelectorAll(".entity-list").forEach((group) => collapseSelectedGroup(group, "change"));
    setBusy(true);
    advanceProcess("search");
    const activity = addActivity(tr("Running retrieval…", "正在筛选候选…"));

    try {
      const result = await api(payload.endpoint, payload.body);
      const pathwayTask = resolution.direction === "pathway_compatibility";
      const routeDesignTask = resolution.direction === "route_design";
      const resultCount = pathwayTask ? (result.steps?.length || 0) : routeDesignTask ? (result.routes?.length || 0) : (result.candidates?.length || 0);
      activity.update(pathwayTask ? tr("Assembling pathway evaluation…", "正在整理整条路径…") : routeDesignTask ? tr("Assembling candidate routes…", "正在整理候选路线…") : tr("Assembling evidence & discovery…", "正在整理结果…"), pathwayTask ? tr(`${resultCount} steps jointly evaluated`, `${resultCount} 个步骤已联合评估`) : routeDesignTask ? tr(`${resultCount} routes ranked`, `${resultCount} 条候选路线已排序`) : tr(`${resultCount} discovery candidates`, `${resultCount} 个新关联候选`));
      advanceProcess("result");
      renderResult(result, resolution.direction);
      updateTechnicalDetails(result);
      activity.finish(pathwayTask ? tr("Pathway evaluation complete", "路径评估完成") : routeDesignTask ? tr("Route design complete", "路线推荐完成") : tr("Retrieval complete", "筛选完成"), pathwayTask ? (uiLanguage === "zh" ? result.verdict_label : tr(`${resultCount} steps evaluated`, `${resultCount} 步已评估`)) : routeDesignTask ? tr(`${resultCount} routes`, `${resultCount} 条候选路线`) : tr(`${resultCount} unrecorded candidates ranked`, `${resultCount} 个新关联候选已排序`));
      completeProcess();
      runButton.textContent = pathwayTask ? tr("Evaluation complete", "评估完成") : routeDesignTask ? tr("Routes ready", "推荐完成") : tr("Retrieval complete", "筛选完成");
      runButton.disabled = true;
      const continuationMode = (!pathwayTask && !routeDesignTask) ? associationMode(result) : null;
      continuation = {
        originalText: effectiveText,
        direction: resolution.direction,
        resultMode: result.discovery_filter?.result_mode || "",
        associationPolicy: continuationMode?.policy || "",
        routeId: result.ranking?.route_id || result.route_view?.route_id || "",
        target: selectedTarget,
      };
      useContinuation = true;
      activeRun = null;
      composerContext.classList.remove("hidden");
      contextSummary.textContent = directionSummary(result, resolution.direction);
      const resultFacts = contextFacts.querySelectorAll("span strong");
      if (resolution.direction === "pathway_compatibility") {
        if (resultFacts[2]) resultFacts[2].textContent = uiLanguage === "zh" ? (result.verdict_label || "联合评估") : tr("Joint evaluation", "联合评估");
      } else if (resolution.direction === "route_design") {
        if (resultFacts[2]) resultFacts[2].textContent = tr(`${result.routes?.length || 0} routes`, `${result.routes?.length || 0} 条路线`);
      } else {
        const actualAssociationMode = associationMode(result);
        if (resultFacts[2]) resultFacts[2].textContent = actualAssociationMode.label;
      }
    } catch (error) {
      activity.fail(tr("Retrieval did not complete", "筛选没有完成"));
      advanceProcess("search");
      addError(error.message, resolution.direction === "pathway_compatibility"
        ? tr("Pathway evaluation did not complete", "整条路径评估没有完成")
        : resolution.direction === "route_design" ? tr("Route generation did not complete", "候选路线生成没有完成")
        : resolution.direction === "reaction_to_enzyme" ? tr("Enzyme candidate ranking did not complete", "候选酶筛选没有完成") : tr("Reaction candidate ranking did not complete", "候选反应筛选没有完成"));
      runButton.disabled = false;
      runButton.textContent = resolution.direction === "pathway_compatibility" ? tr("Confirm pathway & evaluate", "确认路径并评估") : resolution.direction === "route_design" ? tr("Confirm target & design routes", "确认目标并推荐路线") : tr("Confirm & run", "确认并开始筛选");
      activeVerification = { card, button: runButton };
      activeRun = null;
    } finally {
      setBusy(false);
    }
  }

  function associationMode(result) {
    const discovery = result.discovery_filter || {};
    const resultMode = String(discovery.result_mode || "");
    const filterPolicy = String(discovery.policy || "");
    let policy = "allow_known";
    if (resultMode === "known_associations_only" || filterPolicy === "retain_recorded_associations_only") policy = "known_only";
    else if (resultMode === "novel_association_discovery" || filterPolicy === "exclude_recorded_associations") policy = "exclude_known";
    const labels = {
      allow_known: tr("Known evidence + discovery", "已知证据 + 新关联候选"),
      known_only: tr("Known evidence only", "仅已知证据"),
      exclude_known: tr("Discovery only", "仅新关联候选"),
    };
    return {
      policy,
      mixed: policy === "allow_known",
      knownOnly: policy === "known_only",
      excluded: policy === "exclude_known",
      knownCount: Number(discovery.recorded_association_count || 0),
      excludedCount: Number(discovery.excluded_count || 0),
      retainedCount: Number(discovery.retained_count || 0),
      label: labels[policy],
    };
  }

  function directionSummary(result, direction) {
    if (direction === "pathway_compatibility") {
      const count = result.steps?.length || 0;
      const core = result.coverage?.core_condition_steps || 0;
      return tr(
        `${count}-step pathway evaluated jointly. pH / temperature evidence covers ${core}/${count} steps.`,
        `${count} 步路径已联合评估。pH / 温度证据覆盖 ${core}/${count} 步。`,
      );
    }
    if (direction === "route_design") {
      const count = result.routes?.length || 0;
      const target = result.selected_target?.name || result.selected_target?.chebi_id || tr("target product", "目标产物");
      return tr(
        `${count} Rhea-supported candidate route${count === 1 ? "" : "s"} found for ${target}. You can select a route and continue to pathway-level enzyme compatibility analysis.`,
        `已为 ${target} 找到并排序 ${count} 条 Rhea 已知候选路线；可继续指定某条路线做多酶兼容性评估。`,
      );
    }
    const knownCount = Number(result.known_associations?.count || result.discovery_filter?.recorded_association_count || 0);
    const discoveryCount = Number(result.candidates?.length || 0);
    const entity = direction === "reaction_to_enzyme" ? tr("enzyme", "酶") : tr("reaction", "反应");
    const mode = associationMode(result);
    if (mode.knownOnly) return tr(`${knownCount} recorded ${entity}${knownCount === 1 ? "" : "s"} shown as database evidence.`, `展示 ${knownCount} 条数据库已记录${entity}证据。`);
    return tr(
      `${knownCount} recorded ${entity}${knownCount === 1 ? "" : "s"} shown as evidence; ${discoveryCount} unrecorded discovery candidate${discoveryCount === 1 ? "" : "s"} ranked separately.`,
      `展示 ${knownCount} 条数据库已记录${entity}证据，并独立排序 ${discoveryCount} 个尚未记录的新关联候选。`,
    );
  }

  function intervalLabel(interval, suffix = "") {
    if (!Array.isArray(interval) || interval.length < 2) return "";
    const left = Number(interval[0]);
    const right = Number(interval[1]);
    if (!Number.isFinite(left) || !Number.isFinite(right)) return "";
    const fmt = (value) => Number.isInteger(value) ? String(value) : String(Number(value.toFixed(2)));
    return Math.abs(left - right) < 1e-9 ? `${fmt(left)}${suffix}` : `${fmt(left)}–${fmt(right)}${suffix}`;
  }

  function pathwayModeLabel(mode) {
    return ({
      one_pot: tr("One-pot / shared conditions", "同一体系"),
      sequential: tr("Sequential steps", "分步反应"),
      in_vivo: tr("In vivo pathway", "体内路径"),
      auto: tr("Inferred from description", "按描述判断"),
    })[mode] || tr("Inferred from description", "按描述判断");
  }

  function appendPathwayRouteDetails(card, result) {
    const details = document.createElement("details");
    details.className = "result-route-details";
    const summary = document.createElement("summary");
    summary.textContent = tr("Technical details", "查看技术详情");
    details.appendChild(summary);
    const technical = el("div", "result-technical");
    const openRoute = el("button", "result-route-open");
    openRoute.type = "button";
    openRoute.append(
      el("strong", "", tr("View pathway evaluation trace", "查看整条路径评估流程")),
      el("span", "", tr("Open full flow ↗", "打开完整流程图 ↗")),
    );
    openRoute.addEventListener("click", () => openActualRouteDialog(result.route_view || {}));
    technical.appendChild(openRoute);
    technical.appendChild(el("code", "result-route-code", result.route_view?.route_id || "pathway-compatibility-v1"));
    details.appendChild(technical);
    card.appendChild(details);
  }

  function renderPathwayResult(result) {
    const { content } = messageShell("assistant");
    const steps = result.steps || [];
    const coverage = result.coverage || {};
    const shared = result.shared_conditions || {};
    const target = result.target_conditions || {};
    const verdictLabels = {
      compatible: tr("No strong cross-step conflict in available evidence", "现有证据未见明显跨步冲突"),
      partial_evidence: tr("Partial condition evidence", "条件证据不完整"),
      insufficient_evidence: tr("Insufficient condition evidence", "条件证据不足"),
      conflict: tr("Potential cross-step conflict", "存在潜在跨步冲突"),
    };
    const verdictLabel = verdictLabels[result.verdict] || tr("Pathway evaluation complete", result.verdict_label || "路径评估完成");
    const intro = el("div", "assistant-copy result-intro");
    intro.appendChild(el("p", "", verdictLabel));
    intro.appendChild(el("p", "subtle", tr(
      "Compatibility combines model priorities with available UniProt condition annotations. Missing pH or temperature data is reported as unknown.",
      "兼容性评估结合模型优先级和 UniProt 条件注释；缺失的 pH / 温度数据会标记为未知。",
    )));
    content.appendChild(intro);

    const card = el("div", "result-card pathway-result-card");
    const head = el("div", "result-head");
    const titleWrap = el("div");
    titleWrap.append(
      el("strong", "", tr("Pathway enzyme combination", "整条路径的酶组合")),
      el("small", "", tr("Per-step retrieval priority · global condition compatibility", "各步模型优先级 · 条件兼容性全局联合选择")),
    );
    head.appendChild(titleWrap); card.appendChild(head);

    const chips = el("div", "result-chips");
    [
      tr(`${steps.length} steps`, `${steps.length} 步`), pathwayModeLabel(result.execution_mode),
      target.ph !== null && target.ph !== undefined && Number.isFinite(Number(target.ph)) ? tr(`Target pH ${Number(target.ph)}`, `目标 pH ${Number(target.ph)}`) : null,
      target.temperature_c !== null && target.temperature_c !== undefined && Number.isFinite(Number(target.temperature_c)) ? tr(`Target ${Number(target.temperature_c)} °C`, `目标 ${Number(target.temperature_c)} °C`) : null,
      (target.cofactors || []).length ? tr(`Target cofactors ${(target.cofactors || []).join(" / ")}`, `目标辅因子 ${(target.cofactors || []).join(" / ")}`) : null,
      tr(`pH / temperature evidence ${coverage.core_condition_steps || 0}/${coverage.total_steps || steps.length}`, `pH / 温度证据 ${coverage.core_condition_steps || 0}/${coverage.total_steps || steps.length}`),
      shared.ph_label ? tr(`Shared pH ${shared.ph_label}`, `共同 pH ${shared.ph_label}`) : null,
      shared.temperature_label ? tr(`Shared temperature ${shared.temperature_label}`, `共同温度 ${shared.temperature_label}`) : null,
    ].filter(Boolean).forEach((text) => chips.appendChild(el("span", "", text)));
    card.appendChild(chips);

    const stepList = el("div", "pathway-result-steps");
    steps.forEach((step) => {
      const candidate = step.selected_enzyme || {};
      const profile = candidate.condition_profile || {};
      const row = el("article", "pathway-result-step");
      const marker = el("span", "pathway-step-marker", String(step.step_index || ""));
      const body = el("div", "pathway-step-body");
      const top = el("div", "pathway-step-top");
      const title = el("div");
      title.append(el("small", "", step.rhea_id || tr(`Step ${step.step_index}`, `第 ${step.step_index} 步`)));
      const link = externalLink(candidate.uniprot_url || profile.url || "#", candidate.candidate_id || tr("Candidate enzyme", "候选酶"));
      link.classList.add("pathway-enzyme-link"); title.appendChild(link);
      const badges = el("div", "pathway-step-badges");
      if (candidate.local_rank) badges.appendChild(el("span", "", tr(`Step rank #${candidate.local_rank}`, `单步 #${candidate.local_rank}`)));
      if (step.changed_for_pathway_compatibility) badges.appendChild(el("span", "changed", tr("Jointly reranked", "联合重排")));
      top.append(title, badges); body.appendChild(top);
      const meta = [candidate.name, candidate.species].filter(Boolean).join(" · "); if (meta) body.appendChild(el("p", "pathway-step-meta", meta));
      const conditions = el("div", "pathway-condition-chips");
      const ph = profile.ph_active || profile.ph_optimum; const temp = profile.temperature_active_c || profile.temperature_optimum_c;
      const phText = intervalLabel(ph); const tempText = intervalLabel(temp, " °C");
      if (phText) conditions.appendChild(el("span", "", `${profile.ph_active ? tr("pH range", "pH 范围") : tr("optimal pH", "最适 pH")} ${phText}`));
      if (tempText) conditions.appendChild(el("span", "", `${profile.temperature_active_c ? tr("temperature range", "温度范围") : tr("optimal temperature", "最适温度")} ${tempText}`));
      if ((profile.cofactors || []).length) conditions.appendChild(el("span", "", tr(`Cofactors ${(profile.cofactors || []).slice(0,3).join(" / ")}`, `辅因子 ${(profile.cofactors || []).slice(0,3).join(" / ")}`)));
      if (Number.isFinite(Number(profile.theoretical_pi))) conditions.appendChild(el("span", "", tr(`Theoretical pI ${Number(profile.theoretical_pi).toFixed(2)}`, `理论 pI ${Number(profile.theoretical_pi).toFixed(2)}`)));
      if (!phText && !tempText) conditions.appendChild(el("span", "unknown", tr("No structured pH / temperature annotation", "pH / 温度暂无结构化注释")));
      body.appendChild(conditions); row.append(marker, body); stepList.appendChild(row);
    });
    card.appendChild(stepList);

    const conflictBox = el("section", "pathway-conflict-box");
    conflictBox.appendChild(el("strong", "", (result.conflicts || []).length ? tr("Cross-step evidence to review", "需要关注的跨步证据") : tr("No strong cross-step conflict in available evidence", "现有证据未见明显跨步冲突")));
    if ((result.conflicts || []).length) {
      const list = el("div", "pathway-conflict-list");
      (result.conflicts || []).slice(0,8).forEach((item) => {
        const row = el("div", `pathway-conflict-item severity-${item.severity || "medium"}`);
        const englishDetail = ({ph:"Reported pH ranges differ across these steps.",temperature:"Reported temperature ranges differ across these steps.",target_ph:"A selected enzyme is distant from the requested pH.",target_temperature:"A selected enzyme is distant from the requested temperature.",cofactor_regulation:"A cofactor or metal annotation may interfere with another enzyme.",localization:"Subcellular-location annotations differ across steps."})[item.type] || "Condition evidence differs across these steps; inspect the source annotations.";
        row.append(el("span", "", tr(`Steps ${(item.steps || []).join(" / ")}`, `步骤 ${(item.steps || []).join(" / ")}`)), el("p", "", uiLanguage === "zh" ? (item.detail || "条件存在差异，需要人工核对。") : englishDetail));
        list.appendChild(row);
      }); conflictBox.appendChild(list);
    } else conflictBox.appendChild(el("p", "", tr("Available annotations show no explicit cross-step conflict. Unreported conditions remain unknown.", "现有注释中未发现明确的跨步冲突；未报道条件仍保持未知。")));
    card.appendChild(conflictBox);
    card.appendChild(el("p", "score-note", tr("Concentration, pI, salts, buffer, substrates/products, solvents, and reaction time can also affect pathway compatibility and still require experimental validation.", "浓度、pI、盐、buffer、底物/产物、溶剂和反应时间也会影响路径兼容性，仍需结合实验验证。")));
    appendPathwayRouteDetails(card, result); content.appendChild(card); scrollConversation();
  }


  function routePriorityLabel(priority) {
    return ({
      balanced: tr("balanced feasibility", "综合可实现性"),
      short: tr("shorter routes", "优先短路线"),
      enzyme_available: tr("enzyme availability", "优先酶可获得性"),
      project_covered: tr("project-model coverage", "优先项目模型覆盖"),
      thermodynamic: tr("thermodynamic driving force", "优先热力学驱动力"),
      host_flux: tr("host-supported flux", "优先宿主可承载通量"),
    })[priority] || tr("balanced feasibility", "综合可实现性");
  }

  function renderRouteDesignResult(result) {
    const { content } = messageShell("assistant");
    const routes = result.routes || [];
    const intro = el("div", "assistant-copy result-intro");
    const target = result.selected_target?.name || result.selected_target?.chebi_id || tr("target product", "目标产物");
    intro.append(el("p", "", routes.length
      ? tr(`Found ${routes.length} Rhea-supported route${routes.length === 1 ? "" : "s"} to ${target}, ranked by ${routePriorityLabel(result.priority)}.`, `为 ${target} 找到了 ${routes.length} 条 Rhea 已知候选路线，并按${routePriorityLabel(result.priority)}排序。`)
      : tr(`No candidate route to ${target} was found in the current Rhea reaction graph.`, `在当前 Rhea 已知反应图中没有找到通向 ${target} 的候选路线。`)));
    const feas = result.feasibility || {};
    const filtered = Number(feas.host_infeasible_filtered_count || 0);
    const thermoCount = Number(feas.thermo_complete_count || 0);
    const evidenceText = filtered
      ? tr(`${Number(feas.preliminary_route_count || routes.length)} preliminary routes were evaluated; iML1515 route-supported FBA removed ${filtered} zero-flux route(s).`, `先评估了 ${Number(feas.preliminary_route_count || routes.length)} 条预候选；iML1515 route-supported FBA 过滤了 ${filtered} 条整路通量为 0 的路线。`)
      : tr(`MDF was calculated for ${thermoCount}/${Number(feas.preliminary_route_count || routes.length)} preliminary routes.`, `${thermoCount}/${Number(feas.preliminary_route_count || routes.length)} 条预候选完成了 MDF 计算。`);
    intro.append(el("p", "subtle", tr(`${evidenceText}`, `${evidenceText}`)));
    content.appendChild(intro);

    const card = el("div", "result-card route-design-result-card");
    const head = el("div", "result-head");
    const titleWrap = el("div");
    titleWrap.append(el("strong", "", tr("Candidate biosynthetic routes", "候选生物合成路线")), el("small", "", result.feasibility?.host_expected ? "Rhea · eQuilibrator MDF · E. coli iML1515 FBA" : tr("Rhea · eQuilibrator MDF · multi-evidence ranking", "Rhea · eQuilibrator MDF · 多指标相对排序")));
    head.appendChild(titleWrap);
    card.appendChild(head);

    const stats = result.graph_stats || {};
    const chips = el("div", "result-chips");
    [
      tr(`${routes.length} routes`, `${routes.length} 条路线`),
      routePriorityLabel(result.priority),
      result.source_mode === "ecoli_iML1515_pool" ? tr("E. coli iML1515 source pool", "E. coli iML1515 起点池") : tr("Confirmed precursor", "确认前体出发"),
      stats.route_nodes ? tr(`${Number(stats.route_nodes).toLocaleString()} Rhea graph nodes`, `${Number(stats.route_nodes).toLocaleString()} 个 Rhea 图节点`) : null,
      stats.route_edges ? tr(`${Number(stats.route_edges).toLocaleString()} main-transformation edges`, `${Number(stats.route_edges).toLocaleString()} 条主转化边`) : null,
      thermoCount ? tr(`${thermoCount} routes with MDF`, `${thermoCount} 条有 MDF`) : null,
      filtered ? tr(`FBA removed ${filtered} zero-flux routes`, `FBA 过滤 ${filtered} 条零通量路线`) : null,
    ].filter(Boolean).forEach((text) => chips.appendChild(el("span", "", text)));
    card.appendChild(chips);

    const list = el("div", "route-design-list");
    routes.forEach((route) => {
      const item = el("article", "route-design-item");
      const top = el("div", "route-design-item-head");
      const rank = el("span", "route-design-rank", `#${route.rank || ""}`);
      const summary = el("div", "route-design-summary");
      summary.append(el("strong", "", (route.compound_names || []).join(" → ") || route.route_id || tr("Candidate route", "候选路线")));
      summary.append(el("small", "", route.route_id || "Rhea route"));
      const score = el("div", "route-design-score");
      score.append(el("strong", "", Number(route.score || 0).toFixed(1)), el("small", "", tr("relative score", "综合相对分")));
      top.append(rank, summary, score);
      item.appendChild(top);

      const metrics = route.metrics || {};
      const metricRow = el("div", "route-design-metrics");
      const thermo = route.thermodynamics || {};
      const hostFeasibility = route.host_feasibility || {};
      const mdf = Number(thermo.mdf_kj_mol);
      const flux50 = Number(hostFeasibility.max_route_flux_50pct_growth);
      [
        tr(`${metrics.step_count || route.steps?.length || 0} steps`, `${metrics.step_count || route.steps?.length || 0} 步`),
        tr(`Enzyme availability ${Math.round(Number(metrics.enzyme_availability || 0) * 100)}%`, `酶可获得性 ${Math.round(Number(metrics.enzyme_availability || 0) * 100)}%`),
        thermo.status === "complete" && Number.isFinite(mdf) ? `MDF ${mdf.toFixed(1)} kJ/mol` : tr("MDF unavailable", "MDF 未覆盖"),
        hostFeasibility.status === "complete" && Number.isFinite(flux50) ? tr(`iML1515 route flux ${flux50.toFixed(2)} @ ≥50% growth`, `iML1515 路线通量 ${flux50.toFixed(2)} @≥50%生长`) : (result.feasibility?.host_expected ? tr("iML1515 FBA unknown", "iML1515 FBA 未知") : null),
        tr(`Project model coverage ${Math.round(Number(metrics.project_model_coverage || 0) * 100)}%`, `项目模型覆盖 ${Math.round(Number(metrics.project_model_coverage || 0) * 100)}%`),
        Number.isFinite(Number(metrics.min_swissprot_count)) ? tr(`Minimum Swiss-Prot records ${Number(metrics.min_swissprot_count)}`, `最少 Swiss-Prot ${Number(metrics.min_swissprot_count)}`) : null,
      ].filter(Boolean).forEach((text) => metricRow.appendChild(el("span", "", text)));
      item.appendChild(metricRow);

      const steps = el("div", "route-design-steps");
      (route.steps || []).forEach((step) => {
        const row = el("div", "route-design-step");
        row.appendChild(el("span", "route-design-step-index", String(step.step_index || "")));
        const copy = el("div", "route-design-step-copy");
        copy.append(el("strong", "", `${step.source_name || step.source} → ${step.target_name || step.target}`));
        const meta = [];
        if (step.swissprot_count !== undefined) meta.push(tr(`Swiss-Prot enzyme records ${step.swissprot_count}`, `Swiss-Prot 酶记录 ${step.swissprot_count}`));
        const thermoStep = (thermo.steps || []).find((row) => Number(row.step_index) === Number(step.step_index));
        const physiological = Number(thermoStep?.physiological_dg_prime?.value_kj_mol);
        if (Number.isFinite(physiological)) meta.push(`ΔG′(phys) ${physiological.toFixed(1)} kJ/mol`);
        if (step.local_model_ready) meta.push(tr("Project R2E directly supported", "项目 R2E 可直接评估"));
        else meta.push(tr("R2E can use external Rhea SMILES", "可用外部 Rhea SMILES 进入 R2E"));
        copy.appendChild(el("small", "", meta.join(" · ")));
        row.append(copy, externalLink(step.url || `https://www.rhea-db.org/rhea/${String(step.rhea_id || "").replace("RHEA:", "")}`, `${step.rhea_id || "Rhea"} ↗`));
        steps.appendChild(row);
      });
      item.appendChild(steps);

      const action = el("button", "route-design-template-action", tr("Use this route for enzyme compatibility analysis", "填入这条路线继续评估酶兼容性"));
      action.type = "button";
      action.addEventListener("click", () => {
        const chain = (route.compound_names || []).join(" → ");
        const rhea = (route.steps || []).map((step) => step.rhea_id).filter(Boolean).join(" → ");
        input.value = tr(`Evaluate enzyme compatibility across this complete pathway: ${chain}. Rhea steps: ${rhea}. If a step has no specified enzyme, select candidates jointly and check pH, temperature, cofactors, and other condition conflicts.`, `请评估这条完整路径的酶组合兼容性：${chain}。对应 Rhea 步骤：${rhea}。如果某一步没有指定酶，请联合选择候选，并检查 pH、温度、辅因子和其他条件冲突。`);
        useContinuation = false;
        composerContext.classList.add("hidden");
        input.focus({ preventScroll: true });
        input.setSelectionRange(input.value.length, input.value.length);
      });
      item.appendChild(action);
      list.appendChild(item);
    });
    card.appendChild(list);

    const thermoConditions = result.thermodynamics_run?.conditions || {};
    if (result.thermodynamics_run?.status === "complete") {
      const conditionText = [
        Number.isFinite(Number(thermoConditions.p_h)) ? `pH ${Number(thermoConditions.p_h)}` : null,
        Number.isFinite(Number(thermoConditions.p_mg)) ? `pMg ${Number(thermoConditions.p_mg)}` : null,
        Number.isFinite(Number(thermoConditions.ionic_strength_m)) ? `I=${Number(thermoConditions.ionic_strength_m)} M` : null,
        Number.isFinite(Number(thermoConditions.temperature_c)) ? `${Number(thermoConditions.temperature_c)} °C` : null,
      ].filter(Boolean).join(" · ");
      card.appendChild(el("p", "route-design-exploration-note", tr(`MDF: eQuilibrator / equilibrator-pathway, ${conditionText || "default aqueous conditions"}; concentration bounds use eQuilibrator defaults. MDF summarizes thermodynamic driving force under these conditions.`, `MDF：eQuilibrator / equilibrator-pathway，${conditionText || "默认水相条件"}；浓度边界使用 eQuilibrator 默认设置。MDF 表示这些条件下的热力学驱动力。`)));
    }
    if (result.host_feasibility_run?.status === "complete") {
      card.appendChild(el("p", "route-design-exploration-note", tr("iML1515 route flux reports stoichiometric pathway capacity under shared-flux and minimum-growth constraints. Kinetics and fermentation yield require separate evidence.", "iML1515 路线通量表示共同通量和最低生长约束下的化学计量通量容量；动力学与发酵产量需要结合其他证据评估。")));
    }

    const exploratory = result.exploratory_routes || [];
    if (exploratory.length) {
      const section = el("section", "route-exploration-section");
      const sectionHead = el("div", "route-exploration-head");
      const copy = el("div");
      copy.append(el("strong", "", tr("Predicted exploration routes", "预测探索路线")), el("small", "", tr("Ranked separately from Rhea-supported routes", "与 Rhea 已知路线分别排序")));
      sectionHead.append(copy, el("span", "route-exploration-count", tr(`${exploratory.length} routes`, `${exploratory.length} 条`)));
      section.appendChild(sectionHead);
      section.appendChild(el("p", "route-design-exploration-note", uiLanguage === "zh" ? (result.exploration_backend?.predicted_note || "这些路线至少包含一个 MINE/Pickaxe + MetaCyc rule 预测步骤，需要进一步验证。") : "These routes include at least one MINE/Pickaxe + MetaCyc rule-predicted step and require validation."));
      const xlist = el("div", "route-design-list exploratory");
      exploratory.forEach((route) => {
        const item = el("article", "route-design-item predicted-route");
        const top = el("div", "route-design-item-head");
        const rank = el("span", "route-design-rank", `P${route.rank || ""}`);
        const summary = el("div", "route-design-summary");
        summary.append(el("strong", "", (route.compound_names || []).join(" → ") || route.route_id || tr("Predicted route", "预测路线")));
        summary.append(el("small", "", tr(`${route.route_id || "Pickaxe route"} · prediction score is relative within the predicted-route set`, `${route.route_id || "Pickaxe route"} · 预测分数用于预测路线候选内的相对排序`)));
        const score = el("div", "route-design-score");
        score.append(el("strong", "", Number(route.score || 0).toFixed(1)), el("small", "", tr("exploration score", "探索相对分")));
        top.append(rank, summary, score);
        item.appendChild(top);
        const steps = el("div", "route-design-steps");
        (route.steps || []).forEach((step) => {
          const row = el("div", `route-design-step ${step.evidence_type === "predicted_pickaxe" ? "predicted" : "known"}`);
          row.appendChild(el("span", "route-design-step-index", String(step.step_index || "")));
          const body = el("div", "route-design-step-copy");
          body.append(el("strong", "", `${step.source_name || step.source} → ${step.target_name || step.target}`));
          if (step.evidence_type === "predicted_pickaxe") {
            body.appendChild(el("small", "", tr(`Predicted step · MetaCyc rules ${(step.prediction_rules || []).join(" / ") || "not annotated"}`, `预测步骤 · MetaCyc rules ${(step.prediction_rules || []).join(" / ") || "未标注"}`)));
            row.append(body, el("span", "prediction-badge", tr("Predicted", "预测")));
          } else {
            body.appendChild(el("small", "", tr(`Rhea-known step · Swiss-Prot ${step.swissprot_count || 0}`, `Rhea 已知步骤 · Swiss-Prot ${step.swissprot_count || 0}`)));
            row.append(body, externalLink(step.url || `https://www.rhea-db.org/rhea/${String(step.rhea_id || "").replace("RHEA:", "")}`, `${step.rhea_id || "Rhea"} ↗`));
          }
          steps.appendChild(row);
        });
        item.appendChild(steps);
        item.appendChild(el("p", "predicted-route-warning", uiLanguage === "zh" ? (route.evidence_note || "这条路线包含规则预测步骤，需要进一步实验验证。") : "This route contains rule-predicted steps and requires experimental validation."));
        xlist.appendChild(item);
      });
      section.appendChild(xlist);
      card.appendChild(section);
    } else if (result.exploration_backend?.predicted_note) {
      card.appendChild(el("p", "route-design-exploration-note", localizedBackendText(result.exploration_backend.predicted_note, "No predicted routes are available for this request.", result.exploration_backend.predicted_note)));
    }
    card.appendChild(el("p", "score-note", uiLanguage === "zh" ? (result.score_note || "路线分数用于候选路线之间的相对排序。") : "Route scores provide relative priorities within the candidate set."));
    content.appendChild(card);
    scrollConversation();
  }

  function renderResult(result, direction) {
    if (direction === "pathway_compatibility") {
      renderPathwayResult(result);
      return;
    }
    if (direction === "route_design") {
      renderRouteDesignResult(result);
      return;
    }

    const { content } = messageShell("assistant");
    const mode = associationMode(result);
    const known = result.known_associations || { count: 0, items: [] };
    const discoveryRows = mode.knownOnly ? [] : (result.candidates || []);
    const requestedTopK = Number(result.ranking?.top_k || 0);
    const knownLabel = direction === "reaction_to_enzyme" ? tr("Known enzymes", "已知酶") : tr("Known reactions", "已知反应");
    const discoveryLabel = direction === "reaction_to_enzyme" ? tr("Unrecorded candidates", "新关联候选酶") : tr("Unrecorded candidates", "新关联候选反应");

    const intro = el("div", "assistant-copy result-intro evidence-first-intro");
    if (mode.knownOnly) {
      intro.appendChild(el("p", "", tr(
        `${known.count} recorded association${known.count === 1 ? "" : "s"} found.`,
        `找到 ${known.count} 条数据库已知关联。`,
      )));
    } else if (mode.excluded) {
      intro.appendChild(el("p", "", tr(
        `${discoveryRows.length} unrecorded candidate association${discoveryRows.length === 1 ? "" : "s"} ranked.`,
        `筛选出 ${discoveryRows.length} 个新关联候选。`,
      )));
    } else if (known.count) {
      intro.appendChild(el("p", "", tr(
        `${known.count} recorded association${known.count === 1 ? "" : "s"} found, with ${discoveryRows.length} unrecorded candidate${discoveryRows.length === 1 ? "" : "s"} ranked.`,
        `找到 ${known.count} 条数据库已知关联，并筛选出 ${discoveryRows.length} 个新关联候选。`,
      )));
    } else {
      intro.appendChild(el("p", "", tr(
        `No recorded association was found; ${discoveryRows.length} unrecorded candidate${discoveryRows.length === 1 ? "" : "s"} were ranked.`,
        `暂未找到数据库已知关联，筛选出 ${discoveryRows.length} 个新关联候选。`,
      )));
    }
    if (result.family) {
      const scopeNote = uiLanguage === "zh"
        ? (result.family.scope_note_zh || result.family.scope_note)
        : result.family.scope_note;
      const caution = uiLanguage === "zh"
        ? (result.family.caution_zh || result.family.caution)
        : result.family.caution;
      if (scopeNote) intro.appendChild(el("p", "subtle", scopeNote));
      if (caution) intro.appendChild(el("p", "subtle", caution));
    }
    content.appendChild(intro);

    const card = el("div", "result-card evidence-discovery-card");
    const head = el("div", "result-head evidence-discovery-head");
    const titleWrap = el("div");
    titleWrap.append(el("strong", "", tr("Results", "筛选结果")));
    let entityNode;
    if (direction === "reaction_to_enzyme") {
      entityNode = result.reaction?.url
        ? externalLink(result.reaction.url, `${result.reaction?.rhea_id || "Rhea"} ↗`)
        : el("strong", "entity-primary-text", result.reaction?.rhea_id || "Rhea");
    } else if (result.protein?.url) {
      entityNode = externalLink(result.protein.url, `${result.protein?.id || "UniProt"} ↗`);
    } else {
      entityNode = el("strong", "entity-primary-text", result.protein?.name || result.protein?.id || tr("Protein family", "蛋白家族"));
    }
    head.append(titleWrap, entityNode);
    card.appendChild(head);

    const chips = el("div", "result-chips evidence-discovery-chips");
    chips.appendChild(el("span", "evidence-chip", tr(`Known ${known.count || 0}`, `已知 ${known.count || 0}`)));
    if (!mode.knownOnly) chips.appendChild(el("span", "discovery-chip", tr(`Unrecorded ${discoveryRows.length}`, `新关联 ${discoveryRows.length}`)));
    if (requestedTopK && !mode.knownOnly) chips.appendChild(el("span", "", tr(`Top ${requestedTopK}`, `Top ${requestedTopK}`)));
    if (mode.knownOnly) chips.appendChild(el("span", "", tr("Known evidence only", "仅已知证据")));
    else if (mode.excluded) chips.appendChild(el("span", "", tr("Unrecorded candidates only", "仅新关联候选")));
    card.appendChild(chips);

    // Layer 1: factual database evidence. Model coverage is metadata, not a trust tier.
    const evidence = document.createElement(known.count ? "details" : "section");
    evidence.className = "evidence-section";
    if (known.count && !mode.excluded) evidence.open = true;
    if (known.count) {
      const summary = document.createElement("summary");
      const summaryCopy = el("div", "evidence-summary-copy");
      summaryCopy.append(el("strong", "", `${knownLabel} · ${known.count}`));
      summary.appendChild(summaryCopy);
      if (mode.excluded) summary.appendChild(el("span", "evidence-reference-badge", tr("Reference", "参考")));
      evidence.appendChild(summary);

      const grid = el("div", "evidence-grid");
      (known.items || []).forEach((row) => {
        const item = el("article", "evidence-card");
        const top = el("div", "evidence-card-top");
        const link = direction === "reaction_to_enzyme"
          ? externalLink(row.uniprot_url || "#", row.candidate_id)
          : externalLink(row.rhea_url || "#", row.candidate_id);
        link.classList.add("evidence-primary-link");
        top.appendChild(link);
        top.appendChild(el(
          "span",
          `evidence-source ${row.source === "rhea_swissprot" ? "official" : "project"}`,
          row.source === "rhea_swissprot"
            ? "Rhea / Swiss-Prot"
            : row.source === "integrated_family_evidence"
              ? tr("Integrated family evidence", "家族整合证据")
              : tr("Project association catalog", "项目关联库"),
        ));
        item.appendChild(top);

        const meta = direction === "reaction_to_enzyme"
          ? [row.name, row.species].filter(Boolean).join(" · ")
          : row.name || [row.substrate_name, row.product_name].filter(Boolean).join(" → ");
        if (meta) item.appendChild(el("p", "evidence-meta", meta));
        if (row.family_support_count !== undefined && row.family_member_count !== undefined) {
          item.appendChild(el(
            "p",
            "evidence-meta",
            tr(
              `Recorded for ${row.family_support_count} of ${row.family_member_count} members in this family scope.`,
              `当前家族范围内有 ${row.family_support_count}/${row.family_member_count} 个成员记录了这条反应。`,
            ),
          ));
        }

        if (row.model_score !== null && row.model_score !== undefined) {
          const modelMeta = el("div", "evidence-coverage");
          modelMeta.appendChild(el(
            "span",
            "model-aux-score",
            tr(`Model retrieval score ${Number(row.model_score).toFixed(4)}`, `模型检索分数 ${Number(row.model_score).toFixed(4)}`),
          ));
          item.appendChild(modelMeta);
        }
        grid.appendChild(item);
      });
      evidence.appendChild(grid);
      if (known.truncated) {
        const more = el("div", "evidence-more");
        more.appendChild(el("span", "", tr(`Showing 20 of ${known.count} recorded associations for quick review.`, `为便于快速浏览，当前展示 ${known.count} 条已记录关联中的前 20 条。`)));
        if (known.source_record_url) {
          const sourceLink = externalLink(known.source_record_url, direction === "reaction_to_enzyme" ? tr("Open full Rhea record ↗", "在 Rhea 查看完整记录 ↗") : tr("Open full UniProt record ↗", "在 UniProt 查看完整记录 ↗"));
          sourceLink.classList.add("evidence-more-link");
          more.appendChild(sourceLink);
        }
        evidence.appendChild(more);
      }
    } else {
      const empty = el("div", "evidence-empty");
      empty.append(
        el("strong", "", knownLabel),
        el("p", "", tr("None found in the integrated database evidence sources.", "整合的数据库证据来源中暂未找到。")),
      );
      evidence.appendChild(empty);
    }
    card.appendChild(evidence);

    // Layer 2: model discovery. Every row here is intentionally unrecorded.
    if (!mode.knownOnly) {
      const discovery = el("section", "discovery-section");
      const discoveryHead = el("div", "discovery-head");
      const discoveryCopy = el("div");
      discoveryCopy.append(
        el("strong", "", `${discoveryLabel} · ${discoveryRows.length}`),
        el("small", "", tr("Neural retrieval ranking", "神经检索模型排序")),
      );
      discoveryHead.appendChild(discoveryCopy);
      discoveryHead.appendChild(el("span", "discovery-status", tr("Needs validation", "需实验验证")));
      discovery.appendChild(discoveryHead);

      if (discoveryRows.length) {
        const tableWrap = el("div", "table-wrap discovery-table-wrap");
        const table = document.createElement("table");
        const thead = document.createElement("thead");
        const hr = document.createElement("tr");
        [tr("Rank", "排名"), direction === "reaction_to_enzyme" ? tr("Enzyme", "候选酶") : tr("Reaction", "候选反应"), tr("Retrieval score", "检索分数")]
          .forEach((text) => hr.appendChild(el("th", "", text)));
        thead.appendChild(hr);
        const tbody = document.createElement("tbody");
        discoveryRows.forEach((row) => {
          const tableRow = document.createElement("tr");
          tableRow.appendChild(el("td", "rank-cell", String(row.rank)));
          const entity = el("td", "result-entity");
          const primary = el("div", "result-entity-primary");
          if (direction === "reaction_to_enzyme") {
            const link = externalLink(row.uniprot_url || "#", row.candidate_id);
            link.classList.add("entity-primary-link");
            primary.appendChild(link);
            entity.appendChild(primary);
            const meta = [row.name, row.species].filter(Boolean).join(" · ");
            if (meta) entity.appendChild(el("small", "", meta));
          } else {
            if (row.rhea_url) {
              const link = externalLink(row.rhea_url, row.candidate_id);
              link.classList.add("entity-primary-link");
              primary.appendChild(link);
            } else {
              primary.appendChild(el("strong", "entity-primary-text", row.candidate_id));
            }
            entity.appendChild(primary);
            const meta = row.name || [row.substrate_name, row.product_name].filter(Boolean).join(" → ");
            if (meta) entity.appendChild(el("small", "", meta));
          }
          if (Number(row.rank) <= 3) primary.appendChild(el("span", "priority-badge", tr("Priority", "优先查看")));
          tableRow.appendChild(entity);
          const score = el("td", "score-cell");
          score.appendChild(el("span", "score-number", Number(row.score || 0).toFixed(4)));
          const track = el("span", "score-track");
          const fill = el("i");
          fill.style.width = `${Math.max(2, Math.min(100, Number(row.score_fraction || 0) * 100))}%`;
          track.appendChild(fill);
          score.appendChild(track);
          tableRow.appendChild(score);
          tbody.appendChild(tableRow);
        });
        table.append(thead, tbody);
        tableWrap.appendChild(table);
        discovery.appendChild(tableWrap);
      } else {
        discovery.appendChild(el("p", "discovery-empty", tr("No unrecorded candidates were returned for this request.", "本次请求没有返回新关联候选。")));
      }
      discovery.appendChild(el("p", "score-note", localizedBackendText(result.score_note, "Retrieval scores compare model priority within this candidate list.", "检索分数用于比较当前候选列表中的模型优先级。")));
      card.appendChild(discovery);
    }

    const details = document.createElement("details");
    details.className = "result-route-details";
    const detailsSummary = document.createElement("summary");
    detailsSummary.textContent = tr("Technical details", "查看技术详情");
    details.appendChild(detailsSummary);
    const technical = el("div", "result-technical");
    const openRoute = el("button", "result-route-open");
    openRoute.type = "button";
    openRoute.append(
      el("strong", "", tr("View model execution trace", "查看本次模型路线")),
      el("span", "", tr("Open full flow ↗", "打开完整流程图 ↗")),
    );
    openRoute.addEventListener("click", () => openActualRouteDialog(result.route_view || {}));
    technical.appendChild(openRoute);
    technical.appendChild(el("code", "result-route-code", result.route_view?.route_id || result.ranking?.route_id || ""));
    const route = el("div", "inline-route");
    (result.route_view?.nodes || []).forEach((node, index) => {
      const item = el("div", `inline-route-node kind-${node.kind || "control"}`);
      const englishNodeTitle = node.id ? node.id.replaceAll("-", " ") : routeKindNames[node.kind] || "step";
      item.append(
        el("span", "", String(index + 1).padStart(2, "0")),
        el("strong", "", uiLanguage === "zh" ? (node.title || node.id || "步骤") : englishNodeTitle),
        el("small", "", node.metric || (uiLanguage === "zh" ? node.subtitle : routeKindNames[node.kind]) || ""),
      );
      route.appendChild(item);
      if (index < (result.route_view?.nodes || []).length - 1) route.appendChild(el("i", "route-arrow", "→"));
    });
    technical.appendChild(route);
    details.appendChild(technical);
    card.appendChild(details);
    content.appendChild(card);
    scrollConversation();
  }

  function normalizeRouteFlow(route, actual = false) {
    const source = actual ? (route.nodes || []) : (route.flow || []);
    if (source.length) return source;
    return (route.modules || []).map((id) => ({
      id,
      title: id,
      subtitle: "repository module",
      kind: "control",
      detail: tr("This step comes from a repository route definition.", "该步骤来自仓库中的路线定义。"),
    }));
  }

  function routeDialogBadges(route, actual) {
    const badges = [];
    if (actual) badges.push(tr("Actual run", "本次实际执行"));
    if (route.direction === "reaction_to_enzyme") badges.push(tr("Reaction → enzyme", "反应 → 酶"));
    if (route.direction === "enzyme_to_reaction") badges.push(tr("Enzyme → reaction", "酶 → 反应"));
    if (route.direction === "pathway_compatibility") badges.push(tr("Pathway · enzyme compatibility", "整条路径 · 多酶兼容性"));
    if (route.direction === "route_design") badges.push(tr("Route design & ranking", "候选路线 · 生成与排序"));
    if (route.scope && route.scope !== "any") badges.push(route.scope === "current" ? tr("Model-catalog entity", "库内实体") : route.scope === "external" ? tr("External entity", "外部实体") : route.scope);
    if (route.objective) badges.push(String(route.objective).replace("top", "Top "));
    if (route.availability) badges.push(route.availability);
    return badges;
  }

  function routeDialogIntro(route, flow, actual) {
    if (actual) return localizedBackendText(route.summary, "This trace reflects the route actually selected for the current request. Each module below is part of the production execution path.", "这条流程由本次输入和生产路由规则实际确定。下面逐步展示每个模块在做什么。");
    if (uiLanguage === "zh") {
      const chineseDescription = [route.description, route.use_case].find((text) => containsCjk(text));
      if (chineseDescription) return chineseDescription;
    }
    const direction = route.direction === "reaction_to_enzyme" ? tr("reaction-to-enzyme", "反应到酶") : route.direction === "enzyme_to_reaction" ? tr("enzyme-to-reaction", "酶到反应") : route.direction === "pathway_compatibility" ? tr("pathway compatibility", "整条路径兼容性") : route.direction === "route_design" ? tr("route design and ranking", "候选路线生成与排序") : tr("extended", "扩展");
    const scope = route.scope === "current" ? tr("model-catalog entity", "库内实体") : route.scope === "external" ? tr("external entity", "外部实体") : tr("multiple scopes", "多场景");
    const depth = route.objective ? ` · ${String(route.objective).replace("top", "Top ")}` : "";
    return tr(`This is a ${scope} ${direction} workflow${depth}, with ${flow.length} steps. The diagram follows the production execution order.`, `这是一条${scope}的${direction}流程${depth}，包含 ${flow.length} 个步骤。下面按执行顺序说明每一步处理什么信息，以及它如何影响最终候选。`);
  }

  function openRouteDialog(route, { actual = false } = {}) {
    if (!routeDialog || !route) return;
    const flow = normalizeRouteFlow(route, actual);
    routeDialogFlow.replaceChildren();
    routeDialogMeta.replaceChildren();
    routeDialogType.textContent = actual ? tr("ACTUAL RUN", "本次实际路线") : route.availability === "downstream" || route.availability === "batch" || route.availability === "specialist" ? tr("EXTENDED WORKFLOW", "扩展工作流") : tr("MODEL ROUTE", "模型路线");
    routeDialogTitle.textContent = uiLanguage === "zh" ? (route.title || route.label || route.key || "路线流程") : (route.key || route.route_id || tr("Execution flow", "路线流程"));
    routeDialogKey.textContent = route.route_id || route.key || "";
    routeDialogDescription.textContent = routeDialogIntro(route, flow, actual);
    routeDialogBadges(route, actual).forEach((text) => routeDialogMeta.appendChild(el("span", "", text)));

    flow.forEach((step, index) => {
      const row = el("article", `route-diagram-step kind-${step.kind || "control"}`);
      const rail = el("div", "route-diagram-rail");
      rail.appendChild(el("span", "route-diagram-number", String(index + 1).padStart(2, "0")));
      if (index < flow.length - 1) rail.appendChild(el("i", "route-diagram-connector"));
      const card = el("div", "route-diagram-card");
      const head = el("div", "route-diagram-step-head");
      const title = el("div");
      title.append(el("strong", "", uiLanguage === "zh" ? (step.title || step.id || `步骤 ${index + 1}`) : (step.id || `Step ${index + 1}`)), el("small", "", uiLanguage === "zh" ? (step.subtitle || routeKindNames[step.kind] || "流程步骤") : (routeKindNames[step.kind] || "Workflow step")));
      head.append(title, el("em", "", routeKindLabels[step.kind] || "STEP"));
      card.appendChild(head);
      if (actual && step.metric) {
        const metric = el("div", "route-diagram-metric");
        metric.append(el("small", "", tr("This run", "本次运行")), el("strong", "", step.metric));
        card.appendChild(metric);
      }
      card.appendChild(el("p", "", localizedBackendText(step.detail, "This step is defined by the production route in the repository.", "该步骤来自仓库中的生产路线定义。")));
      const foot = el("div", "route-diagram-step-foot");
      if (step.id) foot.appendChild(el("code", "", step.id));
      if (actual && step.note) foot.appendChild(el("span", "", localizedBackendText(step.note, "Runtime note", step.note)));
      if (foot.childNodes.length) card.appendChild(foot);
      row.append(rail, card);
      routeDialogFlow.appendChild(row);
    });

    if (typeof routeDialog.showModal === "function") routeDialog.showModal();
    else routeDialog.setAttribute("open", "");
  }

  function openActualRouteDialog(view = currentRouteView || {}) {
    if (!view || !(view.nodes || []).length) return;
    openRouteDialog(view, { actual: true });
  }

  function updateTechnicalDetails(result) {
    const view = result.route_view || {};
    const nodes = view.nodes || [];
    routeEmpty.classList.add("hidden");
    routeScroll.classList.remove("hidden");
    routeId.classList.remove("hidden");
    routeTimeline.replaceChildren();
    routeTitle.textContent = uiLanguage === "zh" ? (view.title || "已完成") : (view.route_id || tr("Execution complete", "已完成"));
    routeId.textContent = view.route_id || result.ranking?.route_id || "";
    routeStepCount.textContent = tr(`${nodes.length} modules`, `${nodes.length} 个模块`);
    currentRouteView = view;
    routeTitleButton.disabled = !nodes.length;

    nodes.forEach((node, index) => {
      const row = el("div", `route-step kind-${node.kind || "control"}`);
      const marker = el("span", "route-marker", String(index + 1).padStart(2, "0"));
      const copy = el("div", "route-step-copy");
      const top = el("div", "route-step-top");
      top.append(el("strong", "", uiLanguage === "zh" ? (node.title || node.id || "步骤") : (node.id || routeKindNames[node.kind] || "step").replaceAll("-", " ")), el("em", "route-kind", routeKindLabels[node.kind] || "STEP"));
      copy.append(top, el("small", "route-metric", node.metric || (uiLanguage === "zh" ? node.subtitle : routeKindNames[node.kind]) || ""));
      if (node.id) copy.appendChild(el("code", "route-module-id", node.id));
      if (node.note) copy.appendChild(el("p", "", localizedBackendText(node.note, "Runtime note available in the technical trace.", node.note)));
      if (node.detail) row.title = localizedBackendText(node.detail, "Production route step", node.detail);
      row.append(marker, copy);
      routeTimeline.appendChild(row);
    });

    const baseRoute = view.base_route_id || view.route_id || result.ranking?.route_id || "";
    const overlays = new Set(view.active_overlays || []);
    routeCatalog.querySelectorAll(".catalog-item").forEach((item) => {
      const key = item.dataset.routeKey || "";
      item.classList.toggle("active-route", Boolean(key && (key === baseRoute || overlays.has(key))));
    });
    routeScroll.scrollTop = 0;
  }

  function renderRouteCatalog(payload) {
    routeCatalog.replaceChildren();
    routeCatalogIndex.clear();
    const bases = payload.base_routes || [];
    const overlays = payload.overlays || [];
    const downstream = payload.downstream_workflows || [];
    [...bases, ...overlays, ...downstream].forEach((route) => {
      if (route.key) routeCatalogIndex.set(route.key, route);
    });
    routeCatalogCount.textContent = tr(`${bases.length + overlays.length} routes`, `${bases.length + overlays.length} 条路径`);

    const statNodes = routeCatalogStats?.querySelectorAll("span");
    if (statNodes?.length >= 3) {
      statNodes[0].querySelector("strong").textContent = String(bases.length);
      statNodes[1].querySelector("strong").textContent = String(overlays.length);
      statNodes[2].querySelector("strong").textContent = String(downstream.length);
    }

    const groups = [
      [tr("Reaction → enzyme", "反应 → 酶"), bases.filter((row) => row.direction === "reaction_to_enzyme"), "R2E"],
      [tr("Enzyme → reaction", "酶 → 反应"), bases.filter((row) => row.direction === "enzyme_to_reaction"), "E2R"],
      [tr("Overlays", "附加模块"), overlays, "OVERLAY"],
      [tr("Extended workflows", "扩展流程"), downstream, "WORKFLOW"],
    ];

    groups.forEach(([title, rows, badge]) => {
      if (!rows.length) return;
      const group = el("section", "catalog-group");
      const heading = el("div", "catalog-group-head");
      heading.append(el("strong", "", title), el("span", "", `${badge} · ${rows.length}`));
      group.appendChild(heading);
      rows.forEach((row) => {
        const item = el("div", "catalog-item");
        item.dataset.routeKey = row.key || "";
        const button = el("button", "catalog-route-button");
        button.type = "button";
        const itemHead = el("span", "catalog-item-head");
        itemHead.append(el("strong", "", uiLanguage === "zh" ? (row.label || row.title || row.key || "route") : (row.key || row.route_id || "route")));
        const flowCount = (row.flow || row.modules || []).length;
        if (flowCount) itemHead.appendChild(el("em", "", tr(`${flowCount} steps`, `${flowCount} 步`)));
        button.appendChild(itemHead);
        if (row.key) button.appendChild(el("code", "catalog-route-key", row.key));
        const path = row.modules?.length ? row.modules.join("  →  ") : uiLanguage === "zh" ? (row.description || "点击查看流程图") : (row.modules?.join(" → ") || tr("Open route diagram", "点击查看流程图"));
        button.appendChild(el("small", "catalog-module-path", path));
        button.appendChild(el("span", "catalog-open-hint", tr("View flow ↗", "查看流程图 ↗")));
        button.addEventListener("click", () => openRouteDialog(row));
        item.appendChild(button);
        group.appendChild(item);
      });
      routeCatalog.appendChild(group);
    });
  }



  function renderIntentChoice(resolution, displayText, effectiveText) {
    const { content } = messageShell("assistant");
    content.appendChild(el("p", "", localizedBackendText(resolution.summary, "Your request has more than one plausible interpretation. Please choose one.", "你的描述可能有两种理解，请选择。")));
    const box = el("div", "intent-choice-card");
    (resolution.intent_options || []).filter((x) => x.direction).forEach((option) => {
      const optionLabel = uiLanguage === "zh" ? option.label : (option.direction === "reaction_to_enzyme" ? "Reaction → enzyme" : option.direction === "enzyme_to_reaction" ? "Enzyme → reaction" : option.direction === "route_design" ? "Route design" : option.direction === "pathway_compatibility" ? "Pathway compatibility" : "Continue");
      const button = el("button", "secondary-button", optionLabel);
      button.type = "button";
      button.addEventListener("click", () => {
        setDirection(option.direction);
        directionHintOneShot = true;
        sendPrompt(displayText);
      });
      box.appendChild(button);
    });
    content.appendChild(box);
    scrollConversation();
  }

  async function sendPrompt(text) {
    text = String(text || "").trim();
    if (!text || busy) return;
    const run = activeRun || {
      session_id: sessionId(),
      run_id: newId("run"),
      card_id: "",
      card_title: "",
      prompt_template: "",
    };
    activeRun = run;
    activeVerification = null;
    const continued = Boolean(continuation && useContinuation);
    const effectiveText = continued ? `${continuation.originalText}\n${tr("Follow-up request:", "用户后续要求：")} ${text}` : text;
    // Starter-card directions are soft hints. Once a user edits a template, the
    // final text becomes authoritative and must be re-interpreted from auto; this
    // prevents a route/pathway template from trapping a later rewritten R2E/E2R
    // request. Explicit expert selectors and ambiguity-choice buttons have no card
    // template and therefore remain hard one-shot choices.
    const starterWasEdited = Boolean(run.card_id && run.prompt_template && text !== run.prompt_template);
    const effectiveHint = directionHintOneShot && starterWasEdited ? "auto" : directionHint;
    if (directionHintOneShot) {
      directionHintOneShot = false;
      setDirection("auto");
    }
    const conversationContext = continued ? {
      previous_direction: continuation?.direction || "",
      previous_result_mode: continuation?.resultMode || "",
      previous_association_policy: continuation?.associationPolicy || "",
      previous_route_id: continuation?.routeId || "",
      previous_target: continuation?.target || "",
      ui_language: uiLanguage,
    } : { ui_language: uiLanguage };
    input.value = "";
    addUserMessage(text, continued);
    setBusy(true);
    resetProcess();
    advanceProcess("understand");
    const activity = addActivity(tr("Understanding your experimental goal…", "正在理解你的实验目标…"));

    try {
      const resolution = await api("/api/agent/resolve", {
        text: effectiveText,
        direction_hint: effectiveHint,
        conversation_context: conversationContext,
        ui_language: uiLanguage,
        session_id: run.session_id,
        run_id: run.run_id,
        step_id: `step_${Date.now().toString(36)}`,
        card_id: run.card_id,
        card_title: run.card_title,
        prompt_template: run.prompt_template,
        prompt_source: run.card_id ? "shortcut_card" : "composer",
        edited_after_card_click: Boolean(run.card_id && text !== run.prompt_template),
      });
      updateTechnicalLanguage(resolution.llm_provenance);
      renderAgentExecution(resolution.agent_execution);
      if (resolution.direction === "ambiguous") {
        renderIntentChoice(resolution, text, effectiveText);
        activity.finish(tr("Task type needs confirmation", "需要确认任务类型"), tr("Waiting for your choice", "等待你的选择"));
        activeRun = null;
        return;
      }
      if (resolution.immediate_result) {
        const result = resolution.immediate_result;
        updateContextBeforeRun(resolution);
        activity.update(tr("Reading recorded database evidence…", "正在读取数据库已记录证据…"));
        advanceProcess("verify");
        advanceProcess("search");
        advanceProcess("result");
        renderResult(result, resolution.direction);
        updateTechnicalDetails(result);
        const knownCount = Number(result.known_associations?.count || 0);
        activity.finish(
          tr("Recorded evidence ready", "数据库证据已整理"),
          tr(`${knownCount} recorded association${knownCount === 1 ? "" : "s"}`, `${knownCount} 条已记录关联`),
        );
        completeProcess();
        const continuationMode = associationMode(result);
        const target = result.reaction?.rhea_id || result.protein?.name || result.protein?.id || taskTargetFromResolution(resolution);
        continuation = {
          originalText: effectiveText,
          direction: resolution.direction,
          resultMode: result.discovery_filter?.result_mode || "",
          associationPolicy: continuationMode?.policy || "",
          routeId: result.ranking?.route_id || result.route_view?.route_id || "",
          target,
        };
        useContinuation = true;
        activeRun = null;
        composerContext.classList.remove("hidden");
        contextSummary.textContent = resolution.summary || directionSummary(result, resolution.direction);
        const resultFacts = contextFacts.querySelectorAll("span strong");
        if (resultFacts[1]) resultFacts[1].textContent = target || "—";
        if (resultFacts[2]) resultFacts[2].textContent = continuationMode.label;
        return;
      }
      maybePrewarmProteinEncoder(resolution);
      const pathwayTask = resolution.direction === "pathway_compatibility";
      const routeDesignTask = resolution.direction === "route_design";
      activity.update(tr("Verifying database records…", "正在核对数据库记录…"), pathwayTask
        ? tr("Verifying Rhea steps and specified proteins", "逐步核对 Rhea 与已指定蛋白")
        : routeDesignTask ? tr("Verifying route source and target", "核对路线起点与目标产物")
        : resolution.direction === "reaction_to_enzyme" ? tr("Verifying reaction and protein records", "核对反应与相关蛋白")
          : resolution.protein_resolution?.mode === "protein_family" ? tr("Verifying protein-family scope", "核对蛋白家族范围")
            : tr("Verifying target protein", "核对目标蛋白"));
      renderVerification(resolution, text, effectiveText);
      const count = pathwayTask
        ? (resolution.pathway_resolution?.steps || []).reduce((n, step) => n + (step.reaction_resolution?.candidates?.length || 0) + (step.enzyme_resolution?.candidates?.length || 0), 0)
        : routeDesignTask
          ? (resolution.route_design_resolution?.source_candidates?.length || 0) + (resolution.route_design_resolution?.target_candidates?.length || 0)
          : resolution.direction === "reaction_to_enzyme"
            ? (resolution.reaction_resolution?.candidates?.length || 0) + (resolution.positive_enzyme_resolutions || []).reduce((n, group) => n + (group.candidates?.length || 0), 0)
            : resolution.protein_resolution?.mode === "protein_family"
              ? Number(resolution.protein_resolution?.family?.member_count || 0)
              : resolution.protein_resolution?.candidates?.length || 0;
      const familyTask = resolution.protein_resolution?.mode === "protein_family";
      activity.finish(
        pathwayTask ? tr("Pathway steps verified", "路径步骤已核对")
          : routeDesignTask ? tr("Route target verified", "路线目标已核对")
            : familyTask ? tr("Protein-family scope verified", "蛋白家族范围已核对")
              : tr("Verifiable database records found", "已找到可核对的数据库记录"),
        familyTask
          ? tr(`${count} family members in scope · awaiting confirmation`, `家族范围包含 ${count} 个候选成员 · 等待确认`)
          : tr(`${count} match${count === 1 ? "" : "es"} · awaiting confirmation`, `${count} 条匹配 · 等待确认`),
      );
    } catch (error) {
      activity.fail(tr("Database verification did not complete", "没有完成数据库核对"));
      addError(error.message, error.code === "deepseek_key_missing" ? tr("Natural-language features are temporarily unavailable", "自然语言功能暂不可用") : tr("No record could be verified", "没有找到可确认的记录"));
      activeRun = null;
      resetProcess();
    } finally {
      setBusy(false);
      input.focus({ preventScroll: true });
    }
  }

  function focusFirstPlaceholder() {
    const match = /【[^】]+】|\[[^\]]+\]/.exec(input.value);
    if (!match) {
      input.setSelectionRange(input.value.length, input.value.length);
      return;
    }
    input.setSelectionRange(match.index, match.index + match[0].length);
  }

  function wireStarterButtons(root = document) {
    root.querySelectorAll("[data-prompt]").forEach((button) => {
      button.addEventListener("click", () => {
        const promptTemplate = button.dataset.prompt || "";
        activeRun = {
          session_id: sessionId(),
          run_id: newId("run"),
          card_id: button.dataset.directionTemplate || "shortcut",
          card_title: button.querySelector("span")?.textContent?.trim() || "",
          prompt_template: promptTemplate,
        };
        useContinuation = false;
        composerContext.classList.add("hidden");
        const suggestedDirection = button.dataset.directionTemplate || "auto";
        setDirection(suggestedDirection);
        directionHintOneShot = suggestedDirection !== "auto";
        input.value = promptTemplate;
        input.focus({ preventScroll: true });
        focusFirstPlaceholder();
      });
    });
  }

  function wirePolicyPromptButtons(root = document) {
    root.querySelectorAll("[data-policy-prompt]").forEach((button) => {
      button.addEventListener("click", () => {
        const phrase = button.dataset.policyPrompt || "";
        if (!phrase) return;
        const existing = input.value.trim();
        input.value = existing ? `${existing} ${phrase}` : phrase;
        input.focus({ preventScroll: true });
        input.setSelectionRange(input.value.length, input.value.length);
      });
    });
  }

  function resetConversation() {
    if (busy) return;
    messages.replaceChildren(initialWelcome.cloneNode(true));
    messages.scrollTop = 0;
    wireStarterButtons(messages);
    wirePolicyPromptButtons(messages);
    continuation = null;
    useContinuation = true;
    directionHintOneShot = false;
    activeVerification = null;
    activeRun = null;
    composerContext.classList.add("hidden");
    input.value = "";
    setDirection("auto");
    setRouteMode("intelligent");
    contextTitle.textContent = tr("Not started", "还没有开始");
    contextSummary.textContent = tr("After you send a goal, this panel summarizes the target, verified records, and retrieval scope.", "发送实验目标后，这里会整理目标、已确认记录和当前筛选方式。");
    const facts = contextFacts.querySelectorAll("span strong");
    facts[0].textContent = "—";
    facts[1].textContent = "—";
    facts[2].textContent = tr("Known + discovery", "已知证据 + 新关联候选");
    resetProcess();
    routeTitle.textContent = tr("Not run yet", "尚未执行");
    currentRouteView = null;
    routeTitleButton.disabled = true;
    routeStepCount.textContent = tr("Waiting", "等待执行");
    routeTimeline.replaceChildren();
    routeScroll.classList.add("hidden");
    routeEmpty.classList.remove("hidden");
    routeId.textContent = "";
    routeId.classList.add("hidden");
    routeCatalog.querySelectorAll(".catalog-item.active-route").forEach((item) => item.classList.remove("active-route"));
    updateTechnicalLanguage(null);
    technicalDetails.open = false;
    input.focus({ preventScroll: true });
  }

  languageToggle?.addEventListener("click", () => i18n?.switchLanguage?.());
  railToggle?.addEventListener("click", () => {
    setRailCollapsed(!workspace?.classList.contains("rail-collapsed"));
  });
  setRailCollapsed(true);

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    sendPrompt(input.value);
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      if (!input.value.trim() && triggerActiveVerification()) return;
      sendPrompt(input.value);
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.shiftKey || event.isComposing || event.defaultPrevented) return;
    const target = event.target;
    if (target === input || target?.closest?.("a,button,textarea,input,[contenteditable=true]")) return;
    if (activeVerification) {
      event.preventDefault();
      triggerActiveVerification();
    }
  });

  document.querySelectorAll("[data-direction]").forEach((button) => {
    button.addEventListener("click", () => {
      directionHintOneShot = false;
      setDirection(button.dataset.direction || "auto");
    });
  });
  document.querySelectorAll("[data-route-mode]").forEach((button) => {
    button.addEventListener("click", () => setRouteMode(button.dataset.routeMode || "intelligent"));
  });
  $("dropContext").addEventListener("click", () => {
    useContinuation = false;
    composerContext.classList.add("hidden");
  });
  $("clearConversation").addEventListener("click", resetConversation);
  routeTitleButton.addEventListener("click", () => openActualRouteDialog());
  routeDialogClose.addEventListener("click", () => routeDialog.close());
  routeDialog.addEventListener("click", (event) => {
    if (event.target === routeDialog) routeDialog.close();
  });
  feedbackButton?.addEventListener("click", openFeedback);
  feedbackClose?.addEventListener("click", () => feedbackDialog.close());
  feedbackCancel?.addEventListener("click", () => feedbackDialog.close());
  feedbackDialog?.addEventListener("click", (event) => {
    if (event.target === feedbackDialog) feedbackDialog.close();
  });
  feedbackForm?.addEventListener("submit", submitFeedback);

  function feedbackContext() {
    const facts = contextFacts.querySelectorAll("span strong");
    return {
      direction: continuation?.direction || directionHint || "auto",
      target: facts[1]?.textContent || "",
      route_id: currentRouteView?.route_id || "",
      result_mode: facts[2]?.textContent || "",
      task_summary: contextSummary?.textContent || "",
    };
  }

  function openFeedback() {
    if (!feedbackDialog) return;
    feedbackStatus.classList.add("hidden");
    feedbackStatus.textContent = "";
    if (typeof feedbackDialog.showModal === "function") feedbackDialog.showModal();
    else feedbackDialog.setAttribute("open", "");
  }

  async function submitFeedback(event) {
    event.preventDefault();
    const rating = feedbackForm.querySelector('input[name="rating"]:checked')?.value || "";
    const message = feedbackMessage.value.trim();
    if (!rating && !message) {
      feedbackStatus.textContent = tr("Choose a rating or write a comment.", "请选择一个使用感受，或写下你的意见。");
      feedbackStatus.className = "feedback-status error";
      return;
    }
    feedbackSubmit.disabled = true;
    feedbackSubmit.textContent = tr("Submitting…", "提交中…");
    feedbackStatus.classList.add("hidden");
    try {
      await api("/api/feedback", {
        rating,
        category: feedbackCategory.value,
        message,
        contact: feedbackContact.value.trim(),
        context: feedbackContext(),
      });
      feedbackStatus.textContent = tr("Received. Thank you for the feedback.", "已收到，谢谢你的反馈。");
      feedbackStatus.className = "feedback-status success";
      feedbackForm.querySelectorAll('input[name="rating"]').forEach((node) => { node.checked = false; });
      feedbackMessage.value = "";
      feedbackContact.value = "";
      setTimeout(() => { if (feedbackDialog.open) feedbackDialog.close(); }, 900);
    } catch (error) {
      feedbackStatus.textContent = error.message || tr("Submission failed. Please try again later.", "提交失败，请稍后重试。");
      feedbackStatus.className = "feedback-status error";
    } finally {
      feedbackSubmit.disabled = false;
      feedbackSubmit.textContent = tr("Submit feedback", "提交反馈");
    }
  }

  async function refreshStatus() {
    try {
      const status = await api("/api/status");
      serviceSnapshot = status;
      serviceStatus.classList.toggle("ready", status.status === "ready");
      serviceStatus.querySelector("span").textContent = status.status === "ready" ? tr("System ready", "系统正常") : tr("Some features unavailable", "部分功能不可用");
      updateTechnicalLanguage(null);
    } catch (_) {
      serviceStatus.classList.remove("ready");
      serviceStatus.querySelector("span").textContent = tr("Connection error", "连接异常");
    }
  }

  wireStarterButtons();
  wirePolicyPromptButtons();
  resetProcess();
  refreshStatus();
  api("/api/routes").then(renderRouteCatalog).catch(() => { routeCatalogCount.textContent = tr("Unavailable", "暂不可用"); });
})();
