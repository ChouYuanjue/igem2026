(() => {
  const $ = (id) => document.getElementById(id);
  const messages = $("messages");
  const form = $("composerForm");
  const input = $("composerInput");
  const sendButton = $("sendButton");
  const serviceStatus = $("serviceStatus");
  const contextTitle = $("contextTitle");
  const contextSummary = $("contextSummary");
  const contextFacts = $("contextFacts");
  const technicalDetails = $("technicalDetails");
  const technicalAgentTrace = $("technicalAgentTrace");
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

  let busy = false;
  let activeVerification = null;
  let serviceSnapshot = null;
  let capabilitySnapshot = null;
  let currentRouteView = null;
  let activeRun = null;
  let latestUserText = "";
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

  function rotateSessionId() {
    const created = newId("sess");
    try { sessionStorage.setItem(`catalyst_finder_session_id_${uiLanguage}`, created); } catch (_) { /* session remains in-memory for this request */ }
    return created;
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

  function supersedeActiveVerification(reason = "new_user_message") {
    if (!activeVerification) return false;
    const pending = activeVerification;
    const previousRun = activeRun;
    pending.button.disabled = true;
    pending.button.textContent = tr("Superseded by later request", "已被后续请求替代");
    pending.card.dataset.superseded = "true";
    pending.card.setAttribute("aria-disabled", "true");
    recordClientEvent("verification_superseded", previousRun, { reason }, {
      direction: currentRouteView?.direction || "",
    });
    activeVerification = null;
    activeRun = null;
    return true;
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

  function addUserMessage(text) {
    const { content } = messageShell("user");
    const bubble = el("div", "user-bubble");
    bubble.appendChild(el("p", "", text));
    content.appendChild(bubble);
    scrollConversation();
  }

  function addAssistantResponse(text, { clarification = false } = {}) {
    const { content } = messageShell("assistant");
    const copy = el("div", clarification ? "assistant-copy clarification-copy markdown-body" : "assistant-copy conversational-copy markdown-body");
    if (window.CatalystMarkdown?.renderInto) window.CatalystMarkdown.renderInto(copy, String(text || ""));
    else copy.appendChild(el("p", "", String(text || "")));
    content.appendChild(copy);
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
    const { article, content } = messageShell("assistant");
    const card = el("div", "activity-card");
    const dot = el("span", "pulse-dot");
    const copy = el("div", "activity-copy");
    const strong = el("strong", "", title);
    const small = el("small", "", tr("Processing", "处理中"));
    copy.append(strong, small);
    card.append(dot, copy);
    content.appendChild(card);
    scrollConversation();
    return {
      update(nextTitle, detail = tr("Processing", "处理中")) {
        strong.textContent = nextTitle;
        small.textContent = detail;
      },
      finish() {
        article.remove();
      },
      fail(nextTitle = tr("Did not complete", "没有完成")) {
        strong.textContent = nextTitle;
        small.textContent = tr("Check the input and try again", "请检查输入后重试");
        dot.classList.add("failed");
      },
    };
  }

  const agentToolLabels = {
    resolve_reaction: ["Resolve reaction", "核对反应"],
    resolve_protein_scope: ["Resolve protein scope", "核对蛋白范围"],
    lookup_recorded_associations: ["Read recorded associations", "查询已记录关联"],
    lookup_recorded_protein_reactions: ["Read recorded reactions", "查询蛋白已记录反应"],
    list_protein_scope_members: ["List scope members", "列出范围成员"],
    resolve_compound: ["Resolve compound", "核对化合物"],
    inspect_verified_entity: ["Inspect verified entity", "查看实体详情"],
    compare_verified_entities: ["Compare verified entities", "比较已核对实体"],
    summarize_recorded_relations: ["Summarize recorded reactions", "汇总已记录反应"],
    broaden_protein_scope: ["Broaden annotation scope", "扩展注释范围"],
    prepare_candidate_retrieval: ["Prepare candidate retrieval", "准备候选检索"],
    prepare_route_design: ["Prepare route design", "准备路线设计"],
    prepare_pathway_compatibility: ["Prepare pathway evaluation", "准备路径评估"],
  };

  function agentStepLabel(step) {
    if (step.action_kind === "respond") return tr("Answer directly", "直接回答");
    if (step.action_kind === "ask_user") return tr("Ask a clarification", "追问关键信息");
    if (step.action_kind === "synthesize") return tr("Synthesize verified evidence", "综合已核对证据");
    if (step.action_kind === "final") return tr("Accept scientific result", "确认科学结果");
    if (step.action_kind === "turn_limit") return tr("Return verified result", "返回已核对结果");
    const labels = agentToolLabels[step.tool];
    return labels ? (uiLanguage === "zh" ? labels[1] : labels[0]) : (step.tool || tr("Scientific step", "科学处理步骤")).replaceAll("_", " ");
  }

  function renderAgentExecution(execution) {
    if (!technicalAgentTrace) return;
    const steps = Array.isArray(execution?.steps) ? execution.steps.filter((step) => step.tool || step.action_kind === "synthesize") : [];
    technicalAgentTrace.replaceChildren();
    if (!steps.length) {
      technicalAgentTrace.classList.add("hidden");
      return;
    }
    technicalAgentTrace.classList.remove("hidden");
    const details = el("details", "technical-tool-trace");
    const summary = el("summary");
    summary.append(
      el("span", "", tr("Tool steps", "工具步骤")),
      el("small", "", tr(`${steps.length} step${steps.length === 1 ? "" : "s"}`, `${steps.length} 步`)),
    );
    details.appendChild(summary);
    const list = el("div", "agent-trace-list");
    steps.forEach((step, index) => {
      const row = el("div", `agent-trace-step status-${step.status || "ok"}`);
      row.append(
        el("span", "agent-trace-index", String(index + 1).padStart(2, "0")),
        el("span", "agent-trace-step-label", agentStepLabel(step)),
        el("small", "agent-trace-status", step.status === "error" || step.status === "rejected"
          ? tr("Adjusted", "已调整") : tr("Done", "完成")),
      );
      list.appendChild(row);
    });
    details.appendChild(list);
    technicalAgentTrace.appendChild(details);
  }

  function localizedCapability(row, field) {
    return String(row?.[`${field}_${uiLanguage === "zh" ? "zh" : "en"}`] || row?.[`${field}_en`] || "");
  }

  function capabilityButton(example, group) {
    const button = el("button", "capability-action");
    button.type = "button";
    button.dataset.prompt = localizedCapability(example, "prompt");
    button.dataset.capabilityId = group.id || "capability";
    button.append(
      el("span", "", localizedCapability(example, "title")),
      el("small", "", localizedCapability(example, "description")),
    );
    return button;
  }

  function renderCapabilities(payload) {
    capabilitySnapshot = payload;
    const groups = Array.isArray(payload?.groups) ? payload.groups : [];
    const guideBody = $("capabilityGuideBody");
    const guideCount = $("capabilityGuideCount");
    if (!guideBody) return;
    guideBody.replaceChildren();
    const guideNote = String(payload?.interaction?.[`guide_note_${uiLanguage === "zh" ? "zh" : "en"}`] || payload?.interaction?.guide_note_en || "").trim();
    if (guideNote) guideBody.appendChild(el("p", "capability-use-note", guideNote));
    let exampleCount = 0;
    groups.forEach((group) => {
      const section = document.createElement("details");
      section.className = "capability-group";
      const summary = document.createElement("summary");
      const copy = el("div");
      const examples = Array.isArray(group.examples) ? group.examples : [];
      exampleCount += examples.length;
      copy.append(el("strong", "", localizedCapability(group, "title")), el("small", "", localizedCapability(group, "description")));
      summary.append(copy, el("span", "capability-group-count", String(examples.length)));
      const actions = el("div", "capability-actions");
      examples.forEach((example) => actions.appendChild(capabilityButton(example, group)));
      section.append(summary, actions);
      guideBody.appendChild(section);
    });
    if (guideCount) guideCount.textContent = tr(`${groups.length} areas · ${exampleCount} examples`, `${groups.length} 类能力 · ${exampleCount} 个示例`);
    wireStarterButtons(guideBody);
  }

  function appendContextualFollowUps(host, prompts) {
    const clean = [...new Set((Array.isArray(prompts) ? prompts : []).map((value) => String(value || "").trim()).filter(Boolean))].slice(0, 3);
    if (!host || !clean.length || !host.isConnected) return;
    const wrap = el("div", "result-follow-ups");
    wrap.appendChild(el("small", "result-follow-up-label", tr("You can ask next", "接着可以问")));
    const actions = el("div", "result-follow-up-actions");
    clean.forEach((prompt) => {
      const button = el("button", "result-follow-up");
      button.type = "button";
      button.dataset.prompt = prompt;
      button.dataset.capabilityId = "contextual_followup";
      button.appendChild(el("span", "", prompt));
      actions.appendChild(button);
    });
    wrap.appendChild(actions);
    host.appendChild(wrap);
    wireStarterButtons(wrap);
  }

  function compactFollowUpContext(result, direction = "") {
    const panels = Array.isArray(result?.source_panels) ? result.source_panels : [];
    const sourcePanels = panels.map((panel) => ({
      section: panel?.section || panel?.id || "",
      source: panel?.title || panel?.id || "",
      status: panel?.status || "",
      count: Number(panel?.count ?? panel?.items?.length ?? 0),
      items: (Array.isArray(panel?.items) ? panel.items : []).slice(0, 4).map((row) => ({
        id: row?.id || "",
        title: row?.title || "",
        name: row?.name || "",
        source: row?.source || "",
        method: row?.method || "",
        year: row?.year || "",
      })),
    }));
    const modelFrontier = Array.isArray(result?.model_lens?.frontier) ? result.model_lens.frontier : [];
    const candidates = Array.isArray(result?.candidates) ? result.candidates : [];
    const entities = Array.isArray(result?.entities) ? result.entities : [];
    const routes = Array.isArray(result?.routes) ? result.routes : [];
    const pathwaySteps = Array.isArray(result?.steps) ? result.steps : Array.isArray(result?.selected_steps) ? result.selected_steps : [];
    return {
      user_request: latestUserText,
      direction: direction || result?.direction || "",
      answer_mode: result?.answer_mode || "",
      workspace_kind: result?.workspace_kind || "",
      entity: result?.entity ? { kind: result.entity.kind || "", id: result.entity.id || "", name: result.entity.name || "" } : null,
      reaction: result?.reaction ? { id: result.reaction.rhea_id || result.reaction.id || "", name: result.reaction.equation || result.reaction.name || "" } : null,
      protein: result?.protein ? { id: result.protein.id || "", name: result.protein.name || "" } : null,
      selected_sections: Array.isArray(result?.selected_sections) ? result.selected_sections : [],
      recorded_association_count: Number(result?.known_associations?.count || 0),
      model_frontier: modelFrontier.slice(0, 5).map((row) => ({ id: row?.candidate_id || "", name: row?.name || row?.substrate_name || row?.product_name || "", score: row?.score })),
      candidates: candidates.slice(0, 5).map((row) => ({ id: row?.candidate_id || row?.id || "", name: row?.name || "", score: row?.score })),
      entities: entities.slice(0, 5).map((row) => ({ id: row?.id || "", name: row?.name || "", source: row?.source || "" })),
      source_panels: sourcePanels,
      routes: routes.slice(0, 4).map((row) => ({ id: row?.route_id || "", compounds: Array.isArray(row?.compound_names) ? row.compound_names.slice(0, 6) : [], score: row?.score })),
      pathway: {
        verdict: result?.verdict || result?.summary?.verdict || "",
        steps: pathwaySteps.slice(0, 6).map((row) => ({
          step: row?.step_index || "",
          reaction: row?.rhea_id || "",
          enzyme: row?.selected_enzyme?.candidate_id || row?.enzyme_id || "",
        })),
      },
    };
  }

  function requestContextualFollowUps(host, result, direction = "") {
    if (!host || !result) return;
    api("/api/followups", {
      session_id: sessionId(),
      result_context: compactFollowUpContext(result, direction),
    }).then((response) => {
      const prompts = (Array.isArray(response?.items) ? response.items : [])
        .map((row) => String(row?.prompt || "").trim())
        .filter(Boolean);
      appendContextualFollowUps(host, prompts);
    }).catch(() => { /* Follow-up suggestions are optional and must never block results. */ });
  }

  function externalLink(url, text) {
    const a = el("a", "external-link", text);
    a.href = url;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    return a;
  }

  const RESULT_PAGE_SIZE = 10;

  function publishVisiblePage(viewContext, rows, pageIndex) {
    if (!viewContext?.entityKind) return;
    const resolver = typeof viewContext.idOf === "function" ? viewContext.idOf : (row) => row?.id;
    const entityIds = [...new Set((Array.isArray(rows) ? rows : [])
      .map((row) => String(resolver(row) || "").trim())
      .filter(Boolean))];
    if (!entityIds.length) return;
    api("/api/session/view-context", {
      session_id: sessionId(),
      entity_kind: viewContext.entityKind,
      entity_ids: entityIds,
      page_index: Math.max(0, Number(pageIndex || 0)),
    }).catch(() => { /* View state is optional context and must never block result browsing. */ });
  }

  function paginateInto(viewport, rows, renderRow, { pageSize = RESULT_PAGE_SIZE, controlsHost = null, viewContext = null } = {}) {
    const items = Array.isArray(rows) ? rows : [];
    const host = controlsHost || viewport.parentElement;
    let page = 0;
    const totalPages = Math.max(1, Math.ceil(items.length / pageSize));
    const nav = el("nav", "result-pagination");
    nav.setAttribute("aria-label", tr("Result pages", "结果分页"));
    const previous = el("button", "pagination-button", tr("Previous", "上一页"));
    const next = el("button", "pagination-button", tr("Next", "下一页"));
    previous.type = "button";
    next.type = "button";
    const label = el("span", "pagination-label");
    nav.append(previous, label, next);

    function paint() {
      const start = page * pageSize;
      const visible = items.slice(start, start + pageSize);
      viewport.replaceChildren(...visible.map((row, localIndex) => renderRow(row, start + localIndex)).filter(Boolean));
      label.textContent = tr(`${page + 1} / ${totalPages}`, `第 ${page + 1} / ${totalPages} 页`);
      previous.disabled = page === 0;
      next.disabled = page >= totalPages - 1;
      nav.classList.toggle("hidden", totalPages <= 1);
      publishVisiblePage(viewContext, visible, page);
    }

    previous.addEventListener("click", () => {
      if (page <= 0) return;
      page -= 1;
      paint();
      viewport.scrollIntoView({ block: "nearest" });
    });
    next.addEventListener("click", () => {
      if (page >= totalPages - 1) return;
      page += 1;
      paint();
      viewport.scrollIntoView({ block: "nearest" });
    });
    if (host) host.appendChild(nav);
    paint();
    return { pageCount: totalPages };
  }

  function paginateRemoteInto(viewport, initialRows, renderRow, { query, pagination = {}, totalCount = 0, controlsHost = null, viewContext = null } = {}) {
    const host = controlsHost || viewport.parentElement;
    const pageSize = Math.max(1, Number(pagination.page_size || RESULT_PAGE_SIZE));
    const count = Math.max(Number(totalCount || 0), Array.isArray(initialRows) ? initialRows.length : 0);
    const totalPages = Math.max(1, Math.ceil(count / pageSize));
    const pages = [{ rows: Array.isArray(initialRows) ? initialRows : [], cursor: pagination.cursor || "*", nextCursor: pagination.next_cursor || "" }];
    let page = 0;
    let loading = false;
    const nav = el("nav", "result-pagination");
    nav.setAttribute("aria-label", tr("Result pages", "结果分页"));
    const previous = el("button", "pagination-button", tr("Previous", "上一页"));
    const next = el("button", "pagination-button", tr("Next", "下一页"));
    previous.type = "button";
    next.type = "button";
    const label = el("span", "pagination-label");
    nav.append(previous, label, next);

    function paint() {
      const current = pages[page] || { rows: [] };
      viewport.replaceChildren(...current.rows.map((row, localIndex) => renderRow(row, page * pageSize + localIndex)).filter(Boolean));
      label.textContent = tr(`${page + 1} / ${totalPages}`, `第 ${page + 1} / ${totalPages} 页`);
      previous.disabled = loading || page === 0;
      const hasKnownNext = Boolean(current.nextCursor) || Boolean(pages[page + 1]);
      next.disabled = loading || page >= totalPages - 1 || !hasKnownNext;
      nav.classList.toggle("hidden", totalPages <= 1);
      if (!loading) publishVisiblePage(viewContext, current.rows, page);
    }

    async function ensurePage(index) {
      if (pages[index]) return true;
      const prior = pages[index - 1];
      if (!prior?.nextCursor || !query) return false;
      loading = true;
      paint();
      try {
        const response = await api("/api/research/literature-page", {
          query,
          cursor: prior.nextCursor,
          page_size: pageSize,
          page_index: index,
          session_id: sessionId(),
        });
        pages[index] = {
          rows: Array.isArray(response?.items) ? response.items : [],
          cursor: response?.pagination?.cursor || prior.nextCursor,
          nextCursor: response?.pagination?.next_cursor || "",
        };
        return true;
      } catch (_) {
        return false;
      } finally {
        loading = false;
        paint();
      }
    }

    previous.addEventListener("click", () => {
      if (loading || page <= 0) return;
      page -= 1;
      paint();
      viewport.scrollIntoView({ block: "nearest" });
    });
    next.addEventListener("click", async () => {
      if (loading || page >= totalPages - 1) return;
      const target = page + 1;
      if (!await ensurePage(target)) return;
      page = target;
      paint();
      viewport.scrollIntoView({ block: "nearest" });
    });
    if (host) host.appendChild(nav);
    paint();
    return { pageCount: totalPages };
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

    const { content } = messageShell("assistant");
    const copy = el("div", "assistant-copy");
    copy.append(el("p", "", resolution.summary || tr("I found database records that can be verified.", "我找到了可核对的数据库记录。")));
    copy.append(el("p", "subtle", tr("Confirm the target record. The best match is shown first; open alternatives when needed.", "确认目标记录。首选匹配在前，需要时查看其他结果。")));
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
          psec.appendChild(el("p", "empty-inline", tr("No verified known-active protein record was matched for this description.", "这条描述暂未匹配到可核对的已知有效酶记录。")));
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
        ssec.appendChild(el("p", "pathway-auto-enzyme", tr(`Route search will use the ${rd.host || "E. coli"} iML1515 metabolite pool as the source set.`, `路线搜索以 ${rd.host || "E. coli"} 的 iML1515 代谢物池作为起点集合。`)));
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
      card.appendChild(el("p", "pathway-auto-enzyme", tr(`${policy} · up to ${rd.max_steps || 6} steps · return ${rd.route_count || 10}.`, `${policy} · 最多 ${rd.max_steps || 6} 步 · 返回 ${rd.route_count || 10} 条。`)));
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
          section.appendChild(el("p", "pathway-auto-enzyme", tr("This step has no fixed enzyme; candidates will be selected jointly with the other steps.", "这一步未指定酶，候选将与其他步骤联合选择。")));
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
      ? (() => {
        const dims = resolution.pathway_resolution?.evidence_dimensions || [];
        const labels = { ph: tr("pH", "pH"), temperature: tr("temperature", "温度"), cofactors: tr("cofactors", "辅因子"), localization: tr("localization", "定位"), cross_step_activity: tr("cross-step activity", "跨步活性") };
        return dims.length
          ? tr(`Confirm the pathway; joint selection will evaluate ${dims.map((d) => labels[d] || d).join(", ")}.`, `确认路径；联合选酶时只评估${dims.map((d) => labels[d] || d).join("、")}。`)
          : tr("Confirm the pathway; this turn will jointly select enzymes from model priorities only.", "确认路径；本轮只按模型优先级联合选酶。");
      })()
      : routeDesignTask
        ? (() => {
          const layers = resolution.route_design_resolution?.analysis_layers || [];
          const suffix = [
            layers.includes("thermodynamics") ? tr("thermodynamics", "热力学") : "",
            layers.includes("host_flux") ? tr("host flux", "宿主通量") : "",
          ].filter(Boolean).join(" · ");
          return suffix
            ? tr(`Confirm source and target; route search will also run ${suffix}.`, `确认起点和目标；本轮还会执行${suffix}分析。`)
            : tr("Confirm source and target; this turn will search and rank Rhea routes only.", "确认起点和目标；本轮只搜索并排序 Rhea 路线。");
        })()
        : resolution.protein_resolution?.mode === "protein_family"
          ? tr("Confirm the family scope to summarize recorded reactions. Sequence-level prediction uses a concrete member or sequence.", "确认家族范围后汇总已记录反应；序列级预测使用具体成员或序列。")
          : tr("Open Rhea / UniProt for source records. Press Enter to continue.", "可打开 Rhea / UniProt 查看原始记录，按 Enter 继续。")));
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
    if (busy || card.dataset.superseded === "true") return;
    const confirmationRun = activeRun;
    recordClientEvent("confirmation_clicked", confirmationRun, { direction: resolution.direction });
    const confirmationError = (message, title, code, details = {}) => {
      recordClientEvent("confirmation_validation_failed", confirmationRun, { code, ...details }, { direction: resolution.direction });
      addError(message, title);
    };
    let payload;
    let selectedTarget = "";

    if (resolution.direction === "route_design") {
      const rd = resolution.route_design_resolution || {};
      const sourceRadio = card.querySelector(".route-source-list .compound-option input:checked");
      const targetRadio = card.querySelector(".route-target-list .compound-option input:checked");
      if ((rd.source_candidates || []).length && !sourceRadio) { confirmationError(tr("Confirm the starting precursor first.", "请先确认起始前体。"), tr("Route origin still needs confirmation", "还需要确认路线起点"), "route_source_missing"); return; }
      if (!targetRadio) { confirmationError(tr("Confirm the target product first.", "请先确认目标产物。"), tr("Route target still needs confirmation", "还需要确认路线目标"), "route_target_missing"); return; }
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
          analysis_layers: rd.analysis_layers || [],
          user_text: effectiveText,
        },
      };
    } else if (resolution.direction === "pathway_compatibility") {
      const stepSections = Array.from(card.querySelectorAll(".pathway-step-section"));
      const steps = [];
      for (const [index, section] of stepSections.entries()) {
        const reactionRadio = section.querySelector(".pathway-reaction-list .reaction-option input:checked");
        if (!reactionRadio) { confirmationError(tr(`Confirm reaction step ${index + 1} first.`, `请先确认第 ${index + 1} 步反应。`), tr("Pathway still needs confirmation", "还需要确认路径"), "pathway_reaction_missing", { step_index: index + 1 }); return; }
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
          evidence_dimensions: resolution.pathway_resolution?.evidence_dimensions || [],
        },
      };
    } else if (resolution.direction === "reaction_to_enzyme") {
      const reactionRadio = card.querySelector(".reaction-option input:checked");
      if (!reactionRadio) { confirmationError(tr("Select the target reaction first.", "请先选择目标反应。"), tr("Reaction still needs confirmation", "还需要确认反应"), "reaction_selection_missing"); return; }
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
          confirmed_seed_ids: positiveIds,
          confirmed_seed_inputs: positiveSequenceInputs,
        },
      };
    } else {
      const protein = resolution.protein_resolution || {};
      if (protein.mode === "protein_family") {
        const family = protein.family || {};
        const familyId = family.family_id || protein.recommended_id || "";
        if (!familyId) { confirmationError(tr("Confirm the target protein family first.", "请先确认目标蛋白家族。"), tr("Protein family still needs confirmation", "还需要确认蛋白家族"), "protein_family_missing"); return; }
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
        if (!proteinRadio) { confirmationError(tr("Select the target enzyme first.", "请先选择目标酶。"), tr("Protein still needs confirmation", "还需要确认蛋白"), "protein_selection_missing", { candidate_count: card.querySelectorAll(".protein-option input").length }); return; }
        const enzymeSequence = proteinRadio.dataset.sequence || "";
        selectedTarget = proteinRadio.value;
        payload = {
          endpoint: "/api/rank-reactions",
          body: {
            protein_id: enzymeSequence ? "" : proteinRadio.value,
            enzyme_sequence: enzymeSequence,
            query_id: enzymeSequence ? proteinRadio.value : "",
            user_text: effectiveText,
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
    recordClientEvent("confirmation_selection", confirmationRun, {
      direction: resolution.direction,
      selected_target: selectedTarget,
      endpoint: payload.endpoint,
    });

    const facts = contextFacts.querySelectorAll("span strong");
    if (facts[1]) facts[1].textContent = selectedTarget;
    runButton.disabled = true;
    runButton.textContent = resolution.direction === "pathway_compatibility" ? tr("Evaluating jointly…", "正在联合评估…") : resolution.direction === "route_design" ? tr("Generating & ranking routes…", "正在生成并排序路线…") : tr("Running discovery…", "正在筛选…");
    activeVerification = null;
    card.querySelectorAll(".entity-list").forEach((group) => collapseSelectedGroup(group, "change"));
    setBusy(true);
    const activity = addActivity(tr("Running retrieval…", "正在筛选候选…"));

    try {
      const result = await api(payload.endpoint, payload.body);
      const pathwayTask = resolution.direction === "pathway_compatibility";
      const routeDesignTask = resolution.direction === "route_design";
      const resultCount = pathwayTask ? (result.steps?.length || 0) : routeDesignTask ? (result.routes?.length || 0) : (result.candidates?.length || 0);
      activity.update(pathwayTask ? tr("Assembling pathway evaluation…", "正在整理整条路径…") : routeDesignTask ? tr("Assembling candidate routes…", "正在整理候选路线…") : tr("Assembling evidence & discovery…", "正在整理结果…"), pathwayTask ? tr(`${resultCount} steps jointly evaluated`, `${resultCount} 个步骤已联合评估`) : routeDesignTask ? tr(`${resultCount} routes ranked`, `${resultCount} 条候选路线已排序`) : tr(`${resultCount} discovery candidates`, `${resultCount} 个新关联候选`));
        renderResult(result, resolution.direction);
      updateTechnicalDetails(result);
      activity.finish();
      recordClientEvent("confirmation_execution_succeeded", confirmationRun, { endpoint: payload.endpoint, selected_target: selectedTarget });
        runButton.textContent = pathwayTask ? tr("Evaluation complete", "评估完成") : routeDesignTask ? tr("Routes ready", "推荐完成") : tr("Retrieval complete", "筛选完成");
      runButton.disabled = true;
      const continuationMode = (!pathwayTask && !routeDesignTask) ? associationMode(result) : null;
      activeRun = null;
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
      recordClientEvent("confirmation_execution_failed", confirmationRun, { endpoint: payload?.endpoint || "", selected_target: selectedTarget, error_code: error.code || "request_failed", message: String(error.message || "").slice(0, 500) });
      activity.fail(tr("Retrieval did not complete", "筛选没有完成"));
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
      const dims = Array.isArray(result.evidence_dimensions) ? result.evidence_dimensions : [];
      return dims.length
        ? tr(`${count}-step pathway jointly evaluated across ${dims.length} requested evidence dimensions.`, `${count} 步路径已联合评估，本轮使用 ${dims.length} 个请求的证据维度。`)
        : tr(`${count}-step pathway jointly ranked from model priorities only.`, `${count} 步路径已按模型优先级完成联合选酶。`);
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
    technical.appendChild(el("code", "result-route-code", result.route_view?.route_id || "pathway-compatibility-v2"));
    details.appendChild(technical);
    card.appendChild(details);
  }

  function renderPathwayResult(result) {
    const { content } = messageShell("assistant");
    const steps = Array.isArray(result.steps) ? result.steps : [];
    const dimensions = Array.isArray(result.evidence_dimensions) ? result.evidence_dimensions : [];
    const shared = result.shared_conditions || {};
    const target = result.target_conditions || {};
    const dimensionLabels = {
      ph: tr("pH", "pH"),
      temperature: tr("temperature", "温度"),
      cofactors: tr("cofactors", "辅因子"),
      localization: tr("localization", "定位"),
      cross_step_activity: tr("cross-step activity", "跨步活性"),
    };

    const intro = el("div", "assistant-copy result-intro");
    intro.appendChild(el("p", "", result.verdict_label || tr("Pathway evaluation complete", "路径评估完成")));
    intro.appendChild(el("p", "subtle", dimensions.length
      ? tr(`Joint enzyme selection used model priorities plus ${dimensions.map((d) => dimensionLabels[d] || d).join(", ")}.`, `联合选酶使用模型优先级，并只加入${dimensions.map((d) => dimensionLabels[d] || d).join("、")}证据。`)
      : tr("This turn selected the enzyme combination from model priorities; condition evidence can be added as a follow-up.", "本轮按模型优先级联合选择酶组合；后续可继续补充条件证据。")));
    content.appendChild(intro);

    const card = el("div", "result-card pathway-result-card");
    const head = el("div", "result-head");
    const titleWrap = el("div");
    titleWrap.append(
      el("strong", "", tr("Pathway enzyme combination", "整条路径的酶组合")),
      el("small", "", dimensions.length
        ? tr("Model priorities · requested compatibility evidence", "模型优先级 · 请求的兼容性证据")
        : tr("Model-priority joint selection", "模型优先级联合选择")),
    );
    head.appendChild(titleWrap);
    card.appendChild(head);

    const chips = el("div", "result-chips");
    const chipTexts = [tr(`${steps.length} steps`, `${steps.length} 步`), pathwayModeLabel(result.execution_mode)];
    if (dimensions.length) chipTexts.push(tr(`${dimensions.length} evidence dimensions`, `${dimensions.length} 个证据维度`));
    else chipTexts.push(tr("Model only", "仅模型"));
    if (dimensions.includes("ph") && target.ph !== null && target.ph !== undefined && Number.isFinite(Number(target.ph))) chipTexts.push(tr(`Target pH ${Number(target.ph)}`, `目标 pH ${Number(target.ph)}`));
    if (dimensions.includes("temperature") && target.temperature_c !== null && target.temperature_c !== undefined && Number.isFinite(Number(target.temperature_c))) chipTexts.push(tr(`Target ${Number(target.temperature_c)} °C`, `目标 ${Number(target.temperature_c)} °C`));
    if (dimensions.includes("cofactors") && (target.cofactors || []).length) chipTexts.push(tr(`Target cofactors ${(target.cofactors || []).join(" / ")}`, `目标辅因子 ${(target.cofactors || []).join(" / ")}`));
    if (dimensions.includes("ph") && shared.ph_label) chipTexts.push(tr(`Shared pH ${shared.ph_label}`, `共同 pH ${shared.ph_label}`));
    if (dimensions.includes("temperature") && shared.temperature_label) chipTexts.push(tr(`Shared temperature ${shared.temperature_label}`, `共同温度 ${shared.temperature_label}`));
    if (dimensions.includes("cofactors") && (shared.cofactors || []).length) chipTexts.push(tr(`Shared cofactors ${(shared.cofactors || []).join(" / ")}`, `共同辅因子 ${(shared.cofactors || []).join(" / ")}`));
    chipTexts.forEach((text) => chips.appendChild(el("span", "", text)));
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
      link.classList.add("pathway-enzyme-link");
      title.appendChild(link);
      const badges = el("div", "pathway-step-badges");
      if (candidate.local_rank) badges.appendChild(el("span", "", tr(`Step rank #${candidate.local_rank}`, `单步 #${candidate.local_rank}`)));
      if (step.changed_for_pathway_compatibility) badges.appendChild(el("span", "changed", tr("Jointly reranked", "联合重排")));
      top.append(title, badges);
      body.appendChild(top);
      const meta = [candidate.name, candidate.species].filter(Boolean).join(" · ");
      if (meta) body.appendChild(el("p", "pathway-step-meta", meta));

      if (dimensions.length) {
        const evidence = el("div", "pathway-condition-chips");
        const ph = profile.ph_active || profile.ph_optimum;
        const temp = profile.temperature_active_c || profile.temperature_optimum_c;
        const phText = intervalLabel(ph);
        const tempText = intervalLabel(temp, " °C");
        if (dimensions.includes("ph") && phText) evidence.appendChild(el("span", "", `${profile.ph_active ? tr("pH range", "pH 范围") : tr("optimal pH", "最适 pH")} ${phText}`));
        if (dimensions.includes("temperature") && tempText) evidence.appendChild(el("span", "", `${profile.temperature_active_c ? tr("temperature range", "温度范围") : tr("optimal temperature", "最适温度")} ${tempText}`));
        if (dimensions.includes("cofactors") && (profile.cofactors || []).length) evidence.appendChild(el("span", "", tr(`Cofactors ${(profile.cofactors || []).slice(0, 3).join(" / ")}`, `辅因子 ${(profile.cofactors || []).slice(0, 3).join(" / ")}`)));
        if (dimensions.includes("localization") && (profile.locations || []).length) evidence.appendChild(el("span", "", tr(`Localization ${(profile.locations || []).slice(0, 2).join(" / ")}`, `定位 ${(profile.locations || []).slice(0, 2).join(" / ")}`)));
        if (!evidence.childElementCount) evidence.appendChild(el("span", "unknown", tr("No structured evidence for the requested dimensions", "所选维度暂无结构化注释")));
        body.appendChild(evidence);
      }
      row.append(marker, body);
      stepList.appendChild(row);
    });
    card.appendChild(stepList);

    if (dimensions.length) {
      const conflictBox = el("section", "pathway-conflict-box");
      const conflicts = Array.isArray(result.conflicts) ? result.conflicts : [];
      conflictBox.appendChild(el("strong", "", conflicts.length
        ? tr("Evidence to review", "需要关注的证据")
        : tr("No strong conflict in the requested dimensions", "所选维度未见明显冲突")));
      if (conflicts.length) {
        const list = el("div", "pathway-conflict-list");
        conflicts.slice(0, 8).forEach((item) => {
          const row = el("div", `pathway-conflict-item severity-${item.severity || "medium"}`);
          const englishDetail = ({
            ph: "Reported pH ranges differ across these steps.",
            temperature: "Reported temperature ranges differ across these steps.",
            target_ph: "A selected enzyme is distant from the requested pH.",
            target_temperature: "A selected enzyme is distant from the requested temperature.",
            cofactor_regulation: "A cofactor or metal annotation may interfere with another enzyme.",
            localization: "Subcellular-location annotations differ across steps.",
            cross_step_activity: "A selected enzyme is also associated with another pathway step.",
          })[item.type] || "The requested evidence dimensions contain a difference worth checking.";
          row.append(el("span", "", tr(`Steps ${(item.steps || []).join(" / ")}`, `步骤 ${(item.steps || []).join(" / ")}`)), el("p", "", uiLanguage === "zh" ? (item.detail || "存在需要核对的差异。") : englishDetail));
          list.appendChild(row);
        });
        conflictBox.appendChild(list);
      }
      card.appendChild(conflictBox);
    }

    const recommendations = Array.isArray(result.recommendations) ? result.recommendations : [];
    if (recommendations.length) {
      const rec = el("section", "pathway-conflict-box pathway-recommendations");
      rec.appendChild(el("strong", "", tr("Next checks", "下一步")));
      recommendations.slice(0, 5).forEach((text) => rec.appendChild(el("p", "", text)));
      card.appendChild(rec);
    }
    appendPathwayRouteDetails(card, result);
    requestContextualFollowUps(card, result, "pathway_compatibility");
    content.appendChild(card);
    scrollConversation();
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
    const analysisLayers = Array.isArray(result.analysis_layers) ? result.analysis_layers : [];
    const filtered = Number(feas.host_infeasible_filtered_count || 0);
    const thermoCount = Number(feas.thermo_complete_count || 0);
    const evidenceParts = [];
    if (analysisLayers.includes("thermodynamics")) evidenceParts.push(tr(`MDF completed for ${thermoCount}/${Number(feas.preliminary_route_count || routes.length)} preliminary routes`, `${thermoCount}/${Number(feas.preliminary_route_count || routes.length)} 条预候选完成 MDF`));
    if (analysisLayers.includes("host_flux")) evidenceParts.push(filtered
      ? tr(`iML1515 FBA removed ${filtered} zero-flux route(s)`, `iML1515 FBA 过滤 ${filtered} 条零通量路线`)
      : tr("iML1515 host-flux analysis completed", "iML1515 宿主通量分析已完成"));
    intro.append(el("p", "subtle", evidenceParts.length
      ? evidenceParts.join(" · ")
      : tr("No additional feasibility layer was requested for this route search.", "本轮没有额外执行热力学或宿主通量分析。")));
    content.appendChild(intro);

    const card = el("div", "result-card route-design-result-card");
    const head = el("div", "result-head");
    const titleWrap = el("div");
    const routeLayerLabel = [
      "Rhea",
      analysisLayers.includes("thermodynamics") ? "eQuilibrator MDF" : "",
      analysisLayers.includes("host_flux") ? "E. coli iML1515 FBA" : "",
    ].filter(Boolean).join(" · ");
    titleWrap.append(el("strong", "", tr("Candidate biosynthetic routes", "候选生物合成路线")), el("small", "", routeLayerLabel));
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
    card.appendChild(list);
    paginateInto(list, routes, (route) => {
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
        analysisLayers.includes("thermodynamics") ? (thermo.status === "complete" && Number.isFinite(mdf) ? `MDF ${mdf.toFixed(1)} kJ/mol` : tr("MDF unavailable", "MDF 未覆盖")) : null,
        analysisLayers.includes("host_flux") ? (hostFeasibility.status === "complete" && Number.isFinite(flux50) ? tr(`iML1515 route flux ${flux50.toFixed(2)} @ ≥50% growth`, `iML1515 路线通量 ${flux50.toFixed(2)} @≥50%生长`) : tr("iML1515 FBA unknown", "iML1515 FBA 未知")) : null,
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
        input.focus({ preventScroll: true });
        input.setSelectionRange(input.value.length, input.value.length);
      });
      item.appendChild(action);
      return item;
    }, { controlsHost: card });

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
      card.appendChild(el("p", "route-design-exploration-note", tr("iML1515 route flux reports stoichiometric pathway capacity; kinetics and fermentation yield are evaluated separately.", "iML1515 路线通量表示化学计量通量容量；动力学与发酵产量另行评估。")));
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
      section.appendChild(xlist);
      paginateInto(xlist, exploratory, (route) => {
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
        item.appendChild(el("p", "predicted-route-warning", uiLanguage === "zh" ? (route.evidence_note || "这条路线包含预测步骤，需要进一步验证。") : "This route contains predicted steps and requires validation."));
        return item;
      }, { controlsHost: section });
      card.appendChild(section);
    } else if (result.exploration_backend?.predicted_note) {
      card.appendChild(el("p", "route-design-exploration-note", localizedBackendText(result.exploration_backend.predicted_note, "No predicted routes are available for this request.", result.exploration_backend.predicted_note)));
    }
    card.appendChild(el("p", "score-note", uiLanguage === "zh" ? (result.score_note || "路线分数用于候选路线之间的相对排序。") : "Route scores provide relative priorities within the candidate set."));
    requestContextualFollowUps(card, result, "route_design");
    content.appendChild(card);
    scrollConversation();
  }

  function renderEntityListResult(result) {
    const { content } = messageShell("assistant");
    const entities = Array.isArray(result.entities) ? result.entities : [];
    const intro = el("div", "assistant-copy result-intro evidence-first-intro");
    intro.appendChild(el("p", "", result.title || tr("Verified entities", "已核对实体")));
    if (result.note) intro.appendChild(el("p", "subtle", result.note));
    content.appendChild(intro);

    const card = el("div", "result-card evidence-discovery-card");
    const head = el("div", "result-head evidence-discovery-head");
    const titleWrap = el("div");
    titleWrap.append(el("strong", "", result.title || tr("Verified entities", "已核对实体")));
    if (result.scope?.label) titleWrap.appendChild(el("small", "", result.scope.label));
    head.append(titleWrap, el("span", "evidence-chip", tr(`${entities.length} verified`, `${entities.length} 个已核对`)));
    card.appendChild(head);

    const grid = el("div", "evidence-grid");
    card.appendChild(grid);
    paginateInto(grid, entities, (row) => {
      const item = el("article", "evidence-card");
      const top = el("div", "evidence-card-top");
      const primary = row.url ? externalLink(row.url, row.id || row.name || tr("Entity", "实体")) : el("strong", "entity-primary-text", row.id || row.name || tr("Entity", "实体"));
      primary.classList?.add("evidence-primary-link");
      top.appendChild(primary);
      if (row.source) top.appendChild(el("span", "evidence-source project", String(row.source).replaceAll("_", " ")));
      item.appendChild(top);
      if (row.name && row.name !== row.id) item.appendChild(el("p", "evidence-meta", row.name));
      if (row.subtitle) item.appendChild(el("p", "evidence-meta", row.subtitle));
      if (row.abstract) item.appendChild(el("p", "literature-abstract", row.abstract));
      if (row.model_ready !== undefined) {
        item.appendChild(el("p", "evidence-meta", row.model_ready
          ? tr("Model candidate coverage", "当前模型候选库已覆盖")
          : tr("Outside the active model candidate universe", "当前模型候选库未覆盖")));
      }
      return item;
    }, { controlsHost: card });
    requestContextualFollowUps(card, result, "entity_list");
    content.appendChild(card);
    scrollConversation();
  }

  function renderEntityComparisonResult(result) {
    const { content } = messageShell("assistant");
    const entities = Array.isArray(result.entities) ? result.entities : [];
    const rows = Array.isArray(result.comparison_rows) ? result.comparison_rows : [];
    const intro = el("div", "assistant-copy result-intro evidence-first-intro");
    intro.appendChild(el("p", "", result.title || tr("Verified entity comparison", "已核对实体比较")));
    if (result.note) intro.appendChild(el("p", "subtle", result.note));
    content.appendChild(intro);

    const card = el("div", "result-card entity-comparison-card");
    const head = el("div", "result-head evidence-discovery-head");
    head.append(el("strong", "", result.title || tr("Verified entity comparison", "已核对实体比较")), el("span", "evidence-chip", tr(`${entities.length} compared`, `比较 ${entities.length} 个实体`)));
    card.appendChild(head);
    const scroll = el("div", "entity-comparison-scroll");
    const table = document.createElement("table");
    table.className = "entity-comparison-table";
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    headerRow.appendChild(el("th", "", tr("Field", "字段")));
    entities.forEach((entity) => {
      const th = document.createElement("th");
      const primary = entity.url ? externalLink(entity.url, entity.id || entity.name || tr("Entity", "实体")) : el("strong", "", entity.id || entity.name || tr("Entity", "实体"));
      th.appendChild(primary);
      if (entity.name && entity.name !== entity.id) th.appendChild(el("small", "", entity.name));
      headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);
    const tbody = document.createElement("tbody");
    rows.forEach((row) => {
      const trNode = document.createElement("tr");
      trNode.appendChild(el("th", "", row.label || row.key || tr("Field", "字段")));
      const values = Array.isArray(row.values) ? row.values : [];
      entities.forEach((_entity, index) => trNode.appendChild(el("td", "", values[index] ?? "—")));
      tbody.appendChild(trNode);
    });
    table.appendChild(tbody);
    scroll.appendChild(table);
    card.appendChild(scroll);
    requestContextualFollowUps(card, result, "entity_comparison");
    content.appendChild(card);
    scrollConversation();
  }

  function renderResearchWorkspace(result) {
    const { content } = messageShell("assistant");
    const entity = result.entity || {};
    const selected = Array.isArray(result.selected_sections) && result.selected_sections.length
      ? result.selected_sections
      : ["recorded_relations", "model"];
    const primary = selected.includes(result.primary_section) ? result.primary_section : selected[0];
    const known = result.known_associations && typeof result.known_associations === "object" ? result.known_associations : null;
    const model = result.model_lens && typeof result.model_lens === "object" ? result.model_lens : null;
    const panels = Array.isArray(result.source_panels) ? result.source_panels : [];
    const opportunities = Array.isArray(result.opportunities) ? result.opportunities : [];
    const modelOk = model?.status === "ok";
    const frontier = modelOk && Array.isArray(model.frontier) ? model.frontier : [];
    const recovery = modelOk && model.recorded_recovery && typeof model.recorded_recovery === "object" ? model.recorded_recovery : {};
    const modelDomain = modelOk && model.domain && typeof model.domain === "object" ? model.domain : {};
    const retrospectiveAudit = modelDomain.retrospective_audit || {};

    const card = el("div", "result-card research-workspace-card research-workspace-composable");
    const head = el("header", "research-workspace-head compact");
    const identity = el("div", "research-identity");
    const entityPrimary = entity.url
      ? externalLink(entity.url, entity.id || entity.name || tr("Entity", "实体"))
      : el("strong", "", entity.id || entity.name || tr("Entity", "实体"));
    identity.appendChild(entityPrimary);
    if (entity.name && entity.name !== entity.id) identity.appendChild(el("strong", "research-entity-name", entity.name));
    if (entity.subtitle) identity.appendChild(el("small", "", entity.subtitle));
    head.appendChild(identity);

    const metrics = el("div", "research-head-stats");
    if (selected.includes("recorded_relations") && known) {
      metrics.appendChild(el("span", "research-stat", tr(`${known.count || 0} recorded`, `已记录 ${known.count || 0}`)));
    }
    if (selected.includes("model")) {
      metrics.appendChild(el("span", `research-stat ${modelOk ? "model" : "muted"}`, modelOk
        ? tr(`${frontier.length} model frontier`, `模型前沿 ${frontier.length}`)
        : tr("Model unavailable", "模型暂不可用")));
    }
    if (selected.includes("literature")) {
      const literature = panels.find((row) => row?.section === "literature" || row?.id === "literature");
      const count = Number(literature?.count ?? literature?.items?.length ?? 0);
      metrics.appendChild(el("span", "research-stat", tr(`${count} papers`, `文献 ${count}`)));
    }
    if (selected.includes("structures")) {
      const structures = panels.find((row) => row?.section === "structures" || row?.id === "structures");
      metrics.appendChild(el("span", "research-stat", tr(`${structures?.items?.length || 0} structures`, `结构 ${structures?.items?.length || 0}`)));
    }
    if (metrics.childElementCount) head.appendChild(metrics);
    card.appendChild(head);

    function moduleShell(id, title, metric, tone = "") {
      const details = document.createElement("details");
      details.className = `research-module module-${id}${tone ? ` ${tone}` : ""}`;
      details.open = selected.length === 1 || id === primary;
      const summary = document.createElement("summary");
      const titleNode = el("span", "research-module-title", title);
      summary.appendChild(titleNode);
      if (metric) summary.appendChild(el("span", "research-module-metric", metric));
      details.appendChild(summary);
      const body = el("div", "research-module-body");
      details.appendChild(body);
      return { details, body };
    }

    function renderSourcePanel(panel) {
      const source = el("article", `research-source-card ${panel?.status === "ok" ? "available" : "unavailable"}`);
      const sourceTop = el("div", "research-source-head");
      const sourceTitle = panel?.url
        ? externalLink(panel.url, panel.title || panel.id || tr("Source", "资料源"))
        : el("strong", "", panel?.title || panel?.id || tr("Source", "资料源"));
      const state = panel?.status === "ok"
        ? tr("live", "实时")
        : panel?.status === "not_applicable" ? tr("not applicable", "不适用") : tr("unavailable", "暂不可用");
      sourceTop.append(sourceTitle, el("span", "source-live-state", state));
      source.appendChild(sourceTop);
      if (panel?.status !== "ok") {
        const note = panel?.note || tr("No result was returned for this module.", "这个模块本次没有返回可用结果。");
        source.appendChild(el("p", "research-source-empty", note));
        return source;
      }
      if (Array.isArray(panel.facts) && panel.facts.length) {
        const group = el("div", "research-source-group compact");
        const facts = el("div", "research-fact-grid");
        paginateInto(facts, panel.facts, (fact) => {
          const item = el("span", "research-fact");
          item.append(el("small", "", fact.label || tr("Field", "字段")), el("strong", "", String(fact.value ?? "—")));
          return item;
        }, { pageSize: 8, controlsHost: group });
        group.prepend(facts);
        source.appendChild(group);
      }
      if (Array.isArray(panel.participants) && panel.participants.length) {
        const group = el("div", "research-source-group");
        group.appendChild(el("strong", "", tr("Participants", "反应参与物")));
        const tags = el("div", "research-participant-tags");
        paginateInto(tags, panel.participants, (row) => {
          const label = [row.name, row.id].filter(Boolean).join(" · ");
          return row.url ? externalLink(row.url, label) : el("span", "", label);
        }, { pageSize: 10, controlsHost: group });
        group.insertBefore(tags, group.querySelector(".result-pagination"));
        source.appendChild(group);
      }
      if (panel.reaction_smiles) source.appendChild(el("code", "research-reaction-smiles", panel.reaction_smiles));
      if (Array.isArray(panel.catalytic_activities) && panel.catalytic_activities.length) {
        const group = el("div", "research-source-group");
        group.appendChild(el("strong", "", tr("Catalytic activity", "催化活性")));
        const rows = el("div", "research-source-list");
        paginateInto(rows, panel.catalytic_activities, (row) => el("p", "", [row.reaction, row.ec_number ? `EC ${row.ec_number}` : "", ...(row.rhea_ids || [])].filter(Boolean).join(" · ")), { pageSize: 8, controlsHost: group });
        group.insertBefore(rows, group.querySelector(".result-pagination"));
        source.appendChild(group);
      }
      if (Array.isArray(panel.cofactors) && panel.cofactors.length) {
        const group = el("div", "research-source-group compact");
        group.appendChild(el("strong", "", tr("Cofactors", "辅因子")));
        const rows = el("div", "research-source-list compact");
        paginateInto(rows, panel.cofactors, (value) => el("span", "research-inline-value", String(value)), { pageSize: 10, controlsHost: group });
        group.insertBefore(rows, group.querySelector(".result-pagination"));
        source.appendChild(group);
      }
      const annotations = panel.annotations && typeof panel.annotations === "object" ? panel.annotations : {};
      const annotationRows = Object.entries(annotations).flatMap(([label, values]) =>
        (Array.isArray(values) ? values : []).filter(Boolean).map((value) => ({ label, value }))
      );
      if (annotationRows.length) {
        const group = el("div", "research-source-group research-annotations");
        const rows = el("div", "research-annotation-pages");
        paginateInto(rows, annotationRows, (row) => {
          const item = el("div", "research-annotation-item");
          item.append(el("small", "", String(row.label).replaceAll("_", " ")), el("p", "", String(row.value)));
          return item;
        }, { pageSize: 8, controlsHost: group });
        group.insertBefore(rows, group.querySelector(".result-pagination"));
        source.appendChild(group);
      }
      const xrefItems = Array.isArray(panel.cross_reference_items) ? panel.cross_reference_items : [];
      if (xrefItems.length) {
        const group = el("div", "research-source-group compact");
        group.appendChild(el("strong", "", tr("Cross-references", "数据库交叉引用")));
        const tags = el("div", "research-xrefs");
        paginateInto(tags, xrefItems, (row) => {
          const label = `${row.database || tr("Source", "资料源")} · ${row.id || ""}`;
          return row.url ? externalLink(row.url, label) : el("span", "", label);
        }, { pageSize: 12, controlsHost: group });
        group.insertBefore(tags, group.querySelector(".result-pagination"));
        source.appendChild(group);
      }
      const officialProteins = Array.isArray(panel.official_uniprot_items) ? panel.official_uniprot_items : [];
      if (officialProteins.length) {
        const group = el("div", "research-source-group compact");
        group.appendChild(el("strong", "", tr("Rhea-linked proteins", "Rhea 关联蛋白")));
        const tags = el("div", "research-xrefs");
        paginateInto(tags, officialProteins, (row) => row.url ? externalLink(row.url, row.id) : el("span", "", row.id || tr("Protein", "蛋白")), { pageSize: 10, controlsHost: group });
        group.insertBefore(tags, group.querySelector(".result-pagination"));
        source.appendChild(group);
      }
      if (panel.id === "literature" && panel.curated_by) {
        source.appendChild(el("p", "research-source-curation", panel.curated_by === "keyword_fallback"
          ? tr("Broad literature search", "广泛文献检索")
          : tr(`References linked by ${panel.curated_by}`, `${panel.curated_by} 关联文献`)));
      }
      if (Array.isArray(panel.items) && panel.items.length) {
        const items = el("div", `research-source-items ${panel.id === "literature" ? "literature" : ""}`);
        const renderSourceItem = (row) => {
          const item = el("div", "research-source-item");
          const primaryLabel = row.title || row.name || row.id || tr("Record", "记录");
          item.appendChild(row.url ? externalLink(row.url, primaryLabel) : el("strong", "", primaryLabel));
          const structureMeta = panel.id === "structures"
            ? [row.source, row.method, Number.isFinite(Number(row.resolution_angstrom)) ? `${Number(row.resolution_angstrom).toFixed(2)} Å` : "", Number.isFinite(Number(row.global_plddt)) ? `pLDDT ${Number(row.global_plddt).toFixed(1)}` : "", row.released || row.created || ""].filter(Boolean).join(" · ")
            : "";
          const meta = panel.id === "literature"
            ? [row.authors, row.journal, row.year, Number.isFinite(Number(row.cited_by)) ? tr(`${row.cited_by} citations`, `被引 ${row.cited_by}`) : ""].filter(Boolean).join(" · ")
            : (structureMeta || [row.id, row.type, ...(row.member_entries || []).slice(0, 3)].filter(Boolean).join(" · "));
          if (meta) item.appendChild(el("small", "", meta));
          if (panel.id === "literature" && Array.isArray(row.publication_types) && row.publication_types.length) item.appendChild(el("small", "research-literature-type", row.publication_types.slice(0, 3).join(" · ")));
          if (panel.id === "literature" && Array.isArray(row.annotation_context) && row.annotation_context.length) item.appendChild(el("small", "research-literature-context", row.annotation_context.slice(0, 3).join(" · ")));
          return item;
        };
        const literatureViewContext = panel.id === "literature" ? {
          entityKind: "literature",
          idOf: (row) => {
            const pmid = String(row?.pmid || "").trim();
            if (pmid) return `MED:${pmid}`;
            const rawId = String(row?.id || row?.pmcid || "").trim();
            if (!rawId) return "";
            if (rawId.includes(":")) return rawId;
            const sourceId = String(row?.source || "").trim().toUpperCase();
            return sourceId === "MED" || sourceId === "PMC" ? `${sourceId}:${rawId}` : rawId;
          },
        } : null;
        const remoteLiterature = panel.id === "literature" && panel?.pagination?.mode === "remote" && panel.query;
        if (remoteLiterature) {
          paginateRemoteInto(items, panel.items, renderSourceItem, {
            query: panel.query,
            pagination: panel.pagination,
            totalCount: panel.count,
            viewContext: literatureViewContext,
          });
        } else {
          paginateInto(items, panel.items, renderSourceItem, { viewContext: literatureViewContext });
        }
        source.appendChild(items);
      }
      return source;
    }

    function appendSourceModule(sectionId, title) {
      const rows = panels.filter((row) => row?.section === sectionId || (sectionId === "literature" && row?.id === "literature") || (sectionId === "structures" && row?.id === "structures"));
      const ok = rows.filter((row) => row?.status === "ok").length;
      const metric = tr(`${ok}/${rows.length} available`, `${ok}/${rows.length} 可用`);
      const { details, body } = moduleShell(sectionId, title, metric);
      const grid = el("div", `research-source-grid ${rows.length === 1 ? "single" : ""}`);
      rows.forEach((panel) => grid.appendChild(renderSourcePanel(panel)));
      if (!rows.length) grid.appendChild(el("p", "research-empty", tr("No source result was returned.", "本次没有返回该模块的资料。")));
      body.appendChild(grid);
      card.appendChild(details);
    }

    selected.forEach((section) => {
      if (section === "annotations") {
        appendSourceModule("annotations", tr("Database annotations", "数据库注释"));
        return;
      }
      if (section === "structures") {
        appendSourceModule("structures", tr("Structures", "结构信息"));
        return;
      }
      if (section === "literature") {
        appendSourceModule("literature", tr("Literature", "关联文献"));
        return;
      }
      if (section === "recorded_relations") {
        const count = Number(known?.count || 0);
        const { details, body } = moduleShell("recorded_relations", tr("Recorded relationships", "已记录关系"), String(count));
        if (Array.isArray(known?.items) && known.items.length) {
          const grid = el("div", "research-known-grid");
          paginateInto(grid, known.items, (row) => {
            const item = el("article", "research-known-item");
            const url = result.workspace_kind === "protein" ? row.rhea_url : row.uniprot_url;
            item.appendChild(url ? externalLink(url, row.candidate_id) : el("strong", "", row.candidate_id || tr("Association", "关联")));
            const meta = result.workspace_kind === "protein"
              ? row.name || [row.substrate_name, row.product_name].filter(Boolean).join(" → ")
              : [row.name, row.species].filter(Boolean).join(" · ");
            if (meta) item.appendChild(el("small", "", meta));
            const provenance = (row.sources || row.evidence_sources || []).join(" · ") || row.source || "";
            if (provenance) item.appendChild(el("span", "research-provenance", provenance));
            return item;
          });
          body.appendChild(grid);
        } else {
          body.appendChild(el("p", "research-empty", tr("No recorded association was found in the current evidence layer.", "当前证据层没有可核对的已记录关联。")));
        }
        card.appendChild(details);
        return;
      }
      if (section === "model") {
        const metric = modelOk ? tr(`${frontier.length} frontier`, `前沿 ${frontier.length}`) : tr("unavailable", "暂不可用");
        const { details, body } = moduleShell("model", tr("Model lens", "模型视角"), metric, "model");
        if (!modelOk) {
          body.appendChild(el("p", "research-empty", tr("The model did not complete for this target.", "这个目标的模型视角本次没有完成。")));
          card.appendChild(details);
          return;
        }
        const domainLine = el("div", "research-model-domain-line");
        const domainLabel = uiLanguage === "zh" ? modelDomain.label_zh : modelDomain.label_en;
        if (domainLabel) domainLine.appendChild(el("span", `model-domain-chip status-${modelDomain.status || "unknown"}`, domainLabel));
        const seedCount = Number(model.seed_count || 0);
        if (seedCount) domainLine.appendChild(el("span", "research-module-inline-note", tr(`${seedCount} evidence anchors`, `${seedCount} 个证据锚点`)));
        if (domainLine.childElementCount) body.appendChild(domainLine);
        const interpretation = uiLanguage === "zh" ? modelDomain.interpretation_zh : modelDomain.interpretation_en;
        if (interpretation) body.appendChild(el("p", `model-domain-note domain-${modelDomain.status || "unknown"}`, interpretation));
        if (modelDomain.status === "project_aligned" && Number(retrospectiveAudit.queries || 0) > 0) {
          const audit = el("div", "model-retrospective-audit");
          audit.appendChild(el("strong", "", tr("Official-relation retrospective audit", "官方关系回顾性审计")));
          const medianRank = Number(retrospectiveAudit.median_best_rank_among_hits || 0);
          audit.appendChild(el("p", "", `Hit@5 ${Math.round(Number(retrospectiveAudit.hit_at_5 || 0) * 1000) / 10}% · Hit@10 ${Math.round(Number(retrospectiveAudit.hit_at_10 || 0) * 1000) / 10}% · Hit@20 ${Math.round(Number(retrospectiveAudit.hit_at_20 || 0) * 1000) / 10}%${medianRank ? ` · ${tr("median best rank", "命中中位最佳排名")} #${medianRank}` : ""} · n=${Number(retrospectiveAudit.queries || 0)}`));
          body.appendChild(audit);
        }
        if (Array.isArray(recovery.items) && recovery.items.length) {
          const recovered = el("div", "model-recovery-list");
          recovery.items.slice(0, 6).forEach((row) => recovered.appendChild(el("span", "", `${row.id} · #${row.rank}`)));
          body.appendChild(recovered);
        }
        if (frontier.length) {
          const frontierGrid = el("div", "research-frontier-grid");
          frontier.forEach((row, index) => {
            const item = el("article", "research-frontier-item");
            item.appendChild(el("span", "frontier-rank", String(index + 1).padStart(2, "0")));
            const copy = el("div", "");
            copy.appendChild(row.url ? externalLink(row.url, row.candidate_id) : el("strong", "", row.candidate_id || tr("Candidate", "候选")));
            const meta = row.name || [row.substrate_name, row.product_name].filter(Boolean).join(" → ") || row.species || "";
            if (meta) copy.appendChild(el("small", "", meta));
            item.append(copy, el("span", "frontier-score", Number(row.score || 0).toFixed(4)));
            frontierGrid.appendChild(item);
          });
          body.appendChild(frontierGrid);
        } else {
          body.appendChild(el("p", "research-empty", tr("No additional frontier association was returned in this Top list.", "当前 Top 列表没有额外的新关联候选。")));
        }
        card.appendChild(details);
        return;
      }
      if (section === "next_steps") {
        const { details, body } = moduleShell("next_steps", tr("Next steps", "下一步"), String(opportunities.length));
        if (opportunities.length) {
          const grid = el("div", "research-next-grid");
          opportunities.slice(0, 4).forEach((row) => {
            const item = el("article", `research-next-item priority-${row.priority || "medium"}`);
            item.append(el("strong", "", localizedBackendText(row.title, row.title || tr("Continue research", "继续研究"), row.title || tr("Continue research", "继续研究"))), el("p", "", row.reason || ""));
            grid.appendChild(item);
          });
          body.appendChild(grid);
        } else {
          body.appendChild(el("p", "research-empty", tr("No additional next-step suggestion was requested or produced.", "本轮没有生成额外的下一步建议。")));
        }
        card.appendChild(details);
      }
    });

    requestContextualFollowUps(card, result, "research_workspace");

    const routeNodes = Array.isArray(result.route_view?.nodes) ? result.route_view.nodes : [];
    if (routeNodes.length > 1) {
      const technical = document.createElement("details");
      technical.className = "result-route-details research-route-details";
      const summary = document.createElement("summary");
      summary.textContent = tr("Technical composition", "技术编排");
      technical.appendChild(summary);
      const route = el("div", "research-route-mini");
      routeNodes.forEach((node, index) => {
        const step = el("div", `research-route-mini-step kind-${node.kind || "control"}`);
        step.append(el("span", "", String(index + 1)), el("strong", "", node.title || node.id), el("small", "", node.metric || ""));
        route.appendChild(step);
      });
      technical.appendChild(route);
      card.appendChild(technical);
    }
    content.appendChild(card);
    scrollConversation();
  }

  function renderResult(result, direction) {
    if (result?.answer_mode === "research_workspace") {
      renderResearchWorkspace(result);
      return;
    }
    if (result?.answer_mode === "entity_comparison") {
      renderEntityComparisonResult(result);
      return;
    }
    if (result?.answer_mode === "entity_list") {
      renderEntityListResult(result);
      return;
    }
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
    titleWrap.append(el("strong", "", tr("Results", "结果")));
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
      evidence.appendChild(grid);
      paginateInto(grid, known.items || [], (row) => {
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
          item.appendChild(el("p", "evidence-meta", tr(
            `Recorded for ${row.family_support_count} of ${row.family_member_count} members in this family scope.`,
            `当前范围内 ${row.family_support_count}/${row.family_member_count} 个成员有这条记录。`,
          )));
        }
        if (row.model_score !== null && row.model_score !== undefined) {
          item.appendChild(el("span", "model-aux-score", tr(
            `Model retrieval score ${Number(row.model_score).toFixed(4)}`,
            `模型检索分数 ${Number(row.model_score).toFixed(4)}`,
          )));
        }
        return item;
      }, { controlsHost: evidence });
      if (known.truncated) {
        const more = el("div", "evidence-more");
        more.appendChild(el("span", "", tr(`Loaded ${Math.min((known.items || []).length, known.count)} of ${known.count} recorded associations.`, `已载入 ${Math.min((known.items || []).length, known.count)}/${known.count} 条已记录关联。`)));
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
        el("small", "", tr("Model ranking", "模型排序")),
      );
      discoveryHead.appendChild(discoveryCopy);
      discoveryHead.appendChild(el("span", "discovery-status", tr("For validation", "待验证")));
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
        table.append(thead, tbody);
        tableWrap.appendChild(table);
        discovery.appendChild(tableWrap);
        paginateInto(tbody, discoveryRows, (row) => {
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
          return tableRow;
        }, { controlsHost: tableWrap });
      } else {
        discovery.appendChild(el("p", "discovery-empty", tr("No unrecorded candidates were returned for this request.", "本次请求没有返回新关联候选。")));
      }
      discovery.appendChild(el("p", "score-note", localizedBackendText(result.score_note, "Retrieval scores compare model priority within this candidate list.", "检索分数用于比较当前候选列表中的模型优先级。")));
      card.appendChild(discovery);
    }

    requestContextualFollowUps(card, result, direction);

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



  async function sendPrompt(text) {
    text = String(text || "").trim();
    if (!text || busy) return;
    supersedeActiveVerification("new_user_message");
    const starterRun = activeRun?.card_id ? activeRun : null;
    const run = starterRun || {
      session_id: sessionId(),
      run_id: newId("run"),
      card_id: "",
      card_title: "",
      prompt_template: "",
    };
    activeRun = run;
    const effectiveText = text;
    latestUserText = effectiveText;
    input.value = "";
    addUserMessage(text);
    setBusy(true);
    const activity = addActivity(tr("Understanding your experimental goal…", "正在理解你的实验目标…"));

    try {
      const resolution = await api("/api/agent/resolve", {
        text: effectiveText,
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
      if (resolution.assistant_response) {
        addAssistantResponse(resolution.assistant_response, { clarification: resolution.response_type === "clarification" });
        if (!resolution.immediate_result) {
          if ((resolution.agent_execution?.steps || []).some((step) => step.tool || step.action_kind === "synthesize")) renderAgentExecution(resolution.agent_execution);
          activity.finish();
          activeRun = null;
          return;
        }
      }
      if (resolution.immediate_result) {
        const result = resolution.immediate_result;
        updateContextBeforeRun(resolution);
        activity.update(result.answer_mode === "research_workspace"
          ? tr("Assembling research evidence and model view…", "正在汇集资料、证据与模型视角…")
          : tr("Reading recorded database evidence…", "正在读取数据库已记录证据…"));
                    renderResult(result, resolution.direction);
        renderAgentExecution(resolution.agent_execution);
        updateTechnicalDetails(result);
        const entityListMode = result.answer_mode === "entity_list";
        const entityComparisonMode = result.answer_mode === "entity_comparison";
        const researchWorkspaceMode = result.answer_mode === "research_workspace";
        const entityMode = entityListMode || entityComparisonMode;
        const knownCount = Number(result.known_associations?.count || 0);
        const entityCount = Array.isArray(result.entities) ? result.entities.length : 0;
        activity.finish();
            const continuationMode = researchWorkspaceMode
          ? { policy: "research_workspace", label: tr("Research evidence + model", "资料证据 + 模型视角") }
          : entityComparisonMode
            ? { policy: "entity_comparison", label: tr("Verified comparison", "已核对实体比较") }
            : entityListMode
              ? { policy: "entity_list", label: tr("Verified entities", "已核对实体") }
              : associationMode(result);
        const target = result.entity?.name || result.entity?.id || result.reaction?.rhea_id || result.protein?.name || result.protein?.id || result.scope?.label || result.entities?.[0]?.name || result.entities?.[0]?.id || taskTargetFromResolution(resolution);
        activeRun = null;
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
      renderAgentExecution(resolution.agent_execution);
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
      activity.finish();
    } catch (error) {
      activity.fail(tr("Database verification did not complete", "没有完成数据库核对"));
      addError(error.message, error.code === "deepseek_key_missing" ? tr("Natural-language features are temporarily unavailable", "自然语言功能暂不可用") : tr("No record could be verified", "没有找到可确认的记录"));
      activeRun = null;
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
          card_id: button.dataset.capabilityId || "capability_example",
          card_title: button.querySelector("span")?.textContent?.trim() || "",
          prompt_template: promptTemplate,
        };
        input.value = promptTemplate;
        input.focus({ preventScroll: true });
        focusFirstPlaceholder();
      });
    });
  }

  function resetConversation() {
    if (busy) return;
    supersedeActiveVerification("conversation_reset");
    rotateSessionId();
    messages.replaceChildren(initialWelcome.cloneNode(true));
    messages.scrollTop = 0;
    wireStarterButtons(messages);
    if (capabilitySnapshot) renderCapabilities(capabilitySnapshot);
    activeVerification = null;
    activeRun = null;
    input.value = "";
    contextTitle.textContent = tr("Not started", "还没有开始");
    contextSummary.textContent = tr("The current target and result scope will appear here.", "这里显示当前目标与结果范围。");
    const facts = contextFacts.querySelectorAll("span strong");
    facts[0].textContent = "—";
    facts[1].textContent = "—";
    facts[2].textContent = "—";
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
    technicalAgentTrace?.replaceChildren();
    technicalAgentTrace?.classList.add("hidden");
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
      direction: currentRouteView?.direction || "conversation",
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
  refreshStatus();
  api("/api/capabilities").then(renderCapabilities).catch(() => {
    const guideCount = $("capabilityGuideCount");
    const guideBody = $("capabilityGuideBody");
    if (guideCount) guideCount.textContent = tr("Unavailable", "暂不可用");
    if (guideBody) guideBody.replaceChildren(el("p", "capability-loading", tr("Capability list is temporarily unavailable.", "能力列表暂时不可用。")));
  });
  api("/api/routes").then(renderRouteCatalog).catch(() => { routeCatalogCount.textContent = tr("Unavailable", "暂不可用"); });
})();
