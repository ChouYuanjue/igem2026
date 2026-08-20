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

  let directionHint = "auto";
  let directionHintOneShot = false;
  let routeMode = "intelligent";
  let busy = false;
  let continuation = null;
  let useContinuation = true;
  let activeVerification = null;
  let serviceSnapshot = null;
  let currentRouteView = null;
  const routeCatalogIndex = new Map();
  const initialWelcome = messages.firstElementChild.cloneNode(true);

  function setRailCollapsed(collapsed) {
    const isCollapsed = Boolean(collapsed);
    workspace?.classList.toggle("rail-collapsed", isCollapsed);
    if (runRail) runRail.setAttribute("aria-hidden", isCollapsed ? "true" : "false");
    if (railToggle) {
      const actionLabel = isCollapsed ? "展开任务侧栏" : "收起任务侧栏";
      railToggle.setAttribute("aria-expanded", isCollapsed ? "false" : "true");
      railToggle.setAttribute("aria-label", isCollapsed ? "展开本次任务侧栏" : "收起本次任务侧栏");
      railToggle.dataset.tooltip = actionLabel;
      railToggle.dataset.mobileLabel = actionLabel;
    }
  }

  const directionLabels = {
    reaction_to_enzyme: "寻找候选酶",
    enzyme_to_reaction: "预测可能反应",
    route_design: "推荐并排序路线",
    pathway_compatibility: "评估整条路径",
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
    input: "输入与核对",
    decision: "条件判断",
    encode: "特征表示",
    universe: "候选空间",
    filter: "候选过滤",
    router: "路线选择",
    model: "模型计算",
    seed: "已知证据扩展",
    fusion: "多路结果融合",
    novelty: "新关联过滤",
    rescue: "补充候选",
    rank: "候选排序",
    trust: "证据解释",
    output: "结果输出",
    control: "流程步骤",
  };

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function api(path, payload) {
    return fetch(path, {
      method: payload === undefined ? "GET" : "POST",
      headers: payload === undefined ? {} : { "Content-Type": "application/json" },
      body: payload === undefined ? undefined : JSON.stringify(payload),
    }).then(async (response) => {
      let data = null;
      try { data = await response.json(); } catch (_) { /* no-op */ }
      if (!response.ok) {
        const error = new Error(data?.error?.message || `请求失败 (${response.status})`);
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
    meta.append(el("strong", "", type === "user" ? "你" : "Catalyst Finder"), el("span", "", "刚刚"));
    content.appendChild(meta);
    article.appendChild(content);
    messages.appendChild(article);
    return { article, content };
  }

  function addUserMessage(text, continued) {
    const { content } = messageShell("user");
    const bubble = el("div", "user-bubble");
    bubble.appendChild(el("p", "", text));
    if (continued) bubble.appendChild(el("span", "context-tag", "继续上一轮"));
    content.appendChild(bubble);
    scrollConversation();
  }

  function addError(message, title = "这一步没有完成") {
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
    const small = el("small", "", "正在处理");
    copy.append(strong, small);
    card.append(dot, copy);
    content.appendChild(card);
    scrollConversation();
    return {
      update(nextTitle, detail = "正在处理") {
        strong.textContent = nextTitle;
        small.textContent = detail;
      },
      finish(nextTitle, detail = "已完成") {
        strong.textContent = nextTitle;
        small.textContent = detail;
        dot.classList.add("done");
      },
      fail(nextTitle = "没有完成") {
        strong.textContent = nextTitle;
        small.textContent = "请检查输入后重试";
        dot.classList.add("failed");
      },
    };
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
    if (candidate.source === "model_catalog") return "候选库内";
    if (candidate.model_ready) return "UniProt · 可直接筛选";
    return "UniProt · 外部记录";
  }

  function updateGroupToggle(group, mode = "change") {
    const toggle = group?._toggleButton;
    if (!toggle) return;
    const count = Number(group.dataset.optionCount || 0);
    if (mode === "expanded") toggle.textContent = "收起其他结果";
    else if (mode === "initial") toggle.textContent = `查看其他 ${Math.max(0, count - 1)} 个结果`;
    else toggle.textContent = "更改";
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
    const toggle = el("button", "selection-change", `查看其他 ${options.length - 1} 个结果`);
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
    const dot = el("span", "option-radio");
    const main = el("span", "entity-main");
    const top = el("span", "entity-top");
    top.append(el("strong", "", candidate.name || candidate.id), el("em", "", sourceBadge(candidate)));
    const idline = el("span", "entity-idline", candidate.accession ? `${candidate.id} · ${candidate.accession}` : candidate.id);
    const meta = [candidate.organism, candidate.gene_names?.length ? candidate.gene_names.join(", ") : null, candidate.length ? `${candidate.length} aa` : null]
      .filter(Boolean).join(" · ");
    main.append(top, idline, el("small", "", meta || "蛋白记录"));
    label.append(radio, dot, main, externalLink(candidate.url, "UniProt ↗"));
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
    radio.checked = checked;
    const dot = el("span", "option-radio");
    const main = el("span", "entity-main");
    const top = el("span", "entity-top");
    top.append(el("strong", "", candidate.rhea_id), el("em", "", candidate.model_ready ? "可直接筛选" : "外部反应"));
    main.append(top, el("span", "reaction-equation", candidate.equation || ""));
    const meta = [];
    if (candidate.enzyme_count !== null && candidate.enzyme_count !== undefined) meta.push(`Rhea 已关联 ${candidate.enzyme_count} 个酶记录`);
    if (candidate.orientation === "reverse") meta.push("将按反向反应处理");
    main.appendChild(el("small", "", meta.join(" · ") || "已由 Rhea 核对"));
    label.append(radio, dot, main, externalLink(candidate.url, "Rhea ↗"));
    bindStableEntitySelection(label, radio);
    return label;
  }

  function compoundOption(candidate, name, checked, roleLabel = "化合物") {
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
    const meta = candidate.smiles ? `结构已与 Rhea/ChEBI 参与物索引核对 · ${candidate.smiles.slice(0, 86)}${candidate.smiles.length > 86 ? "…" : ""}` : "已与 Rhea 参与物索引核对";
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
      return resolution.reaction_resolution?.recommended_id || "待确认";
    }
    if (resolution.direction === "route_design") {
      const rd = resolution.route_design_resolution || {};
      return rd.recommended_target_id || rd.target_terms?.[0] || "待确认目标";
    }
    if (resolution.direction === "pathway_compatibility") {
      const count = resolution.pathway_resolution?.steps?.length || 0;
      return count ? `${count} 步路径` : "待确认路径";
    }
    return resolution.protein_resolution?.recommended_id || "待确认";
  }

  function updateContextBeforeRun(resolution) {
    const direction = resolution.direction;
    const taskLabel = directionLabels[direction] || "实验筛选";
    contextTitle.textContent = taskLabel;
    contextSummary.textContent = resolution.summary || "已理解目标，等待你确认数据库记录。";
    const facts = contextFacts.querySelectorAll("span");
    facts[0].querySelector("strong").textContent = taskLabel;
    facts[1].querySelector("strong").textContent = taskTargetFromResolution(resolution);
    facts[2].querySelector("strong").textContent = direction === "pathway_compatibility" ? "联合选择" : direction === "route_design" ? "路线排序" : "默认包含已知";
  }

  function updateTechnicalLanguage(provenance) {
    if (provenance?.live_verified) {
      techLanguageModel.textContent = `${provenance.provider || "DeepSeek"} · ${provenance.model || "API"} · 已验证调用`;
      return;
    }
    if (serviceSnapshot?.deepseek?.live_verified) {
      techLanguageModel.textContent = `${serviceSnapshot.deepseek.provider || "DeepSeek"} · ${serviceSnapshot.deepseek.model || "API"} · 已验证调用`;
      return;
    }
    if (serviceSnapshot?.deepseek_configured) {
      techLanguageModel.textContent = `${serviceSnapshot.deepseek_model || "DeepSeek"} · 已配置`;
      return;
    }
    techLanguageModel.textContent = "未配置";
  }

  function renderVerification(resolution, displayText, effectiveText) {
    updateContextBeforeRun(resolution);
    updateTechnicalLanguage(resolution.llm_provenance);
    advanceProcess("verify");

    const { content } = messageShell("assistant");
    const copy = el("div", "assistant-copy");
    copy.append(el("p", "", resolution.summary || "我找到了可核对的数据库记录。"));
    copy.append(el("p", "subtle", "请确认与实验目标一致的记录。默认只显示最可能的匹配，需要时可以展开其他结果。"));
    content.appendChild(copy);

    const card = el("div", "verification-card");
    const cardHead = el("div", "tool-card-head");
    cardHead.append(el("span", "tool-icon", "✓"), el("div", "", ""));
    cardHead.querySelector("div").append(el("strong", "", "请确认数据库记录"), el("small", "", "确认后再开始筛选"));
    card.appendChild(cardHead);

    if (resolution.direction === "reaction_to_enzyme") {
      const reaction = resolution.reaction_resolution;
      const rsec = verificationSection("目标反应", reaction?.interpreted_reaction || "Rhea 匹配结果");
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
        const psec = verificationSection(`已知有效酶${resolution.positive_enzyme_resolutions.length > 1 ? ` ${groupIndex + 1}` : ""}`, group.mention || "蛋白匹配结果");
        const plist = el("div", "entity-list");
        const pname = `positive-${groupIndex}-${Math.random().toString(36).slice(2)}`;
        (group.candidates || []).forEach((candidate, index) => {
          const checked = candidate.id === group.recommended_id || (!group.recommended_id && index === 0);
          plist.appendChild(proteinOption(candidate, pname, checked));
        });
        if (!(group.candidates || []).length) {
          psec.appendChild(el("p", "empty-inline", "没有找到足够匹配的蛋白记录，这条描述不会作为已知有效酶使用。"));
        } else {
          psec.appendChild(plist);
          prepareCollapsibleGroup(psec, plist);
        }
        card.appendChild(psec);
      });
    } else if (resolution.direction === "route_design") {
      const rd = resolution.route_design_resolution || {};
      if ((rd.source_candidates || []).length) {
        const ssec = verificationSection("起始前体", (rd.source_terms || []).join(" / ") || "Rhea / ChEBI 匹配结果");
        const slist = el("div", "entity-list route-source-list");
        const sname = `route-source-${Math.random().toString(36).slice(2)}`;
        (rd.source_candidates || []).forEach((candidate, index) => {
          const checked = candidate.chebi_id === rd.recommended_source_id || (!rd.recommended_source_id && index === 0);
          slist.appendChild(compoundOption(candidate, sname, checked, "起始前体"));
        });
        ssec.appendChild(slist);
        card.appendChild(ssec);
        prepareCollapsibleGroup(ssec, slist);
      } else if (rd.host_pool_supported) {
        const ssec = verificationSection("路线起点", rd.host || "E. coli");
        ssec.appendChild(el("p", "pathway-auto-enzyme", `将从 ${rd.host || "E. coli"} 的 iML1515 代谢物池中寻找可达目标的候选路线，不会凭空指定前体。`));
        card.appendChild(ssec);
      }
      const tsec = verificationSection("目标产物", (rd.target_terms || []).join(" / ") || "Rhea / ChEBI 匹配结果");
      const tlist = el("div", "entity-list route-target-list");
      const tname = `route-target-${Math.random().toString(36).slice(2)}`;
      (rd.target_candidates || []).forEach((candidate, index) => {
        const checked = candidate.chebi_id === rd.recommended_target_id || (!rd.recommended_target_id && index === 0);
        tlist.appendChild(compoundOption(candidate, tname, checked, "目标产物"));
      });
      tsec.appendChild(tlist);
      card.appendChild(tsec);
      prepareCollapsibleGroup(tsec, tlist);
      const policy = ({ short: "优先短路线", enzyme_available: "优先酶可获得性", project_covered: "优先项目模型覆盖", thermodynamic: "优先热力学驱动力", host_flux: "优先宿主可承载通量", balanced: "综合可实现性" })[rd.priority] || "综合可实现性";
      card.appendChild(el("p", "pathway-auto-enzyme", `${policy} · 最多 ${rd.max_steps || 6} 步 · 返回 ${rd.route_count || 10} 条。需要改变偏好时，直接在自然语言里说。`));
    } else if (resolution.direction === "pathway_compatibility") {
      const pathway = resolution.pathway_resolution || {};
      (pathway.steps || []).forEach((step, stepIndex) => {
        const section = verificationSection(`第 ${stepIndex + 1} 步`, step.mention || "核对这一步反应");
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
          label.append(el("strong", "", "你指定的酶"), el("small", "", enzyme.interpreted_protein || "请核对蛋白记录"));
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
            section.appendChild(el("p", "empty-inline", "没有找到可核对的蛋白记录；请修改这一步的酶描述后再试。"));
          }
        } else {
          section.appendChild(el("p", "pathway-auto-enzyme", "这一步未指定酶：确认路径后，系统会从候选中与其他步骤一起联合选择。"));
        }
        card.appendChild(section);
      });
    } else {
      const protein = resolution.protein_resolution;
      const psec = verificationSection("目标酶", protein?.interpreted_protein || "蛋白匹配结果");
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

    const footer = el("div", "verification-actions");
    const pathwayTask = resolution.direction === "pathway_compatibility";
    const routeDesignTask = resolution.direction === "route_design";
    footer.appendChild(el("p", "", pathwayTask
      ? "逐步核对 Rhea 记录；只有你主动指定的酶才需要额外确认。未指定的步骤会在整条路径中联合选择。"
      : routeDesignTask
        ? "这里只核对起点/目标实体；路线本身由数据库图搜索生成，不由语言模型编写。"
        : "可以打开 Rhea / UniProt 查看原始记录。按 Enter 也可以确认并继续。"));
    const runText = pathwayTask ? "确认路径并评估" : routeDesignTask ? "确认目标并推荐路线" : "确认并开始筛选";
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
      if ((rd.source_candidates || []).length && !sourceRadio) { addError("请先确认起始前体。", "还需要确认路线起点"); return; }
      if (!targetRadio) { addError("请先确认目标产物。", "还需要确认路线目标"); return; }
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
        if (!reactionRadio) { addError(`请先确认第 ${index + 1} 步反应。`, "还需要确认路径"); return; }
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
      selectedTarget = `${steps.length} 步路径`;
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
      if (!reactionRadio) { addError("请先选择目标反应。", "还需要确认反应"); return; }
      const positiveIds = Array.from(card.querySelectorAll(".protein-option input:checked")).map((node) => node.value);
      selectedTarget = reactionRadio.value;
      payload = {
        endpoint: "/api/rank",
        body: {
          rhea_id: reactionRadio.value,
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
        },
      };
    } else {
      const proteinRadio = card.querySelector(".protein-option input:checked");
      if (!proteinRadio) { addError("请先选择目标酶。", "还需要确认蛋白"); return; }
      selectedTarget = proteinRadio.value;
      payload = {
        endpoint: "/api/rank-reactions",
        body: {
          protein_id: proteinRadio.value,
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

    const facts = contextFacts.querySelectorAll("span strong");
    if (facts[1]) facts[1].textContent = selectedTarget;
    runButton.disabled = true;
    runButton.textContent = resolution.direction === "pathway_compatibility" ? "正在联合评估…" : resolution.direction === "route_design" ? "正在生成并排序路线…" : "正在筛选…";
    activeVerification = null;
    card.querySelectorAll(".entity-list").forEach((group) => collapseSelectedGroup(group, "change"));
    setBusy(true);
    advanceProcess("search");
    const activity = addActivity("正在筛选候选…");

    try {
      const result = await api(payload.endpoint, payload.body);
      const pathwayTask = resolution.direction === "pathway_compatibility";
      const routeDesignTask = resolution.direction === "route_design";
      const resultCount = pathwayTask ? (result.steps?.length || 0) : routeDesignTask ? (result.routes?.length || 0) : (result.candidates?.length || 0);
      activity.update(pathwayTask ? "正在整理整条路径…" : routeDesignTask ? "正在整理候选路线…" : "正在整理结果…", pathwayTask ? `${resultCount} 个步骤已联合评估` : routeDesignTask ? `${resultCount} 条候选路线已排序` : `${resultCount} 个候选`);
      advanceProcess("result");
      renderResult(result, resolution.direction);
      updateTechnicalDetails(result);
      activity.finish(pathwayTask ? "路径评估完成" : routeDesignTask ? "路线推荐完成" : "筛选完成", pathwayTask ? result.verdict_label || `${resultCount} 步已评估` : routeDesignTask ? `${resultCount} 条候选路线` : `${resultCount} 个候选已排序`);
      completeProcess();
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
      composerContext.classList.remove("hidden");
      contextSummary.textContent = directionSummary(result, resolution.direction);
      const resultFacts = contextFacts.querySelectorAll("span strong");
      if (resolution.direction === "pathway_compatibility") {
        if (resultFacts[2]) resultFacts[2].textContent = result.verdict_label || "联合评估";
      } else if (resolution.direction === "route_design") {
        if (resultFacts[2]) resultFacts[2].textContent = `${result.routes?.length || 0} 条路线`;
      } else {
        const actualAssociationMode = associationMode(result);
        if (resultFacts[2]) resultFacts[2].textContent = actualAssociationMode.label;
      }
    } catch (error) {
      activity.fail("筛选没有完成");
      advanceProcess("search");
      addError(error.message, resolution.direction === "pathway_compatibility"
        ? "整条路径评估没有完成"
        : resolution.direction === "route_design" ? "候选路线生成没有完成"
        : resolution.direction === "reaction_to_enzyme" ? "候选酶筛选没有完成" : "候选反应筛选没有完成");
      runButton.disabled = false;
      runButton.textContent = resolution.direction === "pathway_compatibility" ? "确认路径并评估" : resolution.direction === "route_design" ? "确认目标并推荐路线" : "确认并开始筛选";
      activeVerification = { card, button: runButton };
    } finally {
      setBusy(false);
    }
  }

  function associationMode(result) {
    const discovery = result.discovery_filter || {};
    const resultMode = String(discovery.result_mode || "");
    const filterPolicy = String(discovery.policy || "");
    let policy = "allow_known";
    if (resultMode === "known_associations_only" || filterPolicy === "retain_recorded_associations_only") {
      policy = "known_only";
    } else if (resultMode === "novel_association_discovery" || filterPolicy === "exclude_recorded_associations") {
      policy = "exclude_known";
    }
    const labels = {
      allow_known: "全部候选",
      known_only: "仅已记录",
      exclude_known: "仅未记录",
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
      return `${count} 步路径已联合评估：${result.verdict_label || "已完成"}。pH / 温度证据覆盖 ${core}/${count} 步。`;
    }
    if (direction === "route_design") {
      const count = result.routes?.length || 0;
      const target = result.selected_target?.name || result.selected_target?.chebi_id || "目标产物";
      return `已为 ${target} 找到并排序 ${count} 条 Rhea 已知候选路线；可继续用自然语言指定某条路线做多酶兼容性评估。`;
    }
    const count = result.candidates?.length || 0;
    const mode = associationMode(result);
    const noun = direction === "reaction_to_enzyme" ? "候选酶" : "候选反应";
    if (mode.knownOnly) {
      return `已得到 ${count} 个已记录${noun}，按模型评分排序。`;
    }
    if (mode.excluded) {
      return mode.knownCount
        ? `已得到 ${count} 个未记录${noun}，并过滤 ${mode.knownCount} 条当前知识库关联。`
        : `已得到 ${count} 个未记录${noun}；当前知识库没有关联需要过滤。`;
    }
    return mode.knownCount
      ? `已得到 ${count} 个${noun}，其中 ${mode.knownCount} 条知识库关联可参与同一模型排序。`
      : `已得到 ${count} 个${noun}；当前知识库没有已记录关联。`;
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
    return ({ one_pot: "同一体系", sequential: "分步反应", in_vivo: "体内路径", auto: "按描述判断" })[mode] || "按描述判断";
  }

  function appendPathwayRouteDetails(card, result) {
    const details = document.createElement("details");
    details.className = "result-route-details";
    const summary = document.createElement("summary");
    summary.textContent = "查看技术详情";
    details.appendChild(summary);
    const technical = el("div", "result-technical");
    const openRoute = el("button", "result-route-open");
    openRoute.type = "button";
    openRoute.append(el("strong", "", result.route_view?.title || "查看整条路径评估流程"), el("span", "", "打开完整流程图 ↗"));
    openRoute.addEventListener("click", () => openActualRouteDialog(result.route_view || {}));
    technical.appendChild(openRoute);
    technical.appendChild(el("code", "result-route-code", result.route_view?.route_id || "pathway-compatibility-v1"));
    const route = el("div", "inline-route");
    (result.route_view?.nodes || []).forEach((node, index) => {
      const item = el("div", `inline-route-node kind-${node.kind || "control"}`);
      item.append(el("span", "", String(index + 1).padStart(2, "0")), el("strong", "", node.title), el("small", "", node.metric || node.subtitle || ""));
      route.appendChild(item);
      if (index < (result.route_view?.nodes || []).length - 1) route.appendChild(el("i", "route-arrow", "→"));
    });
    technical.appendChild(route);
    details.appendChild(technical);
    card.appendChild(details);
  }

  function renderPathwayResult(result) {
    const { content } = messageShell("assistant");
    const steps = result.steps || [];
    const coverage = result.coverage || {};
    const shared = result.shared_conditions || {};
    const target = result.target_conditions || {};
    const intro = el("div", "assistant-copy result-intro");
    intro.append(el("p", "", result.verdict_label || "整条路径兼容性评估完成。"));
    if (result.verdict === "partial_evidence" || result.verdict === "insufficient_evidence") {
      intro.append(el("p", "subtle", "缺失的 pH / 温度数据被视为未知，不会被当作“没有冲突”。"));
    } else {
      intro.append(el("p", "subtle", "结果综合了各步模型排序和可获得的条件证据；它用于缩小实验空间，不替代混合稳定性与活性验证。"));
    }
    content.appendChild(intro);

    const card = el("div", "result-card pathway-result-card");
    const head = el("div", "result-head");
    const titleWrap = el("div");
    titleWrap.append(el("strong", "", "整条路径的酶组合"), el("small", "", "各步模型优先级为主 · 条件兼容性做全局联合选择"));
    head.appendChild(titleWrap);
    card.appendChild(head);

    const chips = el("div", "result-chips");
    [
      `${steps.length} 步`,
      pathwayModeLabel(result.execution_mode),
      target.ph !== null && target.ph !== undefined && Number.isFinite(Number(target.ph)) ? `目标 pH ${Number(target.ph)}` : null,
      target.temperature_c !== null && target.temperature_c !== undefined && Number.isFinite(Number(target.temperature_c)) ? `目标 ${Number(target.temperature_c)} °C` : null,
      (target.cofactors || []).length ? `目标辅因子 ${(target.cofactors || []).join(" / ")}` : null,
      `pH / 温度证据 ${coverage.core_condition_steps || 0}/${coverage.total_steps || steps.length}`,
      shared.ph_label ? `共同 pH ${shared.ph_label}` : null,
      shared.temperature_label ? `共同温度 ${shared.temperature_label}` : null,
      (shared.cofactors || []).length ? `共同辅因子 ${(shared.cofactors || []).join(" / ")}` : null,
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
      title.append(el("small", "", step.rhea_id || `第 ${step.step_index} 步`));
      const link = externalLink(candidate.uniprot_url || profile.url || "#", candidate.candidate_id || "候选酶");
      link.classList.add("pathway-enzyme-link");
      title.appendChild(link);
      const badges = el("div", "pathway-step-badges");
      if (candidate.local_rank) badges.appendChild(el("span", "", `单步 #${candidate.local_rank}`));
      if (step.changed_for_pathway_compatibility) badges.appendChild(el("span", "changed", "联合重排"));
      top.append(title, badges);
      body.appendChild(top);
      const meta = [candidate.name, candidate.species].filter(Boolean).join(" · ");
      if (meta) body.appendChild(el("p", "pathway-step-meta", meta));
      const conditions = el("div", "pathway-condition-chips");
      const ph = profile.ph_active || profile.ph_optimum;
      const temp = profile.temperature_active_c || profile.temperature_optimum_c;
      const phText = intervalLabel(ph);
      const tempText = intervalLabel(temp, " °C");
      if (phText) conditions.appendChild(el("span", "", `${profile.ph_active ? "pH 范围" : "最适 pH"} ${phText}`));
      if (tempText) conditions.appendChild(el("span", "", `${profile.temperature_active_c ? "温度范围" : "最适温度"} ${tempText}`));
      if ((profile.cofactors || []).length) conditions.appendChild(el("span", "", `辅因子 ${(profile.cofactors || []).slice(0, 3).join(" / ")}`));
      if (Number.isFinite(Number(profile.theoretical_pi))) conditions.appendChild(el("span", "", `理论 pI ${Number(profile.theoretical_pi).toFixed(2)}`));
      if ((profile.locations || []).length) conditions.appendChild(el("span", "", (profile.locations || []).slice(0, 2).join(" / ")));
      if (!phText && !tempText) conditions.appendChild(el("span", "unknown", "pH / 温度暂无结构化注释"));
      body.appendChild(conditions);
      row.append(marker, body);
      stepList.appendChild(row);
    });
    card.appendChild(stepList);

    const conflictBox = el("section", "pathway-conflict-box");
    conflictBox.appendChild(el("strong", "", (result.conflicts || []).length ? "需要关注的跨步证据" : "没有发现有证据支持的强冲突"));
    if ((result.conflicts || []).length) {
      const list = el("div", "pathway-conflict-list");
      (result.conflicts || []).slice(0, 8).forEach((item) => {
        const row = el("div", `pathway-conflict-item severity-${item.severity || "medium"}`);
        row.append(el("span", "", `步骤 ${(item.steps || []).join(" / ")}`), el("p", "", item.detail || "条件存在差异，需要人工核对。"));
        list.appendChild(row);
      });
      conflictBox.appendChild(list);
    } else {
      conflictBox.appendChild(el("p", "", "这里只表示当前可获得注释中没有发现明确冲突；未报道条件仍是未知。"));
    }
    card.appendChild(conflictBox);

    if ((result.recommendations || []).length) {
      const rec = el("section", "pathway-recommendations");
      rec.appendChild(el("strong", "", "实验建议"));
      const list = document.createElement("ul");
      (result.recommendations || []).slice(0, 5).forEach((text) => list.appendChild(el("li", "", text)));
      rec.appendChild(list);
      card.appendChild(rec);
    }
    card.appendChild(el("p", "score-note", "路径兼容性层不会直接预测蛋白沉淀或长期失活；浓度、pI、盐、buffer、底物/产物、溶剂与时间仍需要实验验证。"));
    appendPathwayRouteDetails(card, result);
    content.appendChild(card);
    scrollConversation();
  }

  function resultChips(result, direction, topK) {
    const mode = associationMode(result);
    const chips = [`${topK} 个结果`, mode.label];
    if (mode.knownOnly) {
      if (mode.knownCount) chips.push(`${mode.knownCount} 条知识库关联`);
    } else if (mode.excluded && mode.knownCount) {
      chips.push(`过滤 ${mode.knownCount} 条已记录关联`);
    } else if (mode.mixed && mode.knownCount) {
      chips.push(`${mode.knownCount} 条已记录关联可参与排序`);
    }
    if (direction === "reaction_to_enzyme") {
      const taxonomy = result.ranking?.enzyme_taxonomy_scope;
      if (taxonomy === "eukaryote") chips.push("仅真核候选");
      else if (taxonomy === "prokaryote") chips.push("仅原核候选");
      if (result.ranking?.shot_mode === "few_shot") {
        const count = result.routing?.known_enzyme_ids?.length || result.routing?.confirmed_positive_enzymes?.length || 0;
        chips.push(count ? `参考 ${count} 个已知有效酶` : "参考已知有效酶");
      }
      if (result.routing?.homology_policy === "cross_cluster") chips.push("已避开近缘同源");
    } else if (result.routing?.known_activity_policy === "seed_known") {
      chips.push("参考已有活性扩展");
    }
    return chips;
  }

  function routePriorityLabel(priority) {
    return ({ balanced: "综合可实现性", short: "优先短路线", enzyme_available: "优先酶可获得性", project_covered: "优先项目模型覆盖", thermodynamic: "优先热力学驱动力", host_flux: "优先宿主可承载通量" })[priority] || "综合可实现性";
  }

  function renderRouteDesignResult(result) {
    const { content } = messageShell("assistant");
    const routes = result.routes || [];
    const intro = el("div", "assistant-copy result-intro");
    const target = result.selected_target?.name || result.selected_target?.chebi_id || "目标产物";
    intro.append(el("p", "", routes.length
      ? `为 ${target} 找到了 ${routes.length} 条 Rhea 已知候选路线，并按${routePriorityLabel(result.priority)}排序。`
      : `在当前 Rhea 已知反应图中没有找到通向 ${target} 的候选路线。`));
    const feas = result.feasibility || {};
    const filtered = Number(feas.host_infeasible_filtered_count || 0);
    const thermoCount = Number(feas.thermo_complete_count || 0);
    const evidenceText = filtered
      ? `先评估了 ${Number(feas.preliminary_route_count || routes.length)} 条预候选；iML1515 route-supported FBA 过滤了 ${filtered} 条整路通量为 0 的路线。`
      : `热力学可计算 ${thermoCount}/${Number(feas.preliminary_route_count || routes.length)} 条预候选；缺失数据保持未知。`;
    intro.append(el("p", "subtle", `语言模型只理解起点、目标和排序偏好；反应步骤来自 Rhea。${evidenceText} 路线分数是相对优先级，不是实验成功率。`));
    content.appendChild(intro);

    const card = el("div", "result-card route-design-result-card");
    const head = el("div", "result-head");
    const titleWrap = el("div");
    titleWrap.append(el("strong", "", "候选生物合成路线"), el("small", "", result.feasibility?.host_expected ? "Rhea · eQuilibrator MDF · E. coli iML1515 FBA" : "Rhea · eQuilibrator MDF · 多指标相对排序"));
    head.appendChild(titleWrap);
    card.appendChild(head);

    const stats = result.graph_stats || {};
    const chips = el("div", "result-chips");
    [
      `${routes.length} 条路线`,
      routePriorityLabel(result.priority),
      result.source_mode === "ecoli_iML1515_pool" ? "E. coli iML1515 起点池" : "确认前体出发",
      stats.route_nodes ? `${Number(stats.route_nodes).toLocaleString()} 个 Rhea 图节点` : null,
      stats.route_edges ? `${Number(stats.route_edges).toLocaleString()} 条主转化边` : null,
      thermoCount ? `${thermoCount} 条有 MDF` : null,
      filtered ? `FBA 过滤 ${filtered} 条零通量路线` : null,
    ].filter(Boolean).forEach((text) => chips.appendChild(el("span", "", text)));
    card.appendChild(chips);

    const list = el("div", "route-design-list");
    routes.forEach((route) => {
      const item = el("article", "route-design-item");
      const top = el("div", "route-design-item-head");
      const rank = el("span", "route-design-rank", `#${route.rank || ""}`);
      const summary = el("div", "route-design-summary");
      summary.append(el("strong", "", (route.compound_names || []).join(" → ") || route.route_id || "候选路线"));
      summary.append(el("small", "", route.route_id || "Rhea route"));
      const score = el("div", "route-design-score");
      score.append(el("strong", "", Number(route.score || 0).toFixed(1)), el("small", "", "综合相对分"));
      top.append(rank, summary, score);
      item.appendChild(top);

      const metrics = route.metrics || {};
      const metricRow = el("div", "route-design-metrics");
      const thermo = route.thermodynamics || {};
      const hostFeasibility = route.host_feasibility || {};
      const mdf = Number(thermo.mdf_kj_mol);
      const flux50 = Number(hostFeasibility.max_route_flux_50pct_growth);
      [
        `${metrics.step_count || route.steps?.length || 0} 步`,
        `酶可获得性 ${Math.round(Number(metrics.enzyme_availability || 0) * 100)}%`,
        thermo.status === "complete" && Number.isFinite(mdf) ? `MDF ${mdf.toFixed(1)} kJ/mol` : "MDF 未覆盖",
        hostFeasibility.status === "complete" && Number.isFinite(flux50) ? `iML1515 路线通量 ${flux50.toFixed(2)} @≥50%生长` : (result.feasibility?.host_expected ? "iML1515 FBA 未知" : null),
        `项目模型覆盖 ${Math.round(Number(metrics.project_model_coverage || 0) * 100)}%`,
        Number.isFinite(Number(metrics.min_swissprot_count)) ? `最少 Swiss-Prot ${Number(metrics.min_swissprot_count)}` : null,
      ].filter(Boolean).forEach((text) => metricRow.appendChild(el("span", "", text)));
      item.appendChild(metricRow);

      const steps = el("div", "route-design-steps");
      (route.steps || []).forEach((step) => {
        const row = el("div", "route-design-step");
        row.appendChild(el("span", "route-design-step-index", String(step.step_index || "")));
        const copy = el("div", "route-design-step-copy");
        copy.append(el("strong", "", `${step.source_name || step.source} → ${step.target_name || step.target}`));
        const meta = [];
        if (step.swissprot_count !== undefined) meta.push(`Swiss-Prot 酶记录 ${step.swissprot_count}`);
        const thermoStep = (thermo.steps || []).find((row) => Number(row.step_index) === Number(step.step_index));
        const physiological = Number(thermoStep?.physiological_dg_prime?.value_kj_mol);
        if (Number.isFinite(physiological)) meta.push(`ΔG′(phys) ${physiological.toFixed(1)} kJ/mol`);
        if (step.local_model_ready) meta.push("项目 R2E 可直接评估");
        else meta.push("可用外部 Rhea SMILES 进入 R2E");
        copy.appendChild(el("small", "", meta.join(" · ")));
        row.append(copy, externalLink(step.url || `https://www.rhea-db.org/rhea/${String(step.rhea_id || "").replace("RHEA:", "")}`, `${step.rhea_id || "Rhea"} ↗`));
        steps.appendChild(row);
      });
      item.appendChild(steps);

      const action = el("button", "route-design-template-action", "填入这条路线继续评估酶兼容性");
      action.type = "button";
      action.addEventListener("click", () => {
        const chain = (route.compound_names || []).join(" → ");
        const rhea = (route.steps || []).map((step) => step.rhea_id).filter(Boolean).join(" → ");
        input.value = `请评估这条完整路径的酶组合兼容性：${chain}。对应 Rhea 步骤：${rhea}。如果某一步没有指定酶，请联合选择候选，并检查 pH、温度、辅因子和其他条件冲突。`;
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
      card.appendChild(el("p", "route-design-exploration-note", `MDF：eQuilibrator / equilibrator-pathway，${conditionText || "默认水相条件"}；浓度边界使用 eQuilibrator 默认设置。MDF 是热力学驱动力，不代表酶活性。`));
    }
    if (result.host_feasibility_run?.status === "complete") {
      card.appendChild(el("p", "route-design-exploration-note", "iML1515 路线通量表示在候选每一步都实际承载共同通量、并保持指定最低生长时的化学计量容量；不是滴度、动力学或发酵产量预测。"));
    }

    const exploratory = result.exploratory_routes || [];
    if (exploratory.length) {
      const section = el("section", "route-exploration-section");
      const sectionHead = el("div", "route-exploration-head");
      const copy = el("div");
      copy.append(el("strong", "", "预测探索路线"), el("small", "", "独立榜单 · 不与 Rhea 已知路线混排"));
      sectionHead.append(copy, el("span", "route-exploration-count", `${exploratory.length} 条`));
      section.appendChild(sectionHead);
      section.appendChild(el("p", "route-design-exploration-note", result.exploration_backend?.predicted_note || "至少包含一个 MINE/Pickaxe + MetaCyc rule 预测步骤，必须独立验证。"));
      const xlist = el("div", "route-design-list exploratory");
      exploratory.forEach((route) => {
        const item = el("article", "route-design-item predicted-route");
        const top = el("div", "route-design-item-head");
        const rank = el("span", "route-design-rank", `P${route.rank || ""}`);
        const summary = el("div", "route-design-summary");
        summary.append(el("strong", "", (route.compound_names || []).join(" → ") || route.route_id || "预测路线"));
        summary.append(el("small", "", `${route.route_id || "Pickaxe route"} · 预测分数只在探索候选内比较`));
        const score = el("div", "route-design-score");
        score.append(el("strong", "", Number(route.score || 0).toFixed(1)), el("small", "", "探索相对分"));
        top.append(rank, summary, score);
        item.appendChild(top);
        const steps = el("div", "route-design-steps");
        (route.steps || []).forEach((step) => {
          const row = el("div", `route-design-step ${step.evidence_type === "predicted_pickaxe" ? "predicted" : "known"}`);
          row.appendChild(el("span", "route-design-step-index", String(step.step_index || "")));
          const body = el("div", "route-design-step-copy");
          body.append(el("strong", "", `${step.source_name || step.source} → ${step.target_name || step.target}`));
          if (step.evidence_type === "predicted_pickaxe") {
            body.appendChild(el("small", "", `预测步骤 · MetaCyc rules ${(step.prediction_rules || []).join(" / ") || "未标注"}`));
            row.append(body, el("span", "prediction-badge", "预测"));
          } else {
            body.appendChild(el("small", "", `Rhea 已知步骤 · Swiss-Prot ${step.swissprot_count || 0}`));
            row.append(body, externalLink(step.url || `https://www.rhea-db.org/rhea/${String(step.rhea_id || "").replace("RHEA:", "")}`, `${step.rhea_id || "Rhea"} ↗`));
          }
          steps.appendChild(row);
        });
        item.appendChild(steps);
        item.appendChild(el("p", "predicted-route-warning", route.evidence_note || "预测步骤不是数据库事实，不能直接当作已验证通路。"));
        xlist.appendChild(item);
      });
      section.appendChild(xlist);
      card.appendChild(section);
    } else if (result.exploration_backend?.predicted_note) {
      card.appendChild(el("p", "route-design-exploration-note", result.exploration_backend.predicted_note));
    }
    card.appendChild(el("p", "score-note", result.score_note || "路线分数只用于候选路线的相对优先级。"));
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
    const topK = result.ranking?.top_k || result.candidates?.length || 0;
    const intro = el("div", "assistant-copy result-intro");
    const mode = associationMode(result);
    const noun = direction === "reaction_to_enzyme" ? "候选酶" : "候选反应";
    if (mode.knownOnly) {
      intro.append(el("p", "", `找到了 ${result.candidates?.length || 0} 个已记录${noun}，按模型评分排序。`));
      intro.append(el("p", "subtle", "“已记录”表示当前知识库已有这个反应–酶配对。"));
    } else if (mode.excluded) {
      intro.append(el("p", "", `找到了 ${result.candidates?.length || 0} 个未记录${noun}。`));
      intro.append(el("p", "subtle", mode.knownCount
        ? `已过滤 ${mode.knownCount} 条当前知识库关联；未记录候选仍需要实验验证。`
        : "当前知识库没有关联需要过滤；未记录候选仍需要实验验证。"));
    } else {
      intro.append(el("p", "", `找到了 ${result.candidates?.length || 0} 个${noun}。已知与潜在候选按同一模型评分混排。`));
      intro.append(el("p", "subtle", "“已知 / 潜在”只表示当前知识库是否已有这个配对。"));
    }
    content.appendChild(intro);

    const card = el("div", "result-card");
    const head = el("div", "result-head");
    const titleWrap = el("div");
    titleWrap.append(el("strong", "", direction === "reaction_to_enzyme" ? "候选酶" : "候选反应"));
    titleWrap.append(el("small", "", mode.knownOnly
      ? "仅当前知识库已记录关联 · 按模型评分排列"
      : mode.excluded
        ? "已排除当前知识库已记录关联 · 按模型评分排列"
        : "已知与潜在混排 · 按模型评分排列"));
    head.appendChild(titleWrap);
    const entityLink = direction === "reaction_to_enzyme"
      ? externalLink(result.reaction?.url || "#", `${result.reaction?.rhea_id || "Rhea"} ↗`)
      : externalLink(result.protein?.url || "#", `${result.protein?.id || "UniProt"} ↗`);
    head.appendChild(entityLink);
    card.appendChild(head);

    const chips = el("div", "result-chips");
    resultChips(result, direction, topK).forEach((text) => chips.appendChild(el("span", "", text)));
    card.appendChild(chips);

    const tableWrap = el("div", "table-wrap");
    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const hr = document.createElement("tr");
    ["排名", direction === "reaction_to_enzyme" ? "候选酶" : "候选反应", "模型评分"].forEach((text) => hr.appendChild(el("th", "", text)));
    thead.appendChild(hr);
    const tbody = document.createElement("tbody");

    (result.candidates || []).forEach((row) => {
      const tr = document.createElement("tr");
      tr.appendChild(el("td", "rank-cell", String(row.rank)));
      const entity = el("td", "result-entity");
      const primary = el("div", "result-entity-primary");

      if (direction === "reaction_to_enzyme") {
        const a = externalLink(row.uniprot_url, row.candidate_id);
        a.classList.add("entity-primary-link");
        primary.appendChild(a);
        const meta = [row.name, row.species].filter(Boolean).join(" · ");
        entity.appendChild(primary);
        if (meta) entity.appendChild(el("small", "", meta));
      } else {
        if (row.rhea_url) {
          const a = externalLink(row.rhea_url, row.candidate_id);
          a.classList.add("entity-primary-link");
          primary.appendChild(a);
        } else {
          primary.appendChild(el("strong", "entity-primary-text", row.candidate_id));
        }
        entity.appendChild(primary);
        const meta = row.name || [row.substrate_name, row.product_name].filter(Boolean).join(" → ");
        if (meta) entity.appendChild(el("small", "", meta));
      }

      if (mode.mixed) {
        primary.appendChild(el(
          "span",
          `association-badge ${row.known_association ? "known" : "potential"}`,
          row.known_association ? "已知" : "潜在",
        ));
      }
      if (Number(row.rank) <= 3) primary.appendChild(el("span", "priority-badge", "优先查看"));
      tr.appendChild(entity);

      const score = el("td", "score-cell");
      score.appendChild(el("span", "score-number", Number(row.score || 0).toFixed(4)));
      const track = el("span", "score-track");
      const fill = el("i");
      fill.style.width = `${Math.max(2, Math.min(100, Number(row.score_fraction || 0) * 100))}%`;
      track.appendChild(fill);
      score.appendChild(track);
      tr.appendChild(score);
      tbody.appendChild(tr);
    });

    table.append(thead, tbody);
    tableWrap.appendChild(table);
    card.appendChild(tableWrap);
    card.appendChild(el("p", "score-note", result.score_note || "评分用于候选之间的相对排序，不代表真实催化成功概率。"));

    const details = document.createElement("details");
    details.className = "result-route-details";
    const summary = document.createElement("summary");
    summary.textContent = "查看技术详情";
    details.appendChild(summary);
    const technical = el("div", "result-technical");
    const openRoute = el("button", "result-route-open");
    openRoute.type = "button";
    openRoute.append(el("strong", "", result.route_view?.title || "查看本次模型路线"), el("span", "", "打开完整流程图 ↗"));
    openRoute.addEventListener("click", () => openActualRouteDialog(result.route_view || {}));
    technical.appendChild(openRoute);
    const routeCode = el("code", "result-route-code", result.route_view?.route_id || result.ranking?.route_id || "");
    technical.appendChild(routeCode);
    const route = el("div", "inline-route");
    (result.route_view?.nodes || []).forEach((node, index) => {
      const item = el("div", `inline-route-node kind-${node.kind || "control"}`);
      item.append(el("span", "", String(index + 1).padStart(2, "0")), el("strong", "", node.title), el("small", "", node.metric || node.subtitle || ""));
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
      detail: "该步骤来自仓库中的路线定义。",
    }));
  }

  function routeDialogBadges(route, actual) {
    const badges = [];
    if (actual) badges.push("本次实际执行");
    if (route.direction === "reaction_to_enzyme") badges.push("反应 → 酶");
    if (route.direction === "enzyme_to_reaction") badges.push("酶 → 反应");
    if (route.direction === "pathway_compatibility") badges.push("整条路径 · 多酶兼容性");
    if (route.direction === "route_design") badges.push("候选路线 · 生成与排序");
    if (route.scope && route.scope !== "any") badges.push(route.scope === "current" ? "库内实体" : route.scope === "external" ? "外部实体" : route.scope);
    if (route.objective) badges.push(String(route.objective).replace("top", "Top "));
    if (route.availability) badges.push(route.availability);
    return badges;
  }

  function routeDialogIntro(route, flow, actual) {
    if (actual) return route.summary || "这条流程由本次输入和生产路由规则实际确定。下面逐步展示每个模块在做什么。";
    const chineseDescription = [route.description, route.use_case].find((text) => /[\u3400-\u9fff]/.test(String(text || "")));
    if (chineseDescription) return chineseDescription;
    const direction = route.direction === "reaction_to_enzyme" ? "反应到酶" : route.direction === "enzyme_to_reaction" ? "酶到反应" : route.direction === "pathway_compatibility" ? "整条路径兼容性" : route.direction === "route_design" ? "候选路线生成与排序" : "扩展";
    const scope = route.scope === "current" ? "库内实体" : route.scope === "external" ? "外部实体" : "多场景";
    const depth = route.objective ? ` · ${String(route.objective).replace("top", "Top ")}` : "";
    return `这是一条${scope}的${direction}流程${depth}，包含 ${flow.length} 个步骤。下面按执行顺序说明每一步处理什么信息，以及它如何影响最终候选。`;
  }

  function openRouteDialog(route, { actual = false } = {}) {
    if (!routeDialog || !route) return;
    const flow = normalizeRouteFlow(route, actual);
    routeDialogFlow.replaceChildren();
    routeDialogMeta.replaceChildren();
    routeDialogType.textContent = actual ? "本次实际路线" : route.availability === "downstream" || route.availability === "batch" || route.availability === "specialist" ? "扩展工作流" : "模型路线";
    routeDialogTitle.textContent = route.title || route.label || route.key || "路线流程";
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
      title.append(el("strong", "", step.title || step.id || `步骤 ${index + 1}`), el("small", "", step.subtitle || routeKindNames[step.kind] || "流程步骤"));
      head.append(title, el("em", "", routeKindLabels[step.kind] || "STEP"));
      card.appendChild(head);
      if (actual && step.metric) {
        const metric = el("div", "route-diagram-metric");
        metric.append(el("small", "", "本次运行"), el("strong", "", step.metric));
        card.appendChild(metric);
      }
      card.appendChild(el("p", "", step.detail || "该步骤来自仓库中的生产路线定义。"));
      const foot = el("div", "route-diagram-step-foot");
      if (step.id) foot.appendChild(el("code", "", step.id));
      if (actual && step.note) foot.appendChild(el("span", "", step.note));
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
    routeTitle.textContent = view.title || "已完成";
    routeId.textContent = view.route_id || result.ranking?.route_id || "";
    routeStepCount.textContent = `${nodes.length} 个模块`;
    currentRouteView = view;
    routeTitleButton.disabled = !nodes.length;

    nodes.forEach((node, index) => {
      const row = el("div", `route-step kind-${node.kind || "control"}`);
      const marker = el("span", "route-marker", String(index + 1).padStart(2, "0"));
      const copy = el("div", "route-step-copy");
      const top = el("div", "route-step-top");
      top.append(el("strong", "", node.title), el("em", "route-kind", routeKindLabels[node.kind] || "STEP"));
      copy.append(top, el("small", "route-metric", node.metric || node.subtitle || ""));
      if (node.id) copy.appendChild(el("code", "route-module-id", node.id));
      if (node.note) copy.appendChild(el("p", "", node.note));
      if (node.detail) row.title = node.detail;
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
    routeCatalogCount.textContent = `${bases.length + overlays.length} 条路径`;

    const statNodes = routeCatalogStats?.querySelectorAll("span");
    if (statNodes?.length >= 3) {
      statNodes[0].querySelector("strong").textContent = String(bases.length);
      statNodes[1].querySelector("strong").textContent = String(overlays.length);
      statNodes[2].querySelector("strong").textContent = String(downstream.length);
    }

    const groups = [
      ["反应 → 酶", bases.filter((row) => row.direction === "reaction_to_enzyme"), "R2E"],
      ["酶 → 反应", bases.filter((row) => row.direction === "enzyme_to_reaction"), "E2R"],
      ["附加模块", overlays, "OVERLAY"],
      ["扩展流程", downstream, "WORKFLOW"],
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
        itemHead.append(el("strong", "", row.label || row.title || row.key || "route"));
        const flowCount = (row.flow || row.modules || []).length;
        if (flowCount) itemHead.appendChild(el("em", "", `${flowCount} steps`));
        button.appendChild(itemHead);
        if (row.key) button.appendChild(el("code", "catalog-route-key", row.key));
        const path = row.modules?.length ? row.modules.join("  →  ") : row.description || "点击查看流程图";
        button.appendChild(el("small", "catalog-module-path", path));
        button.appendChild(el("span", "catalog-open-hint", "查看流程图 ↗"));
        button.addEventListener("click", () => openRouteDialog(row));
        item.appendChild(button);
        group.appendChild(item);
      });
      routeCatalog.appendChild(group);
    });
  }



  function renderIntentChoice(resolution, displayText, effectiveText) {
    const { content } = messageShell("assistant");
    content.appendChild(el("p", "", resolution.summary || "你的描述可能有两种理解，请选择。"));
    const box = el("div", "intent-choice-card");
    (resolution.intent_options || []).filter((x) => x.direction).forEach((option) => {
      const button = el("button", "secondary-button", option.label);
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
    activeVerification = null;
    const continued = Boolean(continuation && useContinuation);
    const effectiveText = continued ? `${continuation.originalText}\n用户后续要求：${text}` : text;
    // Previous direction is context, not a hard routing constraint. An explicit
    // selector/ambiguity choice still uses directionHint; ordinary follow-ups stay
    // auto so DeepSeek can freely switch task and result scope.
    const effectiveHint = directionHint;
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
    } : {};
    input.value = "";
    addUserMessage(text, continued);
    setBusy(true);
    resetProcess();
    advanceProcess("understand");
    const activity = addActivity("正在理解你的实验目标…");

    try {
      const resolution = await api("/api/agent/resolve", {
        text: effectiveText,
        direction_hint: effectiveHint,
        conversation_context: conversationContext,
      });
      updateTechnicalLanguage(resolution.llm_provenance);
      if (resolution.direction === "ambiguous") {
        renderIntentChoice(resolution, text, effectiveText);
        activity.finish("需要确认任务类型", "等待你的选择");
        return;
      }
      const pathwayTask = resolution.direction === "pathway_compatibility";
      const routeDesignTask = resolution.direction === "route_design";
      activity.update("正在核对数据库记录…", pathwayTask
        ? "逐步核对 Rhea 与已指定蛋白"
        : routeDesignTask ? "核对路线起点与目标产物"
        : resolution.direction === "reaction_to_enzyme" ? "核对反应与相关蛋白" : "核对目标蛋白");
      renderVerification(resolution, text, effectiveText);
      const count = pathwayTask
        ? (resolution.pathway_resolution?.steps || []).reduce((n, step) => n + (step.reaction_resolution?.candidates?.length || 0) + (step.enzyme_resolution?.candidates?.length || 0), 0)
        : routeDesignTask
          ? (resolution.route_design_resolution?.source_candidates?.length || 0) + (resolution.route_design_resolution?.target_candidates?.length || 0)
          : resolution.direction === "reaction_to_enzyme"
            ? (resolution.reaction_resolution?.candidates?.length || 0) + (resolution.positive_enzyme_resolutions || []).reduce((n, group) => n + (group.candidates?.length || 0), 0)
            : resolution.protein_resolution?.candidates?.length || 0;
      activity.finish(pathwayTask ? "路径步骤已核对" : routeDesignTask ? "路线目标已核对" : "已找到可核对的数据库记录", `${count} 条匹配 · 等待确认`);
    } catch (error) {
      activity.fail("没有完成数据库核对");
      addError(error.message, error.code === "deepseek_key_missing" ? "自然语言功能暂不可用" : "没有找到可确认的记录");
      resetProcess();
    } finally {
      setBusy(false);
      input.focus({ preventScroll: true });
    }
  }

  function focusFirstPlaceholder() {
    const match = /【[^】]+】/.exec(input.value);
    if (!match) {
      input.setSelectionRange(input.value.length, input.value.length);
      return;
    }
    input.setSelectionRange(match.index, match.index + match[0].length);
  }

  function wireStarterButtons(root = document) {
    root.querySelectorAll("[data-prompt]").forEach((button) => {
      button.addEventListener("click", () => {
        useContinuation = false;
        composerContext.classList.add("hidden");
        const suggestedDirection = button.dataset.directionTemplate || "auto";
        setDirection(suggestedDirection);
        directionHintOneShot = suggestedDirection !== "auto";
        input.value = button.dataset.prompt || "";
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
    composerContext.classList.add("hidden");
    input.value = "";
    setDirection("auto");
    setRouteMode("intelligent");
    contextTitle.textContent = "还没有开始";
    contextSummary.textContent = "发送实验目标后，这里会整理目标、已确认记录和当前筛选方式。";
    const facts = contextFacts.querySelectorAll("span strong");
    facts[0].textContent = "—";
    facts[1].textContent = "—";
    facts[2].textContent = "默认混排";
    resetProcess();
    routeTitle.textContent = "尚未执行";
    currentRouteView = null;
    routeTitleButton.disabled = true;
    routeStepCount.textContent = "等待执行";
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
      feedbackStatus.textContent = "请选择一个使用感受，或写下你的意见。";
      feedbackStatus.className = "feedback-status error";
      return;
    }
    feedbackSubmit.disabled = true;
    feedbackSubmit.textContent = "提交中…";
    feedbackStatus.classList.add("hidden");
    try {
      await api("/api/feedback", {
        rating,
        category: feedbackCategory.value,
        message,
        contact: feedbackContact.value.trim(),
        context: feedbackContext(),
      });
      feedbackStatus.textContent = "已收到，谢谢你的反馈。";
      feedbackStatus.className = "feedback-status success";
      feedbackForm.querySelectorAll('input[name="rating"]').forEach((node) => { node.checked = false; });
      feedbackMessage.value = "";
      feedbackContact.value = "";
      setTimeout(() => { if (feedbackDialog.open) feedbackDialog.close(); }, 900);
    } catch (error) {
      feedbackStatus.textContent = error.message || "提交失败，请稍后重试。";
      feedbackStatus.className = "feedback-status error";
    } finally {
      feedbackSubmit.disabled = false;
      feedbackSubmit.textContent = "提交反馈";
    }
  }

  async function refreshStatus() {
    try {
      const status = await api("/api/status");
      serviceSnapshot = status;
      serviceStatus.classList.toggle("ready", status.status === "ready");
      serviceStatus.querySelector("span").textContent = status.status === "ready" ? "系统正常" : "部分功能不可用";
      updateTechnicalLanguage(null);
    } catch (_) {
      serviceStatus.classList.remove("ready");
      serviceStatus.querySelector("span").textContent = "连接异常";
    }
  }

  wireStarterButtons();
  wirePolicyPromptButtons();
  resetProcess();
  refreshStatus();
  api("/api/routes").then(renderRouteCatalog).catch(() => { routeCatalogCount.textContent = "暂不可用"; });
})();
