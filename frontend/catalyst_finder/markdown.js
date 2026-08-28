(() => {
  "use strict";

  const SAFE_LINK_PROTOCOLS = new Set(["http:", "https:", "mailto:"]);

  function safeHref(raw) {
    const value = String(raw || "").trim();
    if (!value) return null;
    if (value.startsWith("#") || value.startsWith("/")) return value;
    try {
      const url = new URL(value, window.location.origin);
      return SAFE_LINK_PROTOCOLS.has(url.protocol) ? url.href : null;
    } catch (_error) {
      return null;
    }
  }

  function appendText(parent, text) {
    if (text) parent.appendChild(document.createTextNode(text));
  }

  function findClosing(text, start, marker) {
    const index = text.indexOf(marker, start);
    return index >= 0 ? index : -1;
  }

  function renderInline(parent, rawText) {
    const text = String(rawText || "");
    let index = 0;
    while (index < text.length) {
      if (text[index] === "\\" && index + 1 < text.length) {
        appendText(parent, text[index + 1]);
        index += 2;
        continue;
      }

      if (text.startsWith("`", index)) {
        const end = findClosing(text, index + 1, "`");
        if (end > index + 1) {
          const code = document.createElement("code");
          code.textContent = text.slice(index + 1, end);
          parent.appendChild(code);
          index = end + 1;
          continue;
        }
      }

      const linkMatch = text.slice(index).match(/^\[([^\]\n]+)\]\(([^)\s]+)(?:\s+["']([^"']*)["'])?\)/);
      if (linkMatch) {
        const href = safeHref(linkMatch[2]);
        if (href) {
          const anchor = document.createElement("a");
          anchor.href = href;
          anchor.target = "_blank";
          anchor.rel = "noopener noreferrer";
          if (linkMatch[3]) anchor.title = linkMatch[3];
          renderInline(anchor, linkMatch[1]);
          parent.appendChild(anchor);
        } else {
          appendText(parent, linkMatch[0]);
        }
        index += linkMatch[0].length;
        continue;
      }

      const strongMarker = text.startsWith("**", index) ? "**" : text.startsWith("__", index) ? "__" : null;
      if (strongMarker) {
        const end = findClosing(text, index + 2, strongMarker);
        if (end > index + 2) {
          const strong = document.createElement("strong");
          renderInline(strong, text.slice(index + 2, end));
          parent.appendChild(strong);
          index = end + 2;
          continue;
        }
      }

      if (text.startsWith("~~", index)) {
        const end = findClosing(text, index + 2, "~~");
        if (end > index + 2) {
          const del = document.createElement("del");
          renderInline(del, text.slice(index + 2, end));
          parent.appendChild(del);
          index = end + 2;
          continue;
        }
      }

      const emphasisMarker = text[index] === "*" ? "*" : text[index] === "_" ? "_" : null;
      if (emphasisMarker && text[index + 1] !== emphasisMarker) {
        const end = findClosing(text, index + 1, emphasisMarker);
        if (end > index + 1) {
          const em = document.createElement("em");
          renderInline(em, text.slice(index + 1, end));
          parent.appendChild(em);
          index = end + 1;
          continue;
        }
      }

      let next = index + 1;
      while (next < text.length && !"\\`[*_~".includes(text[next])) next += 1;
      appendText(parent, text.slice(index, next));
      index = next;
    }
  }

  function splitTableRow(line) {
    let value = String(line || "").trim();
    if (value.startsWith("|")) value = value.slice(1);
    if (value.endsWith("|")) value = value.slice(0, -1);
    const cells = [];
    let current = "";
    let escaped = false;
    for (const char of value) {
      if (escaped) {
        current += char;
        escaped = false;
      } else if (char === "\\") {
        current += char;
        escaped = true;
      } else if (char === "|") {
        cells.push(current.trim());
        current = "";
      } else {
        current += char;
      }
    }
    cells.push(current.trim());
    return cells;
  }

  function isTableDivider(line) {
    const cells = splitTableRow(line);
    return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
  }

  function tableAlign(cell) {
    const value = String(cell || "").trim();
    const left = value.startsWith(":");
    const right = value.endsWith(":");
    if (left && right) return "center";
    if (right) return "right";
    if (left) return "left";
    return "";
  }

  function renderMarkdownInto(container, markdown) {
    if (!container) return container;
    container.replaceChildren();
    const lines = String(markdown || "").replace(/\r\n?/g, "\n").split("\n");
    let index = 0;

    while (index < lines.length) {
      const line = lines[index];
      if (!line.trim()) {
        index += 1;
        continue;
      }

      const fence = line.match(/^\s*```\s*([^`]*)$/);
      if (fence) {
        const language = String(fence[1] || "").trim().split(/\s+/)[0];
        const body = [];
        index += 1;
        while (index < lines.length && !/^\s*```\s*$/.test(lines[index])) {
          body.push(lines[index]);
          index += 1;
        }
        if (index < lines.length) index += 1;
        const pre = document.createElement("pre");
        const code = document.createElement("code");
        if (language) code.dataset.language = language;
        code.textContent = body.join("\n");
        pre.appendChild(code);
        container.appendChild(pre);
        continue;
      }

      const heading = line.match(/^\s{0,3}(#{1,6})\s+(.+)$/);
      if (heading) {
        const node = document.createElement(`h${heading[1].length}`);
        renderInline(node, heading[2].replace(/\s+#+\s*$/, ""));
        container.appendChild(node);
        index += 1;
        continue;
      }

      if (/^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/.test(line)) {
        container.appendChild(document.createElement("hr"));
        index += 1;
        continue;
      }

      if (index + 1 < lines.length && line.includes("|") && isTableDivider(lines[index + 1])) {
        const headers = splitTableRow(line);
        const dividers = splitTableRow(lines[index + 1]);
        const tableWrap = document.createElement("div");
        tableWrap.className = "markdown-table-wrap";
        const table = document.createElement("table");
        const thead = document.createElement("thead");
        const headerRow = document.createElement("tr");
        headers.forEach((cell, cellIndex) => {
          const th = document.createElement("th");
          const align = tableAlign(dividers[cellIndex] || "");
          if (align) th.style.textAlign = align;
          renderInline(th, cell);
          headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        table.appendChild(thead);
        const tbody = document.createElement("tbody");
        index += 2;
        while (index < lines.length && lines[index].trim() && lines[index].includes("|")) {
          const tr = document.createElement("tr");
          splitTableRow(lines[index]).forEach((cell, cellIndex) => {
            const td = document.createElement("td");
            const align = tableAlign(dividers[cellIndex] || "");
            if (align) td.style.textAlign = align;
            renderInline(td, cell);
            tr.appendChild(td);
          });
          tbody.appendChild(tr);
          index += 1;
        }
        table.appendChild(tbody);
        tableWrap.appendChild(table);
        container.appendChild(tableWrap);
        continue;
      }

      if (/^\s*>\s?/.test(line)) {
        const quoteLines = [];
        while (index < lines.length && /^\s*>\s?/.test(lines[index])) {
          quoteLines.push(lines[index].replace(/^\s*>\s?/, ""));
          index += 1;
        }
        const blockquote = document.createElement("blockquote");
        renderMarkdownInto(blockquote, quoteLines.join("\n"));
        container.appendChild(blockquote);
        continue;
      }

      const unorderedMatch = line.match(/^\s{0,3}[-+*]\s+(.+)$/);
      const orderedMatch = line.match(/^\s{0,3}(\d+)[.)]\s+(.+)$/);
      if (unorderedMatch || orderedMatch) {
        const ordered = Boolean(orderedMatch);
        const list = document.createElement(ordered ? "ol" : "ul");
        if (ordered) list.start = Number(orderedMatch[1]) || 1;
        while (index < lines.length) {
          const current = ordered
            ? lines[index].match(/^\s{0,3}(\d+)[.)]\s+(.+)$/)
            : lines[index].match(/^\s{0,3}[-+*]\s+(.+)$/);
          if (current) {
            const item = document.createElement("li");
            renderInline(item, ordered ? current[2] : current[1]);
            list.appendChild(item);
            index += 1;
            continue;
          }
          if (!lines[index].trim()) {
            let next = index + 1;
            while (next < lines.length && !lines[next].trim()) next += 1;
            const nextMatch = ordered
              ? lines[next]?.match(/^\s{0,3}(\d+)[.)]\s+(.+)$/)
              : lines[next]?.match(/^\s{0,3}[-+*]\s+(.+)$/);
            if (nextMatch) {
              index = next;
              continue;
            }
          }
          break;
        }
        container.appendChild(list);
        continue;
      }

      const paragraphLines = [line.trim()];
      index += 1;
      while (index < lines.length && lines[index].trim()) {
        const nextLine = lines[index];
        if (/^\s*```/.test(nextLine) || /^\s{0,3}#{1,6}\s+/.test(nextLine) || /^\s*>\s?/.test(nextLine) || /^\s{0,3}[-+*]\s+/.test(nextLine) || /^\s{0,3}\d+[.)]\s+/.test(nextLine)) break;
        if (index + 1 < lines.length && nextLine.includes("|") && isTableDivider(lines[index + 1])) break;
        paragraphLines.push(nextLine.trim());
        index += 1;
      }
      const paragraph = document.createElement("p");
      renderInline(paragraph, paragraphLines.join(" "));
      container.appendChild(paragraph);
    }
    return container;
  }

  window.CatalystMarkdown = Object.freeze({
    renderInto: renderMarkdownInto,
    safeHref,
  });
})();
