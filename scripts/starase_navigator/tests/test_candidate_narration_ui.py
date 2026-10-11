"""Portable GitHub mirror contracts for COMPASS's candidate presentation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FRONT = ROOT / "frontend/starase_navigator"


def test_natural_language_response_stays_outside_structured_disclosure():
    app = (FRONT / "app.js").read_text(encoding="utf-8")
    prompt = app[app.index("  async function sendPrompt(text)") : app.index("  function focusFirstPlaceholder()")]
    reply = app[app.index("  function addAssistantResponse(") : app.index("  function makeLastScientificCardCollapsible(")]
    assert 'renderStructuredResults(resolution);' in prompt
    assert 'addAssistantResponse(resolution.assistant_response, { host: presentation?.host, prepend: true });' in prompt
    assert 'makeLastScientificCardCollapsible({ open: true });' in prompt
    assert "content.prepend(copy)" not in reply
    assert 'const firstBody = Array.from(content.children).find((node) => !node.classList.contains("message-meta"));' in reply
    assert "content.insertBefore(copy, firstBody || null);" in reply
    assert "scientific-explanation-fold" not in reply


def test_previous_data_folds_on_next_question_and_each_candidate_has_local_evidence():
    app = (FRONT / "app.js").read_text(encoding="utf-8")
    styles = (FRONT / "styles.css").read_text(encoding="utf-8")
    assert 'messages.querySelectorAll(".structured-card-fold[open]")' in app
    assert 'why.className = "candidate-evidence-details";' in app
    assert 'entity.appendChild(why);' in app
    assert 'why.open = !why.open;' in app
    assert 'appendActiveExpertEvidence(whyBody, row);' in app
    assert 'entry?.active === true' in app
    assert 'if (queryEvidenceOverview) contextDetails.appendChild(queryEvidenceOverview);' in app
    assert ".candidate-row-with-evidence" in styles
    assert "normalizedRankingPercent(row)" in app



def test_updated_assistant_script_version_busts_browser_cache():
    html = (FRONT / "index.html").read_text(encoding="utf-8")
    assert 'src="/app.js?v=20261011-ranking-score-v6"' in html


def test_composite_scientific_result_restores_one_model_generated_followup_group():
    js = (FRONT / "app.js").read_text(encoding="utf-8")
    compose = js[js.index("  function renderStructuredResults(resolution)") : js.index("  async function sendPrompt(text)")]
    followups = js[js.index("  function requestContextualFollowUps(") : js.index("  function externalLink(")]
    assert "const primary = observations.find((item) =>" in compose
    assert "requestContextualFollowUps(fold, primary.result, primary.direction || resolution.direction);" in compose
    assert compose.count("requestContextualFollowUps(") == 1
    assert compose.index("renderResult(result, direction, panel);") < compose.index("requestContextualFollowUps(fold, primary.result")
    assert compose.index("requestContextualFollowUps(fold, primary.result") < compose.index("return { ...observations[0], host: content };")
    assert "ui_language: uiLanguage" in followups
    assert 'api("/api/followups"' in followups
    assert 'result_context: compactFollowUpContext(result, direction)' in followups
    assert 'appendContextualFollowUps(host, prompts)' in followups
    assert 'messages.querySelectorAll(".structured-card-fold[open]")' in js
    assert 'const clean = [...new Set(' in js
    assert 'button.dataset.capabilityId = "contextual_followup";' in js


def test_normalized_ranking_score_does_not_show_confidence_percent():
    js = (FRONT / "app.js").read_text(encoding="utf-8")
    assert 'headers.push(tr("Normalized score", "归一化评分"))' in js
    assert 'normalized === null ? "—" : normalized.toFixed(1)' in js
    assert "normalized.toFixed(1)}%" not in js
    assert "not calibrated confidence or a catalysis probability" in js
    assert "不是置信度百分比，也不代表实验成功概率" in js
    # Percent remains a private CSS width unit for the ranking bar, never
    # attached to the value displayed to the user.
    assert 'fill.style.width = ' in js
