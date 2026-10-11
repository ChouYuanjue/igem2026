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
    assert 'src="/app.js?v=20261011-chat-header-v4"' in html
