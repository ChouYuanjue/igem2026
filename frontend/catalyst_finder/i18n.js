(() => {
  const STORAGE_KEY = "catalyst_finder_ui_language";
  const normalize = (value) => String(value || "").toLowerCase().startsWith("zh") ? "zh" : "en";
  let language = normalize(localStorage.getItem(STORAGE_KEY) || "en");
  function tr(en, zh) { return language === "zh" ? zh : en; }
  function current() { return language; }
  function apply(root = document) {
    document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
    root.querySelectorAll("[data-en][data-zh]").forEach((node) => { node.textContent = language === "zh" ? node.dataset.zh : node.dataset.en; });
    root.querySelectorAll("[data-placeholder-en][data-placeholder-zh]").forEach((node) => { node.setAttribute("placeholder", language === "zh" ? node.dataset.placeholderZh : node.dataset.placeholderEn); });
    root.querySelectorAll("[data-aria-en][data-aria-zh]").forEach((node) => { node.setAttribute("aria-label", language === "zh" ? node.dataset.ariaZh : node.dataset.ariaEn); });
    root.querySelectorAll("[data-prompt-en][data-prompt-zh]").forEach((node) => { node.dataset.prompt = language === "zh" ? node.dataset.promptZh : node.dataset.promptEn; });
    root.querySelectorAll("[data-policy-prompt-en][data-policy-prompt-zh]").forEach((node) => { node.dataset.policyPrompt = language === "zh" ? node.dataset.policyPromptZh : node.dataset.policyPromptEn; });
    const toggle = document.getElementById("languageToggle");
    if (toggle) {
      toggle.textContent = language === "zh" ? "EN" : "中文";
      toggle.setAttribute("aria-label", language === "zh" ? "Switch interface to English" : "切换到中文界面");
      toggle.title = language === "zh" ? "Switching language starts a clean English session" : "切换语言会开启独立的中文会话";
    }
  }
  function switchLanguage() {
    language = language === "zh" ? "en" : "zh";
    localStorage.setItem(STORAGE_KEY, language);
    location.reload();
  }
  window.CatalystI18n = { current, tr, apply, switchLanguage, normalize };
  apply(document);
})();
