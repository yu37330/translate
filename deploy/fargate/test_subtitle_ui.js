// Browser-independent regression tests for session ID and translation routing.
// No browser automation, meeting audio, external network, or DOM package needed.
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const html = fs.readFileSync(__dirname + "/templates/index.html", "utf8");
const source = html.match(/<script>([\s\S]*?)<\/script>/)?.[1];
assert.ok(source, "inline app script exists");

const start = source.indexOf("function updateSubtitle(data) {");
const between = source.indexOf("function updateTranslation(data) {", start);
const end = source.indexOf("function clearHistory()", between);
assert.ok(start >= 0 && between > start && end > between);
const handlers = source.slice(start, end);

class Element {
  constructor(tag = "div") {
    this.tagName = tag;
    this.textContent = "";
    this.className = "";
    this.children = [];
    this.style = {};
    this.title = "";
    this.classList = {add() {}, remove() {}};
    this.scrollTop = 0;
    this.scrollHeight = 0;
  }
  appendChild(child) { this.children.push(child); }
  replaceChildren(...children) { this.children = children; }
  querySelector(cls) {
    if (!cls.startsWith(".")) return null;
    return this.children.find(child => child.className === cls.slice(1)) || null;
  }
}
const nodes = {};
for (const id of ["current", "currentOriginal", "currentTranslated",
                  "originalLabel", "historyList", "history"]) {
  nodes[id] = new Element();
}
const document = {
  getElementById: id => nodes[id],
  createElement: tag => new Element(tag),
  querySelectorAll: selector => {
    assert.equal(selector, "#historyList .item");
    return nodes.historyList.children;
  },
};
const context = {
  document,
  LANG_NAMES: {ja: "JA"},
  getSourceLang: () => "ja",
  historyData: [],
};
vm.createContext(context);
vm.runInContext(handlers, context);

function final(utterance_id, original) {
  return {
    type: "subtitle", is_final: true, original, translated: "",
    source_lang: "ja", target_lang: "none", time: "11:00:00",
    latency_ms: 120, chunk_id: 1, utterance_id,
    audio_start_ms: 0, audio_end_ms: 1000,
  };
}

context.updateSubtitle(final("session-A:s0", "最初の会議"));
context.updateSubtitle(final("session-B:s0", "次の会議"));
assert.equal(context.historyData.length, 2, "two recordings with same chunk must both survive");
context.updateSubtitle(final("session-A:s0", "duplicate should be rejected"));
assert.equal(context.historyData.length, 2, "duplicate final in same session ignored");
assert.equal(nodes.historyList.children.length, 2, "two actual subtitle rows");

assert.equal(context.updateTranslation({utterance_id:"session-B:s0", translated:"Meeting B"}),true);
assert.equal(context.historyData[1].translated,"Meeting B");
assert.equal(context.historyData[0].translated,"");
assert.equal(context.updateTranslation({utterance_id:"session-A:s0", translated:"Meeting A"}),true);
assert.equal(context.historyData[0].translated,"Meeting A");
assert.equal(context.historyData[1].translated,"Meeting B");
assert.equal(context.updateTranslation({utterance_id:"missing", translated:"wrong"}),false);
assert.equal(context.updateTranslation({chunk_id:1, translated:"wrong"}),false);
assert.equal(nodes.historyList.children[0].querySelector(".translated").textContent, "Meeting A");
assert.equal(nodes.historyList.children[1].querySelector(".translated").textContent, "Meeting B");

context.updateSubtitle({
  is_final:false, original:"生産計画について確認", stable_prefix:"生産計画",
  source_lang:"ja", translated:"", utterance_id:"session-C:s0",
});
assert.equal(context.historyData.length,2,"interims never committed");
assert.equal(nodes.currentOriginal.children[0].textContent,"生産計画");
assert.equal(nodes.currentOriginal.children[1].textContent,"について確認");
console.log("Subtitle session, translation and safe interim DOM tests passed");
