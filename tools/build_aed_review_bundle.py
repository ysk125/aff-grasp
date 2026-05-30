#!/usr/bin/env python3
"""Build a blinded static reviewer for AED failure tags."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from analysis_common import FAILURE_TAGS, read_csv, write_csv, write_json


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AED blinded review</title>
  <link rel="stylesheet" href="review.css">
</head>
<body>
  <header>
    <h1>AED blinded failure review</h1>
    <div class="status"><span id="progress"></span><span id="saved">Saved locally</span></div>
  </header>
  <main>
    <section class="viewer">
      <img id="panel" alt="AED review panel">
    </section>
    <section class="controls">
      <div id="tags" class="tag-grid"></div>
      <label class="check"><input id="complex_background" type="checkbox"> complex_background</label>
      <label class="check"><input id="uncertain" type="checkbox"> uncertain</label>
      <label for="note">note</label>
      <textarea id="note" rows="3"></textarea>
      <div class="actions">
        <button id="prev" type="button">Previous</button>
        <button id="next" type="button">Save and next</button>
      </div>
      <div class="actions secondary">
        <button id="export" type="button">Export CSV</button>
        <label class="import">Import CSV<input id="import" type="file" accept=".csv"></label>
      </div>
    </section>
  </main>
  <script src="review.js"></script>
</body>
</html>
"""


CSS = """* { box-sizing: border-box; }
body { margin: 0; font-family: Arial, sans-serif; color: #172029; background: #f4f6f5; }
header { display: flex; align-items: center; justify-content: space-between; padding: 14px 20px; background: #ffffff; border-bottom: 1px solid #d5deda; }
h1 { margin: 0; font-size: 19px; }
.status { display: flex; gap: 16px; color: #52616b; font-size: 14px; }
main { display: grid; grid-template-columns: minmax(0, 1fr) 330px; min-height: calc(100vh - 55px); }
.viewer { display: flex; align-items: center; justify-content: center; padding: 18px; overflow: auto; }
.viewer img { max-width: 100%; max-height: calc(100vh - 92px); object-fit: contain; background: #ffffff; border: 1px solid #d5deda; }
.controls { padding: 18px; background: #ffffff; border-left: 1px solid #d5deda; }
.tag-grid { display: grid; gap: 8px; margin-bottom: 18px; }
.check { display: flex; gap: 8px; align-items: center; padding: 6px 0; font-size: 14px; }
textarea { width: 100%; margin: 6px 0 18px; resize: vertical; border: 1px solid #aebbb6; padding: 8px; }
.actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px; }
button, .import { min-height: 38px; padding: 9px 10px; border: 1px solid #42726c; background: #42726c; color: #ffffff; cursor: pointer; font-size: 14px; text-align: center; }
.secondary button, .import { background: #ffffff; color: #315b56; }
.import input { display: none; }
@media (max-width: 900px) { main { grid-template-columns: 1fr; } .controls { border-left: 0; border-top: 1px solid #d5deda; } }
"""


JS = """const TAGS = __TAGS__;
const STORAGE_KEY = "affgrasp-aed-review-v1";
let items = [];
let cursor = 0;
let annotations = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
const $ = (id) => document.getElementById(id);
const escapeCsv = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;

function emptyAnnotation() {
  const out = { reviewed: false, complex_background: false, uncertain: false, note: "" };
  TAGS.forEach((tag) => out[tag] = false);
  return out;
}
function current() { return items[cursor]; }
function saveLocal() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations));
  $("saved").textContent = "Saved locally";
}
function readControls() {
  const value = emptyAnnotation();
  TAGS.forEach((tag) => value[tag] = $(tag).checked);
  value.complex_background = $("complex_background").checked;
  value.uncertain = $("uncertain").checked;
  value.note = $("note").value;
  value.reviewed = true;
  annotations[current().review_id] = value;
  saveLocal();
}
function loadControls() {
  const value = annotations[current().review_id] || emptyAnnotation();
  TAGS.forEach((tag) => $(tag).checked = !!value[tag]);
  $("complex_background").checked = !!value.complex_background;
  $("uncertain").checked = !!value.uncertain;
  $("note").value = value.note || "";
}
function render() {
  $("panel").src = current().panel;
  loadControls();
  const complete = Object.keys(annotations).length;
  $("progress").textContent = `Review ${cursor + 1} / ${items.length} | annotated ${complete}`;
  $("prev").disabled = cursor === 0;
}
function move(delta) {
  readControls();
  cursor = Math.max(0, Math.min(items.length - 1, cursor + delta));
  render();
}
function exportCsv() {
  readControls();
  const columns = ["review_id", "reviewed", "complex_background", "uncertain", ...TAGS, "note"];
  const lines = [columns.join(",")];
  items.forEach((item) => {
    const value = annotations[item.review_id] || emptyAnnotation();
    lines.push(columns.map((key) => escapeCsv(key === "review_id" ? item.review_id : value[key])).join(","));
  });
  const blob = new Blob([lines.join("\\n") + "\\n"], { type: "text/csv" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "annotations.csv";
  link.click();
  URL.revokeObjectURL(link.href);
}
function parseCsv(text) {
  const rows = [];
  let row = [], value = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i], next = text[i + 1];
    if (quoted && char === '"' && next === '"') { value += '"'; i++; }
    else if (char === '"') quoted = !quoted;
    else if (!quoted && char === ",") { row.push(value); value = ""; }
    else if (!quoted && (char === "\\n" || char === "\\r")) {
      if (char === "\\r" && next === "\\n") i++;
      row.push(value); value = "";
      if (row.some((cell) => cell !== "")) rows.push(row);
      row = [];
    } else value += char;
  }
  return rows;
}
function importCsv(file) {
  const reader = new FileReader();
  reader.onload = () => {
    const rows = parseCsv(reader.result);
    const columns = rows.shift();
    rows.forEach((row) => {
      const value = emptyAnnotation();
      const record = Object.fromEntries(columns.map((key, index) => [key, row[index] || ""]));
      TAGS.forEach((tag) => value[tag] = record[tag] === "true");
      value.complex_background = record.complex_background === "true";
      value.uncertain = record.uncertain === "true";
      value.reviewed = record.reviewed === "true";
      value.note = record.note || "";
      annotations[record.review_id] = value;
    });
    saveLocal();
    render();
  };
  reader.readAsText(file);
}
TAGS.forEach((tag) => {
  const label = document.createElement("label");
  label.className = "check";
  label.innerHTML = `<input id="${tag}" type="checkbox"> ${tag}`;
  $("tags").appendChild(label);
});
$("prev").onclick = () => move(-1);
$("next").onclick = () => move(1);
$("export").onclick = exportCsv;
$("import").onchange = (event) => importCsv(event.target.files[0]);
fetch("review_manifest.json").then((response) => response.json()).then((value) => { items = value; render(); });
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--seed", type=int, default=1311)
    args = parser.parse_args()

    root = Path(args.analysis_root).resolve()
    review_dir = root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    image_rows = read_csv(root / "metrics" / "image_metrics.csv")
    items = [{"review_id": f"{int(row['index']):04d}", "panel": f"../{row['panel']}"} for row in image_rows]
    random.Random(args.seed).shuffle(items)
    write_json(review_dir / "review_manifest.json", items)
    (review_dir / "index.html").write_text(HTML)
    (review_dir / "review.css").write_text(CSS)
    (review_dir / "review.js").write_text(JS.replace("__TAGS__", json.dumps(FAILURE_TAGS)))
    template_rows = [{"review_id": item["review_id"]} for item in items]
    fields = ["review_id", "reviewed", "complex_background", "uncertain", *FAILURE_TAGS, "note"]
    write_csv(review_dir / "annotations_template.csv", template_rows, fieldnames=fields)
    write_json(review_dir / "review_config.json", {"seed": args.seed, "num_images": len(items), "blinded": True})
    print(f"Built blinded reviewer: {review_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
