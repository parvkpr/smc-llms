"""
Embed results/completions.json into index.html and patch renderPanel
to show actual model output text below each prompt panel.

Run once generate_completions.py has finished.
"""
import json, re, sys

HTML = "index.html"
COMPLETIONS_FILE = "results/completions.json"

with open(COMPLETIONS_FILE) as f:
    completions = json.load(f)

# Validate all 4 models have 100 behaviors each
for model, behaviors in completions.items():
    n = len(behaviors)
    ok_direct = sum(1 for v in behaviors.values() if v.get("direct"))
    ok_best   = sum(1 for v in behaviors.values() if v.get("best"))
    print(f"{model}: {n} behaviors, {ok_direct} direct, {ok_best} best")

with open(HTML, encoding="utf-8") as f:
    html = f.read()

# ── 1. Inject COMPLETIONS variable right after "var MODELS = ..." line ────────
completions_js = "      var COMPLETIONS = " + json.dumps(completions, ensure_ascii=False) + ";\n"

target = "      var MODELS = Object.keys(EXP);"
if "var COMPLETIONS" in html:
    # Replace existing COMPLETIONS variable with updated data
    # Use split on the known surrounding anchor to avoid regex across huge JSON
    before, _, after = html.partition("      var COMPLETIONS = ")
    _, _, after = after.partition("\n      var MODELS = ")
    html = before + completions_js + "      var MODELS = " + after
    print("Replaced existing COMPLETIONS variable")
else:
    html = html.replace(target, completions_js + target, 1)
    print("Injected COMPLETIONS variable")

# ── 2. Patch renderPanel to append a model-output block ───────────────────────
OLD_RENDER = (
    "          +'<div style=\"font-size:0.72em;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;margin-bottom:0.3em;\">Prompt sent to model</div>'\n"
    "          +'<div style=\"font-size:0.8em;color:#334155;background:'+bg+';border:1px solid #e2e8f0;border-radius:5px;padding:0.65em 0.85em;line-height:1.6;font-family:monospace;white-space:pre-wrap;word-break:break-word;\">'+esc(prompt)+'</div>';\n"
    "      }"
)

NEW_RENDER = (
    "          +'<div style=\"font-size:0.72em;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;margin-bottom:0.3em;\">Prompt sent to model</div>'\n"
    "          +'<div style=\"font-size:0.8em;color:#334155;background:'+bg+';border:1px solid #e2e8f0;border-radius:5px;padding:0.65em 0.85em;line-height:1.6;font-family:monospace;white-space:pre-wrap;word-break:break-word;\">'+esc(prompt)+'</div>'\n"
    "          +(function(){\n"
    "            if(typeof COMPLETIONS==='undefined') return '';\n"
    "            var mc = COMPLETIONS[curModel];\n"
    "            if(!mc) return '';\n"
    "            var bc = mc[beh.id];\n"
    "            if(!bc) return '';\n"
    "            var txt = isBase ? bc['direct'] : bc['best'];\n"
    "            if(!txt) return '';\n"
    "            return '<div style=\"margin-top:0.85em;\">'\n"
    "              +'<div style=\"font-size:0.72em;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;margin-bottom:0.3em;\">Model output (greedy)</div>'\n"
    "              +'<div style=\"font-size:0.8em;color:#1e293b;background:#f8fafc;border:1px solid #e2e8f0;border-radius:5px;padding:0.65em 0.85em;line-height:1.6;font-family:monospace;white-space:pre-wrap;word-break:break-word;\">'+esc(txt)+'</div>'\n"
    "              +'</div>';\n"
    "          })();\n"
    "      }"
)

# The actual file uses semicolon terminating the innerHTML assignment on the prompt line.
# Adjust OLD_RENDER to match what's really in the file:
# Line ends: ...'+esc(prompt)+'</div>';   (semicolon here, closing brace on next line)
OLD_RENDER = OLD_RENDER  # already correct as written above

if "Model output (greedy)" in html:
    print("renderPanel already patched — skipping")
else:
    if OLD_RENDER not in html:
        print("ERROR: renderPanel old string not found — check whitespace")
        sys.exit(1)
    html = html.replace(OLD_RENDER, NEW_RENDER, 1)
    print("Patched renderPanel")

with open(HTML, "w", encoding="utf-8") as f:
    f.write(html)

print("Done — index.html updated.")
