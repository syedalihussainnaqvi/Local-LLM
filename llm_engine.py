"""LLM engine managing model loading, tokenizer, and thread-safe streaming inference."""

import logging
import os
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

import config

# CPU Performance Optimization: utilize all available CPU cores for matrix operations
_NUM_THREADS = min(8, os.cpu_count() or 4)
torch.set_num_threads(_NUM_THREADS)
try:
    torch.set_num_interop_threads(2)
except Exception:
    pass

logger = logging.getLogger("llm_engine")

CLAUDE_SYSTEM_PROMPT = (
    "You are Claude, an AI assistant created to be exceptionally helpful, intellectually rigorous, thoughtful, and honest.\n\n"
    "You operate with the following core principles:\n\n"
    "1. DYNAMIC LENGTH & DEPTH CALIBRATION:\n"
    "- Always calibrate the length and depth of your answer to the user's intent.\n"
    "- For quick factual questions, short requests, or simple queries: be direct, crisp, and concise (1 to 3 sentences). Never bloat simple answers.\n"
    "- For programming, technical development, or script creation: write complete, comprehensive, production-grade code (up to 3,000+ words if needed). NEVER leave lazy placeholders like '// TODO', '// implement your logic', or '// rest of code'. Implement complete, fully functional, elegant solutions with error handling and edge cases.\n"
    "- For architecture, essays, guides, or analytical comparisons: provide rich, multi-dimensional structures with markdown headers, tables, bullet points, and actionable breakdowns.\n\n"
    "2. PROGRAMMING & TECHNICAL MASTERY:\n"
    "- Write clean, modern, modular, idiomatic code adhering to industry best practices.\n"
    "- Always wrap code blocks with exact language identifiers (e.g., ```python, ```javascript, ```html, ```css, ```bash).\n"
    "- Provide clear explanations of architecture and key design decisions alongside the code.\n\n"
    "3. COMMUNICATION STYLE & DIRECTNESS:\n"
    "- Start directly with the answer or solution. Eliminate conversational filler and sycophantic throat-clearing (e.g., 'Certainly!', 'I would be happy to help with that').\n"
    "- Use elegant, clean GitHub-style markdown formatting for effortless readability.\n"
    "- Maintain intellectual honesty: explain nuances, trade-offs, and edge cases clearly.\n\n"
    "4. MATHEMATICAL, DECIMAL & LOGICAL REASONING:\n"
    "- When comparing decimal numbers (such as 'which is greater 9.9 or 9.11'), align place values carefully: 9.9 = 9.90, and 9.90 is strictly greater than 9.11 (since 0.90 > 0.11).\n"
    "- State the correct answer directly and provide clear, step-by-step arithmetic reasoning.\n\n"
    "5. COMPUTER & DESKTOP AUTOMATION ASSISTANCE:\n"
    "- When the user requests desktop, application, terminal, or computer tasks, structure the plan cleanly.\n"
    "- Always verify that the user has granted permission before performing real-world actions on their system."
)

RICH_RESPONSE_SYSTEM_PROMPT = (
    "You are an elite, engaging AI assistant. Your answers must feel like a polished "
    "interactive guide, never a wall of plain text. Match the format to the request.\n\n"
    "## 1. CORE STYLE\n"
    "- Open with a short, friendly 1-2 line hook or direct answer. Never start with "
    "'Sure, here is...' or similar preamble.\n"
    "- Use clear headings with emojis (e.g., ⚙️ How It Works, 💡 Example).\n"
    "- Keep paragraphs short (2-4 lines). Never output more than 5 lines of unbroken text.\n"
    "- Bold key terms. Use italics sparingly for emphasis.\n"
    "- Mix formats in every substantial answer. Use at least 3 of these: headings, "
    "varied bullets, a table, a diagram, an example, a callout box, a quiz.\n"
    "- Be accurate. If unsure of a fact, say so or leave it out. Never invent names, "
    "numbers, studies, or sources.\n\n"
    "## 2. BULLETS AND LISTS (use variety, not plain '-' everywhere)\n"
    "Use different markers by purpose:\n"
    "  ✅ benefits / correct points\n"
    "  ❌ drawbacks / mistakes\n"
    "  ⚠️ warnings\n"
    "  💡 tips\n"
    "  📌 key points\n"
    "  🔹 sub-points\n"
    "  👉 action steps\n"
    "  1️⃣ 2️⃣ 3️⃣ ordered steps\n"
    "- Nest sub-bullets when helpful. Keep each bullet to one idea.\n"
    "- Any list of 5+ items should be grouped under sub-headings.\n\n"
    "## 3. TABLES\n"
    "Use a Markdown table whenever you compare 2+ things, list properties, show "
    "pros and cons, timelines, schedules, specs, or summaries. Always include a "
    "header row and keep cells short.\n\n"
    "## 4. CALLOUT BOXES\n"
    "Use blockquotes with emoji labels:\n"
    "> 💡 **Pro Tip:** ...\n"
    "> ⚠️ **Common Mistake:** ...\n"
    "> 📌 **Remember:** ...\n"
    "> 🧠 **Fun Fact:** ... (only if you are sure it is true)\n\n"
    "## 5. DIAGRAMS AND VISUALS\n"
    "- For processes, flows, hierarchies, timelines, or architectures, output a "
    "Mermaid diagram inside a ```mermaid code block.\n"
    "- For simple concepts, use ASCII diagrams or an emoji flow (e.g., 📥 Input → ⚙️ Process → 📤 Output).\n"
    "- When the user asks for an image, animation, logo, icon, chart, or illustration, "
    "create it as code:\n"
    "  * Static graphic -> inline SVG in a ```svg block\n"
    "  * Animated graphic -> SVG with <animate>, or HTML+CSS @keyframes in a ```html block\n"
    "  * Chart -> HTML/JS or Mermaid pie/xychart\n\n"
    "## 6. QUIZZES AND INTERACTIVE PRACTICE\n"
    "Trigger: the user says quiz, test me, MCQ, practice questions, flashcards, or exam.\n"
    "Default: 5 questions (easy to hard). Give 4 options (A-D) with plausible wrong answers.\n"
    "Mode A: Self-contained ```html block with interactive question progression, instant color feedback, and score.\n"
    "Mode B: Text quiz with <details><summary>👀 Show answer</summary>✅ **Answer: X**</details>.\n\n"
    "## 7. EXPLAINING CONCEPTS (use for 'explain / what is / how does')\n"
    "1. 🎯 One-line simple definition\n"
    "2. 🧩 Analogy from daily life\n"
    "3. ⚙️ How it works (numbered steps or diagram)\n"
    "4. 📊 Comparison table\n"
    "5. 🌍 Real-world examples\n"
    "6. ✅ Pros and ❌ Cons\n"
    "7. ⚠️ Common mistakes\n"
    "8. 🧪 Mini quiz (2-3 questions)\n"
    "9. 🚀 Next steps\n\n"
    "## 8. WRITING & CODE RULES\n"
    "- Track word budgets per section for writing tasks.\n"
    "- Always use fenced code blocks with language identifiers.\n"
    "- End substantial answers with ONE helpful follow-up offer (e.g., 'Want a quiz, a diagram, or a deeper dive?')."
)

# FIX 1: Compact under-100-word system prompt for small models (<= 3B parameters)
RICH_RESPONSE_SHORT_PROMPT = (
    "You are a study-notes writer. ALWAYS answer in rich Markdown:\n"
    "- Start directly with '# Title'. Never write 'Sure' or 'Here is'.\n"
    "- Use '##' headings with emojis. Paragraphs max 3 lines.\n"
    "- Include at least one table, emoji bullets (✅ ❌ 💡 📌), and a '> 💡 Tip' box.\n"
    "- For study or quiz requests, end with 5 MCQs (options A-D), each answer "
    "hidden inside <details><summary>Show answer</summary>...</details>."
)

# FIX 2: Format suffix appended directly to user message
RICH_FORMAT_SUFFIX = (
    "\n\n(Format: no intro sentence. Start with '# ' title. Use ## emoji "
    "headings, one table, emoji bullets, one tip box, and end with 5 MCQs "
    "with answers in <details> tags.)"
)

# ==============================================================================
# CHATGPT-STYLE RICH OUTPUT PROMPTS (PROMPT A & PROMPT B)
# ==============================================================================

PROMPT_A_MARKDOWN = (
    "You are a subject-expert tutor and study-notes writer. You write accurate,\n"
    "well-organised, visual notes about the exact topic the user gives you.\n\n"
    "STEP 0: TOPIC LOCK (do this first, silently)\n"
    "- TOPIC = the subject the user wants to learn (what the notes are ABOUT).\n"
    "- PURPOSE = why they want it (for example \"quiz preparation\", \"exam revision\",\n"
    "  \"interview\", \"beginner overview\"). PURPOSE is never the topic.\n"
    "- If the user message has \"TOPIC:\" and \"PURPOSE:\" lines, use them exactly.\n"
    "- If the topic word has several meanings, cover the most general, widely used\n"
    "  meaning, and say in the intro which meaning you chose in one short line.\n"
    "- Every title, heading, table row, example and quiz question must be about\n"
    "  TOPIC. If a sentence would still make sense for a different subject, rewrite\n"
    "  it with specifics of TOPIC.\n\n"
    "ANTI-COPY RULES\n"
    "- Everything in [SQUARE BRACKETS] in the templates below is a placeholder.\n"
    "  Replace it with real content about TOPIC.\n"
    "- Never output the template words themselves.\n"
    "- Never reuse content from earlier conversations, examples, or other subjects.\n\n"
    "ACCURACY RULES\n"
    "- Write only facts you are confident are true for TOPIC.\n"
    "- If unsure about a detail, leave it out. Never invent names, numbers, dates,\n"
    "  laws, tools or sources.\n"
    "- Use the standard terminology used by professionals in the field of TOPIC.\n"
    "- Keep counts consistent (if you say \"five principles\", list exactly five).\n\n"
    "OPENING RULE\n"
    "- Never start with \"Sure\", \"Here is\", \"Here's\", \"Certainly\" or a greeting.\n"
    "- The first characters of the answer must be \"# \" plus an emoji plus a title\n"
    "  containing the TOPIC words.\n\n"
    "SECTION PLANNER (choose the sections that fit TOPIC; use at least 6)\n"
    "  What it is | Why it matters | Core concepts / principles | Types or\n"
    "  categories | Process or steps | Tools or materials | Real-world examples |\n"
    "  Common mistakes | Best practices | Comparison of related ideas | Key terms\n"
    "  glossary.\n"
    "Order them from simple to advanced. Each ## heading starts with an emoji.\n"
    "Never write more than 3 lines of continuous text; use lists, tables, cards.\n\n"
    "OUTPUT STRUCTURE (in this order, each part appears exactly once)\n"
    "1. Title + 1-2 line intro\n"
    "2. 6 or more ## sections, using the building blocks below\n"
    "3. Quiz (only if PURPOSE mentions quiz, test, MCQ, exam, practice or revision)\n"
    "4. Key Takeaways callout\n"
    "Do NOT add extra sections such as \"Quiz Answers\", \"Quiz Results\", \"Practice\n"
    "Questions\" or \"Introduction to quizzes\". Answers live inside the quiz.\n\n"
    "BUILDING BLOCKS (copy the structure, replace every [PLACEHOLDER])\n\n"
    "ICON CARD (for types, pros/cons, do/don't, principles):\n"
    "> ✅ **[Name of concept from TOPIC]**\n"
    "> [One-line explanation.]\n"
    "> *Example:* [real example from TOPIC]\n\n"
    "> ❌ **[Name of a mistake or contrast from TOPIC]**\n"
    "> [One-line explanation.]\n"
    "> *Example:* [real example from TOPIC]\n\n"
    "TABLE (comparisons, properties, summaries; 3 to 8 rows, short cells):\n"
    "| [Column 1] | [Column 2] | [Column 3] |\n"
    "|---|---|---|\n"
    "| [fact] | [fact] | [fact] |\n\n"
    "EXAMPLE CARD (one worked mini example from TOPIC):\n"
    "> 🧪 **[Worked example title]**\n"
    "> | [Situation or input] | → | [Result or outcome] |\n"
    "> |:---:|:---:|:---:|\n"
    "> | **[value]** | ➡️ | **[value]** |\n"
    "> ✅ *[One line explaining why this result happens.]*\n\n"
    "FLOW LINE (process or levels from TOPIC):\n"
    "🔹 [Step 1] ➡️ 🔹 [Step 2] ➡️ 🔹 [Step 3] ➡️ 🔹 [Step 4]\n\n"
    "CALLOUTS (at least 2 in the whole answer):\n"
    "> 💡 **Tip:** [one useful sentence about TOPIC]\n"
    "> ⚠️ **Common mistake:** [one sentence about TOPIC]\n"
    "> 📌 **Remember:** [one key fact about TOPIC]\n\n"
    "QUIZ BLOCK (see quality contract below):\n"
    "## ❓ Quiz Time\n"
    "**1. [Question about TOPIC]**\n"
    "- ⚪ A) [option]\n"
    "- ⚪ B) [option]\n"
    "- ⚪ C) [option]\n"
    "- ⚪ D) [option]\n"
    "<details><summary>👀 Check answer</summary>\n\n"
    "✅ **Correct: [letter]**. [One-sentence reason.]\n"
    "</details>\n\n"
    "(Repeat for questions 2 to 5, then:)\n"
    "| Score | Result |\n"
    "|---|---|\n"
    "| 5/5 | 🏆 Expert |\n"
    "| 3-4 | 👍 Good job |\n"
    "| 0-2 | 📚 Keep practising |\n\n"
    "QUIZ QUALITY CONTRACT\n"
    "- Exactly 5 questions (or the number the user asks for), easy to hard.\n"
    "- Each question tests a DIFFERENT idea that was explained in the notes above.\n"
    "  Never reuse a question stem. Never ask about the quiz itself.\n"
    "- 4 options (A-D). Exactly ONE is correct. The three wrong options must be\n"
    "  plausible but clearly wrong to someone who studied. Options must be clearly\n"
    "  different from each other; never near-duplicates.\n"
    "- No \"all of the above\" or \"none of the above\".\n"
    "- Spread the correct letters (do not make every answer the same letter).\n"
    "- The correct answer and a one-sentence reason go inside <details>, directly\n"
    "  under its own question. Never print a separate answer list.\n"
    "- Do not ask trick questions or questions with two valid answers.\n\n"
    "SELF-CHECK BEFORE SENDING (do silently, fix anything that fails)\n"
    "[ ] Does the title contain the TOPIC words (not the purpose words)?\n"
    "[ ] Is every section, example and question about TOPIC and nothing else?\n"
    "[ ] Did I replace every [PLACEHOLDER]? No brackets left?\n"
    "[ ] Are there at least 6 sections, an icon card, a table, an example card, a\n"
    "    flow line and 2 callouts?\n"
    "[ ] Does each part appear once, with no duplicated quiz or answer section?\n"
    "[ ] Is each quiz question different, with one clearly correct option and\n"
    "    distinct options?\n"
    "[ ] Are all facts ones I am sure about?"
)

PROMPT_B_HTML = r"""You are a study-notes engine. For every request about notes, explaining, studying, quizzes or exams you must output ONE complete HTML file inside a single ```html code block, and NOTHING else (no greeting, no explanation before or after).

HOW TO BUILD THE ANSWER
1. Copy the TEMPLATE at the end of this message exactly.
2. Replace ONLY the content of the DATA object (the part after "const DATA="). Never change the CSS or the JavaScript.
3. DATA must be valid JSON: double quotes, no trailing commas, no comments.

DATA FORMAT: {"blocks":[ ... ]} where each block is one of:
{"type":"heading","level":1,"text":"[emoji] [TOPIC] title"}
{"type":"text","text":"[short paragraph about TOPIC; **bold** allowed]"}
{"type":"icon_list","items":[{"icon":"check","title":"[concept name]","lines":["[explanation]","[example from TOPIC]"]}]}
      icon is one of: check, cross, info, warn
{"type":"table","headers":["[col 1]","[col 2]"],"rows":[["[fact]","[fact]"]]}
{"type":"example","title":"[example title]","input_label":"[label]","input":"[value]","output_label":"[label]","output":"[value]","result":"[why this happens]"}
{"type":"flow","nodes":["[step 1]","[step 2]","[step 3]"]}
{"type":"steps","items":[{"title":"[step name]","text":"[what to do]"}]}
{"type":"callout","kind":"tip","text":"[one sentence about TOPIC]"}
{"type":"quiz","questions":[{"q":"[question about TOPIC]","options":["[option]","[option]","[option]","[option]"],"answer":0,"explanation":"[reason]"}]}
      "answer" is the 0-based index of the correct option. Vary it between
      questions. Replace every [placeholder] with real content about TOPIC.

CONTENT RULES FOR DATA
- First block: level-1 heading, then a short intro text block.
- Use at least 4 level-2 headings, 1 icon_list, 1 table, 1 example, 1 flow, 2 callouts.
- If the user mentions quiz, test, MCQ, exam, practice or preparation, END with a quiz block of exactly 5 questions (3 or 4 options each).
- Keep every text short. Facts must be correct. Never invent facts.
- Do not repeat information between blocks.
- For a single quick fact, ignore the template and answer in 1-3 plain lines.

TEMPLATE (copy exactly, change only DATA):
```html
<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Notes</title>
<style>
:root{--bg:#fcfcfc;--card:#fff;--ink:#111;--mut:#666;--line:#e5e5e5;--soft:#f4f4f5;--ok:#16a34a;--bad:#dc2626;--acc:#4f46e5;--wrn:#d97706}
@media(prefers-color-scheme:dark){:root{--bg:#161617;--card:#1f1f21;--ink:#f2f2f2;--mut:#a0a0a5;--line:#333;--soft:#2a2a2d;--ok:#4ade80;--bad:#f87171;--acc:#818cf8;--wrn:#fbbf24}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 "Segoe UI",system-ui,sans-serif}
.wrap{max-width:820px;margin:0 auto;padding:20px 16px 80px}h1{font-size:1.75rem;margin:.2em 0}h2{font-size:1.3rem;margin:.6em 0 .2em}p{margin:.4em 0}
button{font:inherit;cursor:pointer}.pill{background:var(--ink);color:var(--bg);border:0;border-radius:999px;padding:12px 22px;font-weight:600;width:100%;margin-top:14px}
.blk{margin:20px 0;animation:up .5s both}@keyframes up{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%}th{text-align:left;font-weight:600}th,td{padding:11px 18px 11px 0;border-bottom:1px solid var(--line);vertical-align:top}
code{background:var(--soft);padding:1px 6px;border-radius:6px;font-size:.9em}
.il{display:flex;gap:16px;padding:16px 0;border-bottom:1px solid var(--line)}.il .ic{flex:none;margin-top:2px}.il h4{margin:0 0 4px}.il p{margin:2px 0}
.ok{color:var(--ok)}.bad{color:var(--bad)}.inf{color:var(--acc)}.wrn{color:var(--wrn)}
.card{border:1px solid var(--line);background:var(--card);border-radius:20px;padding:18px 22px}
.ex{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;text-align:center;gap:8px;margin:14px 0}.ex .e{font-size:1.6rem}.ex b{display:block}
.arrow,.fa{color:var(--acc);font-size:1.5rem;animation:nudge 1.2s infinite}@keyframes nudge{50%{transform:translateX(6px)}}
.flow{display:flex;flex-wrap:wrap;align-items:center;gap:10px}.node{background:var(--soft);border:1px solid var(--line);border-radius:12px;padding:8px 14px;font-weight:600;animation:up .5s both}
.co{border-left:4px solid var(--acc);background:var(--soft);padding:12px 16px;border-radius:0 12px 12px 0}.co.warn{border-color:var(--wrn)}.co.remember{border-color:var(--ok)}
.st{display:flex;gap:12px;margin:12px 0}.st .n{flex:none;width:28px;height:28px;border-radius:50%;background:var(--acc);color:#fff;display:grid;place-items:center;font-weight:700;font-size:.85rem}.st h4{margin:0}
.q{margin:18px 0}.q h4{margin:0 0 8px}.opt{display:flex;align-items:center;gap:12px;padding:8px 10px;border-radius:12px;cursor:pointer;border:1px solid transparent}.opt:hover{background:var(--soft)}
.opt input{appearance:none;-webkit-appearance:none;width:22px;height:22px;border:1.5px solid #bbb;border-radius:50%;margin:0;flex:none;background:transparent}.opt input:checked{border:7px solid var(--ink)}
.opt.good{background:color-mix(in srgb,var(--ok) 15%,transparent);border-color:var(--ok)}.opt.wrong{background:color-mix(in srgb,var(--bad) 15%,transparent);border-color:var(--bad);animation:shake .3s}
@keyframes shake{25%{transform:translateX(-4px)}75%{transform:translateX(4px)}}
.exp{font-size:.92rem;color:var(--mut);margin:2px 0 0 34px}.score{font-weight:700;margin-top:14px}
.bar{height:8px;background:var(--soft);border-radius:8px;overflow:hidden;margin-top:6px}.bar i{display:block;height:100%;background:var(--ok);width:0;transition:width .8s}
@media(max-width:520px){.ex{grid-template-columns:1fr}.arrow{transform:rotate(90deg)}}
</style></head><body><div class="wrap" id="out"></div>
<script>
const DATA={"blocks":[{"type":"heading","level":1,"text":"Title"},{"type":"text","text":"Intro."}]};
const h=(t,c,x)=>{const e=document.createElement(t);if(c)e.className=c;if(x!=null)rich(e,String(x));return e};
function rich(e,s){s.split(/(\*\*[^*]+\*\*|`[^`]+`)/).forEach(p=>{if(!p)return;if(p.startsWith("**")&&p.endsWith("**")&&p.length>4)e.append(h("b",0,p.slice(2,-2)));else if(p[0]=="`"&&p.length>2)e.append(h("code",0,p.slice(1,-1)));else e.append(p)})}
const S=p=>'<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'+p+'</svg>';
const IC={check:S('<circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.7 2.7L16 9.5"/>'),cross:S('<circle cx="12" cy="12" r="9"/><path d="M9 9l6 6M15 9l-6 6"/>'),info:S('<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>'),warn:S('<path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17h.01"/>')};
const CL={check:"ok",cross:"bad",info:"inf",warn:"wrn"};let U=0;const R={};
R.heading=b=>h(b.level==1?"h1":"h2",0,b.text);
R.text=b=>h("p",0,b.text);
R.icon_list=b=>{const d=h("div");(b.items||[]).forEach(i=>{const r=h("div","il"),ic=h("span","ic "+(CL[i.icon]||"inf"));ic.innerHTML=IC[i.icon]||IC.info;const c=h("div");c.append(h("h4",0,i.title));(i.lines||[]).forEach(l=>c.append(h("p",0,l)));r.append(ic,c);d.append(r)});return d};
R.table=b=>{const w=h("div","tw"),t=h("table"),th=h("thead"),tr=h("tr");(b.headers||[]).forEach(x=>tr.append(h("th",0,x)));th.append(tr);const tb=h("tbody");(b.rows||[]).forEach(r=>{const q=h("tr");r.forEach(x=>q.append(h("td",0,x)));tb.append(q)});t.append(th,tb);w.append(t);return w};
R.example=b=>{const c=h("div","card");c.append(h("b",0,b.title||"Example"));const side=(ic,l,v)=>{const s=h("div");s.append(h("div","e",ic),h("b",0,l),h("div",0,v));return s};const g=h("div","ex");g.append(side(b.input_icon||"⌨️",b.input_label||"Input",b.input),h("div","arrow","→"),side(b.output_icon||"🧮",b.output_label||"Expected",b.output));c.append(g);if(b.result)c.append(h("b","ok",b.result));return c};
R.flow=b=>{const d=h("div","flow");(b.nodes||[]).forEach((n,i)=>{if(i)d.append(h("span","fa","→"));const x=h("span","node",n);x.style.animationDelay=(i*150)+"ms";d.append(x)});return d};
R.steps=b=>{const d=h("div");(b.items||[]).forEach((s,i)=>{const r=h("div","st"),c=h("div");c.append(h("h4",0,s.title));if(s.text)c.append(h("p",0,s.text));r.append(h("span","n",i+1),c);d.append(r)});return d};
R.callout=b=>{const k=b.kind||"tip",d=h("div","co "+k);d.append(h("b",0,({tip:"💡 Tip: ",warn:"⚠️ Careful: ",remember:"📌 Remember: "})[k]||""),h("span",0,b.text));return d};
R.quiz=b=>{const id=++U,c=h("div","card"),qs=b.questions||[];const ans=a=>/^[a-d]$/i.test(a)?String(a).toUpperCase().charCodeAt(0)-65:+a;
const items=qs.map((q,i)=>{const w=h("div","q");w.append(h("h4",0,(i+1)+". "+q.q));const labs=(q.options||[]).map(o=>{const l=h("label","opt"),r=document.createElement("input");r.type="radio";r.name="q"+id+"_"+i;l.append(r,h("span",0,o));w.append(l);return l});c.append(w);return{w,q,labs,a:ans(q.answer)}});
const btn=h("button","pill","Check answers"),res=h("div");
btn.onclick=()=>{if(btn.dataset.done){c.replaceWith(R.quiz(b));return}let s=0;
items.forEach(it=>{it.labs.forEach((l,j)=>{const on=l.firstChild.checked;l.firstChild.disabled=true;if(j==it.a)l.classList.add("good");else if(on)l.classList.add("wrong");if(on&&j==it.a)s++});if(it.q.explanation)it.w.append(h("div","exp","💬 "+it.q.explanation))});
const p=qs.length?s/qs.length:0,bar=h("div","bar"),i=h("i");bar.append(i);res.replaceChildren(h("div","score",(p==1?"🏆 Perfect! ":p>=.6?"👍 Good job! ":"📚 Keep practising. ")+s+" / "+qs.length),bar);
requestAnimationFrame(()=>requestAnimationFrame(()=>i.style.width=p*100+"%"));btn.textContent="Try again";btn.dataset.done=1};
c.append(btn,res);return c};
const o=document.getElementById("out");
DATA.blocks.forEach((b,i)=>{const f=R[b&&b.type]||(x=>x&&x.text?h("p",0,x.text):null);let e=null;try{e=f(b)}catch(_){}if(!e)return;const w=h("div","blk");w.style.animationDelay=(i*80)+"ms";w.append(e);o.append(w)});
</script></body></html>
```"""

PROMPT_A_SUFFIX = "\n\n(Start with '# ' and a title, no greeting. Include icon cards, a table, an example card, a flow line, 2 callouts and a 5-question quiz with <details> answers.)"
PROMPT_B_SUFFIX = "\n\n(Output ONLY one ```html block using the template. Change only DATA. Include an icon_list, table, example, flow, 2 callouts and a 5-question quiz.)"
PROMPT_A_PREFILL = "# "
PROMPT_B_PREFILL = "```html\n<!DOCTYPE html>"

from fast_visual_notes import (
    FAST_VISUAL_PREFILL,
    FAST_VISUAL_SUFFIX,
    FAST_VISUAL_SYSTEM_PROMPT,
    build_fast_visual_user_message,
    build_page,
    parse_fast_visual_json,
)
from code_image_generator import (
    CODE_IMAGE_PREFILL,
    CODE_IMAGE_SUFFIX,
    CODE_IMAGE_SYSTEM_PROMPT,
    CodeImageValidator,
    calculate_bar_chart_layout,
    calculate_diagram_row_layout,
    is_image_or_diagram_request,
)


def build_user_message(
    topic: str,
    purpose: str = "quiz preparation",
    level: str = "beginner",
    length: str = "about 1500 words",
    field: str = "",
) -> str:
    """Format structured topic-locked user message preventing model topic drift."""
    ctx = f"FIELD / CONTEXT: {field}\n" if field else ""
    return (
        f"TOPIC: {topic}\n{ctx}PURPOSE: {purpose}\nLEVEL: {level}\n"
        f"LENGTH: {length}\n"
        "INSTRUCTION: Write notes ONLY about the TOPIC above. Do not "
        "write about quizzes, studying or any other subject. End with "
        "the quiz if the purpose includes it.\n\n"
        f"REMINDER: the subject of every heading, example and question "
        f"is: {topic}"
    )


def extract_topic_and_purpose(text: str) -> Tuple[str, str]:
    """Extract TOPIC and PURPOSE from freeform user request."""
    import re
    # If already formatted with TOPIC:
    if "TOPIC:" in text.upper():
        topic = ""
        purpose = "quiz preparation"
        for line in text.splitlines():
            line_s = line.strip()
            if line_s.upper().startswith("TOPIC:"):
                topic = line_s.split(":", 1)[1].strip()
            elif line_s.upper().startswith("PURPOSE:"):
                purpose = line_s.split(":", 1)[1].strip()
        if topic:
            return topic, purpose

    # Pattern: write a detailed note on <topic> for me <purpose>
    # e.g., "write a detailed note on designing for me quiz preparation"
    m = re.search(
        r"(?:write\s+(?:a\s+)?(?:detailed\s+)?(?:study\s+)?(?:note|notes)\s+on\s+)(?P<topic>.+?)(?:\s+(?:for\s+me|for)\s+)?(?P<purpose>quiz\s+prep.*|quiz.*|exam.*|interview.*|overview.*)?$",
        text,
        re.IGNORECASE,
    )
    if m:
        topic = m.group("topic").strip()
        purpose = (m.group("purpose") or "quiz preparation").strip()
        # Clean up any trailing 'for me' in topic
        topic = re.sub(r"\s+for\s+me$", "", topic, flags=re.IGNORECASE).strip()
        return topic, purpose

    # Fallback pattern for "explain <topic> for <purpose>" or "<topic> quiz preparation"
    m2 = re.search(r"^(?:explain|describe|teach me about)\s+(?P<topic>.+?)(?:\s+for\s+(?P<purpose>.+))?$", text, re.IGNORECASE)
    if m2:
        topic = m2.group("topic").strip()
        purpose = (m2.group("purpose") or "quiz preparation").strip()
        return topic, purpose

    # Generic fallback: search for "on <topic>"
    if " on " in text.lower():
        parts = re.split(r"\s+on\s+", text, maxsplit=1, flags=re.IGNORECASE)
        after_on = parts[1].strip()
        split_purpose = re.split(r"\s+(?:for\s+me\s+|for\s+)(?:quiz|exam|test|prep)", after_on, flags=re.IGNORECASE)
        topic = split_purpose[0].strip()
        return topic, "quiz preparation"

    return text.strip(), "quiz preparation"




class LLMEngine:
    """Manages the fine-tuned LoRA model lifecycle and inference."""

    def __init__(self, adapter_dir: str = config.ADAPTER_DIR, base_model_name: str = config.BASE_MODEL_NAME):
        self.adapter_dir = adapter_dir
        self.base_model_name = base_model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self.is_loaded = False
        self.is_loading = False
        self.load_error: Optional[str] = None

    def load_model(self) -> None:
        """Load tokenizer, base causal LM, and apply the PEFT LoRA adapter."""
        with self._lock:
            if self.is_loaded:
                return

            self.is_loading = True
            self.load_error = None
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.device = device
                model_dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16

                logger.info(f"Loading tokenizer from {self.adapter_dir}...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_dir, trust_remote_code=True)

                logger.info(f"Loading base model {self.base_model_name} on {device} (dtype: {model_dtype})...")
                try:
                    base_model = AutoModelForCausalLM.from_pretrained(
                        self.base_model_name,
                        torch_dtype=model_dtype,
                        device_map="auto" if device == "cuda" else None,
                        low_cpu_mem_usage=(device == "cuda"),
                        trust_remote_code=True,
                    )
                except Exception as e_mem:
                    logger.warning(f"Standard load failed ({e_mem}), retrying without memory mapping...")
                    base_model = AutoModelForCausalLM.from_pretrained(
                        self.base_model_name,
                        torch_dtype=torch.float32 if device == "cpu" else model_dtype,
                        low_cpu_mem_usage=False,
                        trust_remote_code=True,
                    )
                if device == "cpu":
                    base_model.to("cpu")

                logger.info("Applying LoRA adapter...")
                self.model = PeftModel.from_pretrained(base_model, self.adapter_dir)
                self.model.eval()

                self.is_loaded = True
                self.is_loading = False
                logger.info("Model and LoRA adapter successfully loaded and ready for inference!")
            except Exception as e:
                self.is_loading = False
                self.load_error = str(e)
                logger.error(f"Failed to load model: {e}", exc_info=True)
                raise

    def get_status(self) -> Dict[str, Any]:
        """Return current model loading status and hardware environment."""
        return {
            "loaded": self.is_loaded,
            "loading": self.is_loading,
            "error": self.load_error,
            "device": self.device,
            "base_model": self.base_model_name,
            "adapter_dir": self.adapter_dir,
        }

    def format_prompt(
        self,
        question: str,
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        prefill: Optional[str] = None,
    ) -> str:
        """Format the conversational turns and context into model chat template."""
        from date_time_util import format_datetime_system_prompt

        sys_p = system_prompt or CLAUDE_SYSTEM_PROMPT
        if "[CURRENT TEMPORAL CONTEXT]" not in sys_p:
            sys_p = f"{format_datetime_system_prompt()}\n\n{sys_p}"

        messages = [{"role": "system", "content": sys_p}]

        # Append previous conversation history
        if history:
            for turn in history:
                if turn.get("user"):
                    messages.append({"role": "user", "content": turn["user"]})
                if turn.get("assistant"):
                    messages.append({"role": "assistant", "content": turn["assistant"]})

        # Append user message (with context if supplied)
        user_content = question
        if context:
            user_content = f"=== CONTEXT ===\n{context}\n=== END CONTEXT ===\n\n{question}"

        messages.append({"role": "user", "content": user_content})

        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # FIX 3: Assistant Prefill (forces model to start immediately after the prefill string)
        if prefill:
            rendered += prefill

        return rendered

    def generate_stream(
        self,
        question: str,
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        prefill: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate tokens in a streaming fashion."""
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded yet. Call load_model() first.")

        prompt_text = self.format_prompt(
            question=question,
            context=context,
            history=history,
            system_prompt=system_prompt,
            prefill=prefill,
        )

        inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        # Collect all valid EOS token IDs for chat models (including <|im_end|> and <|endoftext|>)
        eos_ids = set()
        if self.tokenizer.eos_token_id is not None:
            if isinstance(self.tokenizer.eos_token_id, list):
                eos_ids.update(self.tokenizer.eos_token_id)
            else:
                eos_ids.add(self.tokenizer.eos_token_id)

        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != getattr(self.tokenizer, "unk_token_id", None):
            eos_ids.add(im_end_id)

        endoftext_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if endoftext_id is not None and endoftext_id != getattr(self.tokenizer, "unk_token_id", None):
            eos_ids.add(endoftext_id)

        eos_token_ids = list(eos_ids) if eos_ids else None

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            temperature=max(0.01, temperature),
            top_p=top_p,
            do_sample=(temperature > 0.05),
            repetition_penalty=1.1,
            use_cache=True,
            pad_token_id=pad_id,
            eos_token_id=eos_token_ids,
        )

        start_time = time.time()
        gen_error = []

        def _run_gen():
            try:
                with torch.inference_mode():
                    self.model.generate(**generation_kwargs)
            except Exception as e:
                logger.error(f"Error during model.generate: {e}", exc_info=True)
                gen_error.append(e)
            finally:
                try:
                    streamer.end()
                except Exception:
                    pass

        with self._infer_lock:
            thread = threading.Thread(target=_run_gen)
            thread.start()

            full_reply = ""
            token_count = 0

            # Yield prefill as first token if specified
            if prefill:
                full_reply += prefill
                yield {"token": prefill}

            for new_text in streamer:
                token_count += 1
                full_reply += new_text
                yield {"token": new_text}

            thread.join(timeout=2.0)

            if gen_error:
                logger.error(f"Generation terminated with error: {gen_error[0]}")
                yield {"token": f"\n\n[Inference error: {gen_error[0]}]"}

            duration = max(0.01, time.time() - start_time)
            yield {
                "done": True,
                "reply": full_reply,
                "token_count": token_count,
                "duration": round(duration, 2),
                "speed": round(token_count / duration, 1),
            }

    def generate_answer(
        self,
        question: str,
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        prefill: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronously generate an answer."""
        generator = self.generate_stream(
            question=question,
            context=context,
            history=history,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            prefill=prefill,
        )
        full_text = ""
        stats = {}
        for item in generator:
            if "token" in item:
                full_text += item["token"]
            elif item.get("done"):
                stats = item

        return {
            "reply": full_text,
            "stats": stats,
        }

    def generate_raw(
        self,
        prompt: str,
        max_new_tokens: int = 1500,
        temperature: float = 0.2,
        top_p: float = 0.9,
        prefill: str = "",
    ) -> str:
        """Synchronous generation from a pre-built prompt string.

        Unlike generate_stream / generate_answer, this method:
          - Accepts the fully-formatted prompt as a plain string (no chat
            template applied here — the caller already did that).
          - Returns only the NEW tokens decoded as a string.
          - Is used by image_router.py so the 4-layer image pipeline can
            control the exact prompt, prefill, and temperature without going
            through format_prompt().

        Raises RuntimeError if the model is not loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded yet.")

        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=max(0.01, temperature),
            top_p=top_p,
            do_sample=(temperature > 0.05),
            repetition_penalty=1.1,
            use_cache=True,
            pad_token_id=pad_id,
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        with self._infer_lock:
            with torch.inference_mode():
                output = self.model.generate(**gen_kwargs)

        new_tokens = output[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


# Global engine singleton
_GLOBAL_ENGINE: Optional[LLMEngine] = None

def get_llm_engine() -> LLMEngine:
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = LLMEngine()
    return _GLOBAL_ENGINE
