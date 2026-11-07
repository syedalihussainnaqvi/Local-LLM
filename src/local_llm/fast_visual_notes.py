"""Fast Visual Study-Notes System module.

Outputs compact JSON (600-900 tokens) rendered by template.html with:
- 100% relevant icons from a strict 18-icon vocabulary
- Live responsive SVG concept maps and cycle diagrams
- Animated, clickable workflow step-throughs
- 5+ interactive MCQs marked on final submit
- Fast generation times with prefill
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("fast_visual_notes")

# ==============================================================================
# PART 1: MASTER SYSTEM PROMPT
# ==============================================================================

FAST_VISUAL_SYSTEM_PROMPT = (
    "You are a subject-expert tutor. Output ONLY one valid JSON object and nothing "
    "else: no greeting, no markdown, no code fences, no comments.\n\n"
    "TOPIC LOCK\n"
    "- TOPIC = what the notes are ABOUT. PURPOSE = why (quiz preparation, revision).\n"
    "  PURPOSE is never the topic. Use the TOPIC: and PURPOSE: lines of the user.\n"
    "- Every title, label, table cell and question must be about TOPIC only.\n"
    "- Everything in [BRACKETS] below is a placeholder; replace it with real content\n"
    "  about TOPIC. Never copy bracket text.\n"
    "- If the topic word has several meanings, use the most general meaning.\n"
    "- Facts must be correct. If unsure, leave the item out. Never invent facts.\n\n"
    "SPEED RULES\n"
    "- Total under 700 words. Every text under 25 words. No repeated ideas.\n"
    "- Use only the fields shown. Do not add extra fields.\n\n"
    "FORMAT: {\"blocks\":[ block, block, ... ]}\n"
    "BLOCK TYPES\n"
    "{\"type\":\"heading\",\"level\":1,\"text\":\"[title with TOPIC words]\"}\n"
    "{\"type\":\"heading\",\"level\":2,\"text\":\"[section]\",\"icon\":\"[optional icon name]\"}\n"
    "{\"type\":\"text\",\"text\":\"[1-2 short sentences; **bold** allowed]\"}\n"
    "{\"type\":\"radial\",\"mode\":\"map\",\"center\":\"[TOPIC]\",\"items\":[\"[4-6 core ideas of TOPIC, 1-3 words each]\"]}\n"
    "{\"type\":\"radial\",\"mode\":\"cycle\",\"center\":\"[cycle name]\",\"items\":[\"[3-6 repeating stages, 1-3 words each]\"]}\n"
    "{\"type\":\"workflow\",\"nodes\":[{\"title\":\"[step]\",\"text\":\"[what happens, max 15 words]\",\"icon\":\"[optional]\"}]}\n"
    "{\"type\":\"layers\",\"items\":[{\"title\":\"[level]\",\"text\":\"[meaning]\",\"icon\":\"[optional]\"}]}\n"
    "{\"type\":\"cards\",\"items\":[{\"title\":\"[concept]\",\"lines\":[\"[explanation]\",\"[example from TOPIC]\"],\"icon\":\"[optional]\"}]}\n"
    "{\"type\":\"table\",\"headers\":[\"[col]\",\"[col]\"],\"rows\":[[\"[fact]\",\"[fact]\"]]}\n"
    "{\"type\":\"example\",\"title\":\"[title]\",\"input_label\":\"[label]\",\"input\":\"[value]\",\"output_label\":\"[label]\",\"output\":\"[value]\",\"result\":\"[why]\"}\n"
    "{\"type\":\"callout\",\"kind\":\"tip\",\"text\":\"[one sentence]\"}   (kind: tip, warn, remember)\n"
    "{\"type\":\"quiz\",\"questions\":[{\"q\":\"[question]\",\"options\":[\"[A]\",\"[B]\",\"[C]\",\"[D]\"],\"answer\":0,\"explanation\":\"[one sentence]\"}]}\n\n"
    "REQUIRED CONTENT (in this order)\n"
    "1. heading level 1, then one text block\n"
    "2. one \"radial\" block with mode \"map\" (the live concept image of TOPIC)\n"
    "3. at least 4 level-2 sections, each followed by cards, table, text or example\n"
    "4. one \"workflow\" block ONLY if TOPIC has a real process or steps; otherwise use\n"
    "   \"layers\" (levels or hierarchy) or \"radial\" mode \"cycle\" (repeating stages).\n"
    "   Use a visual block only when it truly fits TOPIC.\n"
    "5. at least 1 table, 1 cards block, 2 callouts\n"
    "6. if PURPOSE mentions quiz, test, MCQ, exam, practice or revision: ONE quiz\n"
    "   block with 5 to 8 questions as the LAST block. No other quiz or answer\n"
    "   section anywhere.\n\n"
    "QUIZ RULES\n"
    "- Minimum 5 questions, easy to hard, each testing a DIFFERENT idea from the\n"
    "  notes. Never ask about the quiz itself.\n"
    "- Exactly 4 options. Exactly ONE correct. Wrong options are plausible but\n"
    "  clearly wrong, and all 4 options are clearly different from each other.\n"
    "- \"answer\" = 0-based index of the correct option. Vary it across questions.\n"
    "- No \"all of the above\" or \"none of the above\".\n\n"
    "ICON RULES (icons are optional; wrong or decorative icons are forbidden)\n"
    "- Allowed names and the ONLY meaning each may be used for:\n"
    "  check = correct / valid / good practice     cross = wrong / bad / avoid\n"
    "  info = definition / note                    warn = risk / caution\n"
    "  bulb = idea / tip / insight                 target = goal / purpose\n"
    "  book = theory / learning                    sliders = settings / parameters\n"
    "  code = programming / code                   layers = levels / structure\n"
    "  search = analysis / research                shield = security / protection\n"
    "  clock = time / schedule / duration          users = people / users / team\n"
    "  chart = data / metrics / results            pencil = drawing / writing / creating\n"
    "  lock = access / privacy                     database = data storage\n"
    "- Use an icon ONLY if its meaning directly matches that item. If no icon fits,\n"
    "  omit the \"icon\" field. Use at most 8 icons in the whole answer.\n"
    "- Never put emoji in any text. Icons come only from the \"icon\" field.\n\n"
    "SELF-CHECK (silently) before answering\n"
    "- Valid JSON only? No text outside it? No brackets left?\n"
    "- All content about TOPIC, none about the purpose or other subjects?\n"
    "- Does every icon match its item exactly? Remove any that do not.\n"
    "- Quiz has 5+ questions, one correct option each, varied answer indexes?"
)

FAST_VISUAL_PREFILL = '{"blocks":['
FAST_VISUAL_SUFFIX = "\n\n(Output ONLY one valid JSON object starting with {\"blocks\":[ ... ]}. No markdown, no code fences.)"

ALLOWED_ICONS = {
    "check", "cross", "info", "warn", "bulb", "target",
    "book", "sliders", "code", "layers", "search", "shield",
    "clock", "users", "chart", "pencil", "lock", "database"
}


# ==============================================================================
# PART 2: USER MESSAGE FORMAT
# ==============================================================================

def build_fast_visual_user_message(
    topic: str,
    context: str = "",
    purpose: str = "quiz preparation",
    level: str = "beginner",
) -> str:
    """Format structured topic-locked user message for fast visual JSON generation."""
    ctx = f"CONTEXT: {context}\n" if context else ""
    return (
        f"TOPIC: {topic}\n"
        f"{ctx}"
        f"PURPOSE: {purpose}\n"
        f"LEVEL: {level}\n"
        "Notes ONLY about the TOPIC. JSON only."
    )


# ==============================================================================
# PART 4: AUTOMATIC PYTHON PAGE BUILDER
# ==============================================================================

def parse_fast_visual_json(model_text: str) -> Dict[str, Any]:
    """Parse and clean JSON output from the model."""
    s = model_text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s*```$", "", s).strip()

    # If model output omitted the prefill prefix {"blocks":[
    if '"blocks"' not in s:
        if s.startswith("[") and s.endswith("]"):
            s = f'{{"blocks":{s}}}'
        else:
            s = f'{{"blocks":[{s}'
            if not s.endswith("]}"):
                if s.endswith("]"):
                    s += "}"
                elif s.endswith("}"):
                    s += "]}"
    elif s.endswith("]}") and not (s.startswith('{"blocks"') or s.startswith('{ "blocks"')):
        s = '{"blocks":[' + s

    a = s.find("{")
    z = s.rfind("}")
    if a >= 0 and z > a:
        s = s[a : z + 1]

    # Fix trailing commas: [1, 2, ] -> [1, 2]
    s = re.sub(r",\s*([}\]])", r"\1", s)

    return json.loads(s)


def build_page(
    model_text: str,
    template_path: Optional[str] = None,
    template_content: Optional[str] = None,
) -> str:
    """Build standalone HTML notes page by injecting parsed JSON data into template.html."""
    if template_content is None:
        p = template_path or "template.html"
        tpl_file = None
        candidates = [
            Path(p),
            Path(__file__).parent / p,
            Path(__file__).parent / "templates" / p,
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                tpl_file = c
                break

        if tpl_file is not None:
            tpl = tpl_file.read_text(encoding="utf-8")
        else:
            raise FileNotFoundError(f"template.html not found among candidates: {candidates}")
    else:
        tpl = template_content

    data = parse_fast_visual_json(model_text)
    safe = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return tpl.replace("/*DATA*/null", safe)
