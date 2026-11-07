"""Unit tests for the Fast Visual Study-Notes System.

Tests:
- Fast Visual System Prompt structure & strict rules
- Topic-lock user message builder
- JSON parser with prefill recovery and trailing comma fixing
- Standalone HTML page compiler (template.html injection)
- Flask endpoints (/api/presets, /template.html, /api/render-notes)
"""

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_llm.fast_visual_notes import (
    ALLOWED_ICONS,
    FAST_VISUAL_PREFILL,
    FAST_VISUAL_SUFFIX,
    FAST_VISUAL_SYSTEM_PROMPT,
    build_fast_visual_user_message,
    build_page,
    parse_fast_visual_json,
)


class TestFastVisualNotes(unittest.TestCase):

    def test_system_prompt_topic_lock(self):
        self.assertIn("TOPIC LOCK", FAST_VISUAL_SYSTEM_PROMPT)
        self.assertIn("PURPOSE is never the topic", FAST_VISUAL_SYSTEM_PROMPT)
        self.assertIn("Every title, label, table cell and question must be about TOPIC only", FAST_VISUAL_SYSTEM_PROMPT)
        self.assertIn("SPEED RULES", FAST_VISUAL_SYSTEM_PROMPT)
        self.assertIn("Total under 700 words", FAST_VISUAL_SYSTEM_PROMPT)

    def test_system_prompt_block_types(self):
        expected_blocks = [
            '"type":"heading"',
            '"type":"text"',
            '"type":"radial","mode":"map"',
            '"type":"radial","mode":"cycle"',
            '"type":"workflow"',
            '"type":"layers"',
            '"type":"cards"',
            '"type":"table"',
            '"type":"example"',
            '"type":"callout"',
            '"type":"quiz"',
        ]
        for b in expected_blocks:
            self.assertIn(b, FAST_VISUAL_SYSTEM_PROMPT)

    def test_system_prompt_quiz_and_icon_rules(self):
        self.assertIn("Minimum 5 questions", FAST_VISUAL_SYSTEM_PROMPT)
        self.assertIn("Exactly 4 options. Exactly ONE correct", FAST_VISUAL_SYSTEM_PROMPT)
        self.assertIn('"answer" = 0-based index', FAST_VISUAL_SYSTEM_PROMPT)
        self.assertIn("No \"all of the above\" or \"none of the above\"", FAST_VISUAL_SYSTEM_PROMPT)

        # 18 strict icons
        self.assertEqual(len(ALLOWED_ICONS), 18)
        for icon in ALLOWED_ICONS:
            self.assertIn(icon, FAST_VISUAL_SYSTEM_PROMPT)
        self.assertIn("Use at most 8 icons in the whole answer", FAST_VISUAL_SYSTEM_PROMPT)

    def test_prefill(self):
        self.assertEqual(FAST_VISUAL_PREFILL, '{"blocks":[')
        self.assertIn("blocks", FAST_VISUAL_SUFFIX)

    def test_user_message_builder(self):
        msg = build_fast_visual_user_message(
            topic="Designing",
            context="general design principles (graphic and UI)",
            purpose="quiz preparation",
            level="beginner",
        )
        self.assertIn("TOPIC: Designing", msg)
        self.assertIn("CONTEXT: general design principles (graphic and UI)", msg)
        self.assertIn("PURPOSE: quiz preparation", msg)
        self.assertIn("LEVEL: beginner", msg)
        self.assertIn("Notes ONLY about the TOPIC. JSON only.", msg)

    def test_parse_fast_visual_json_clean(self):
        payload = {
            "blocks": [
                {"type": "heading", "level": 1, "text": "🎨 Graphic Design Fundamentals"},
                {"type": "text", "text": "Core visual hierarchy rules."},
            ]
        }
        res = parse_fast_visual_json(json.dumps(payload))
        self.assertEqual(len(res["blocks"]), 2)
        self.assertEqual(res["blocks"][0]["text"], "🎨 Graphic Design Fundamentals")

    def test_parse_fast_visual_json_with_code_fences_and_prefill(self):
        # Model output when prefilled with {"blocks":[
        model_output = '{"type":"heading","level":1,"text":"Designing"},{"type":"text","text":"Intro."}]}'
        res = parse_fast_visual_json(model_output)
        self.assertIn("blocks", res)
        self.assertEqual(len(res["blocks"]), 2)
        self.assertEqual(res["blocks"][0]["text"], "Designing")

    def test_parse_fast_visual_json_trailing_comma_recovery(self):
        broken_json = '{"blocks":[{"type":"callout","kind":"tip","text":"Great design"},]}'
        res = parse_fast_visual_json(broken_json)
        self.assertEqual(res["blocks"][0]["kind"], "tip")

    def test_build_page(self):
        sample_json = json.dumps({
            "blocks": [
                {"type": "heading", "level": 1, "text": "Designing"},
                {"type": "text", "text": "Design is problem solving."},
                {
                    "type": "radial",
                    "mode": "map",
                    "center": "Design",
                    "items": ["Hierarchy", "Contrast", "Balance", "Alignment", "Repetition"]
                },
                {
                    "type": "workflow",
                    "nodes": [
                        {"title": "Empathize", "text": "Understand user needs", "icon": "users"},
                        {"title": "Define", "text": "State the problem", "icon": "target"},
                        {"title": "Ideate", "text": "Brainstorm concepts", "icon": "bulb"},
                        {"title": "Prototype", "text": "Create mockup", "icon": "pencil"},
                        {"title": "Test", "text": "Verify usability", "icon": "check"},
                    ]
                },
                {
                    "type": "quiz",
                    "questions": [
                        {"q": "What is contrast?", "options": ["Visual difference", "Same color", "Small text", "Random shape"], "answer": 0, "explanation": "Contrast highlights differences."}
                    ]
                }
            ]
        })

        html = build_page(sample_json)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("<title>Study Notes</title>", html)
        self.assertNotIn("/*DATA*/null", html)
        self.assertIn('"center": "Design"', html)
        self.assertIn('"type": "workflow"', html)
        self.assertIn("Submit and show all results", html)

    def test_flask_presets_and_render_routes(self):
        from local_llm.app import app
        client = app.test_client()

        # 1. /api/presets must include fast_visual
        resp = client.get("/api/presets")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("fast_visual", data)
        self.assertIn("fast_visual_prefill", data)
        self.assertEqual(data["fast_visual_prefill"], '{"blocks":[')
        self.assertIn("Output ONLY one valid JSON object", data["fast_visual"])

        # 2. /template.html must be served
        tpl_resp = client.get("/template.html")
        self.assertEqual(tpl_resp.status_code, 200)
        self.assertIn("Study Notes", tpl_resp.get_data(as_text=True))
        self.assertIn("/*DATA*/null", tpl_resp.get_data(as_text=True))

        # 3. /api/render-notes must compile JSON into standalone page
        render_resp = client.post(
            "/api/render-notes",
            json={"model_text": '{"blocks":[{"type":"heading","level":1,"text":"Test"}]}'}
        )
        self.assertEqual(render_resp.status_code, 200)
        self.assertIn("<!DOCTYPE html>", render_resp.get_data(as_text=True))
        self.assertIn('"text": "Test"', render_resp.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
