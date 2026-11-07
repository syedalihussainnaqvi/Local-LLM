"""Unit tests for Rich Response system prompt, formatting fixes, and prefill rules."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_llm.llm_engine import (
    CLAUDE_SYSTEM_PROMPT,
    LLMEngine,
    PROMPT_A_MARKDOWN,
    PROMPT_A_PREFILL,
    PROMPT_A_SUFFIX,
    PROMPT_B_HTML,
    PROMPT_B_PREFILL,
    PROMPT_B_SUFFIX,
    RICH_FORMAT_SUFFIX,
    RICH_RESPONSE_SHORT_PROMPT,
    RICH_RESPONSE_SYSTEM_PROMPT,
    build_user_message,
    extract_topic_and_purpose,
)


class TestRichResponsePrompt(unittest.TestCase):

    def test_rich_prompt_contains_core_elements(self):
        self.assertIn("## 1. CORE STYLE", RICH_RESPONSE_SYSTEM_PROMPT)
        self.assertIn("## 2. BULLETS AND LISTS", RICH_RESPONSE_SYSTEM_PROMPT)
        self.assertIn("## 3. TABLES", RICH_RESPONSE_SYSTEM_PROMPT)
        self.assertIn("## 4. CALLOUT BOXES", RICH_RESPONSE_SYSTEM_PROMPT)
        self.assertIn("## 5. DIAGRAMS AND VISUALS", RICH_RESPONSE_SYSTEM_PROMPT)
        self.assertIn("## 6. QUIZZES AND INTERACTIVE PRACTICE", RICH_RESPONSE_SYSTEM_PROMPT)
        self.assertIn("## 7. EXPLAINING CONCEPTS", RICH_RESPONSE_SYSTEM_PROMPT)
        self.assertIn("Mermaid diagram inside a ```mermaid code block", RICH_RESPONSE_SYSTEM_PROMPT)

    def test_bullet_variety_icons(self):
        expected_markers = ["✅", "❌", "⚠️", "💡", "📌", "🔹", "👉", "1️⃣"]
        for marker in expected_markers:
            self.assertIn(marker, RICH_RESPONSE_SYSTEM_PROMPT)

    def test_claude_prompt_distinct_from_rich(self):
        self.assertNotEqual(RICH_RESPONSE_SYSTEM_PROMPT, CLAUDE_SYSTEM_PROMPT)
        self.assertTrue(len(RICH_RESPONSE_SYSTEM_PROMPT) > 500)

    def test_fix1_short_prompt(self):
        words = RICH_RESPONSE_SHORT_PROMPT.split()
        self.assertLess(len(words), 100)
        self.assertIn("Start directly with '# Title'", RICH_RESPONSE_SHORT_PROMPT)
        self.assertIn("Never write 'Sure' or 'Here is'", RICH_RESPONSE_SHORT_PROMPT)
        self.assertIn("<details><summary>Show answer</summary>", RICH_RESPONSE_SHORT_PROMPT)

    def test_fix2_format_suffix(self):
        self.assertIn("no intro sentence", RICH_FORMAT_SUFFIX)
        self.assertIn("Start with '# ' title", RICH_FORMAT_SUFFIX)
        self.assertIn("one table, emoji bullets, one tip box", RICH_FORMAT_SUFFIX)
        self.assertIn("<details> tags", RICH_FORMAT_SUFFIX)

    def test_fix3_format_prompt_prefill(self):
        engine = LLMEngine()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>system\n...<|im_end|>\n<|im_start|>assistant\n"
        engine.tokenizer = mock_tokenizer

        # Without prefill
        prompt_normal = engine.format_prompt(question="Explain testing")
        self.assertEqual(prompt_normal, "<|im_start|>system\n...<|im_end|>\n<|im_start|>assistant\n")

        # With prefill: must be appended directly to the end
        prompt_prefilled = engine.format_prompt(question="Explain testing", prefill="# ")
        self.assertTrue(prompt_prefilled.endswith("<|im_start|>assistant\n# "))

    def test_prompt_a_markdown(self):
        self.assertIn("subject-expert tutor and study-notes writer", PROMPT_A_MARKDOWN)
        self.assertIn("STEP 0: TOPIC LOCK", PROMPT_A_MARKDOWN)
        self.assertIn("ANTI-COPY RULES", PROMPT_A_MARKDOWN)
        self.assertIn("QUIZ QUALITY CONTRACT", PROMPT_A_MARKDOWN)
        self.assertIn("Never start with \"Sure\"", PROMPT_A_MARKDOWN)
        # Ensure purely neutral placeholders, no hardcoded domain example leak
        self.assertNotIn("Positive and Negative Testing", PROMPT_A_MARKDOWN)
        self.assertIn("[Name of concept from TOPIC]", PROMPT_A_MARKDOWN)
        self.assertEqual(PROMPT_A_PREFILL, "# ")
        self.assertIn("<details> answers", PROMPT_A_SUFFIX)

    def test_prompt_b_html(self):
        self.assertIn("study-notes engine", PROMPT_B_HTML)
        self.assertIn("const DATA=", PROMPT_B_HTML)
        self.assertIn(".card{border:1px solid", PROMPT_B_HTML)
        self.assertIn("[emoji] [TOPIC] title", PROMPT_B_HTML)
        self.assertEqual(PROMPT_B_PREFILL, "```html\n<!DOCTYPE html>")
        self.assertIn("Output ONLY one ```html block", PROMPT_B_SUFFIX)

    def test_build_user_message(self):
        msg = build_user_message(
            topic="Designing",
            purpose="quiz preparation",
            level="beginner",
            length="about 1500 words",
            field="graphic and UI design principles",
        )
        self.assertIn("TOPIC: Designing", msg)
        self.assertIn("FIELD / CONTEXT: graphic and UI design principles", msg)
        self.assertIn("PURPOSE: quiz preparation", msg)
        self.assertIn("LEVEL: beginner", msg)
        self.assertIn("LENGTH: about 1500 words", msg)
        self.assertIn("INSTRUCTION: Write notes ONLY about the TOPIC above.", msg)
        self.assertIn("REMINDER: the subject of every heading, example and question is: Designing", msg)

    def test_extract_topic_and_purpose(self):
        # 1. Exact problem sentence from user report
        t1, p1 = extract_topic_and_purpose("write a detailed note on designing for me quiz preparation")
        self.assertEqual(t1.lower(), "designing")
        self.assertIn("quiz", p1.lower())

        # 2. Already structured prompt
        t2, p2 = extract_topic_and_purpose("TOPIC: Machine Learning\nPURPOSE: interview preparation")
        self.assertEqual(t2, "Machine Learning")
        self.assertEqual(p2, "interview preparation")

        # 3. Simple explain pattern
        t3, p3 = extract_topic_and_purpose("explain Operating Systems for exam revision")
        self.assertEqual(t3, "Operating Systems")
        self.assertEqual(p3, "exam revision")


if __name__ == "__main__":
    unittest.main()
