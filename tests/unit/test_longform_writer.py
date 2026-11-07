"""Unit tests for the longform_writer module."""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_llm.longform_writer import (
    ContentSpec,
    DraftMemory,
    LongformWriterPipeline,
    SectionSpec,
    audit_quality_gates,
    count_words,
    extract_bold_terms,
    parse_outline_response,
)


class TestLongformWriter(unittest.TestCase):

    def test_word_counter(self):
        text = "## Heading One\nThis is a simple test sentence with **bold markdown** and `inline code`."
        wc = count_words(text)
        self.assertEqual(wc, 15)
        self.assertEqual(count_words(""), 0)

    def test_extract_bold_terms(self):
        text = "In this section, we define **Gradient Descent** and compare it to **Stochastic Gradient Descent**."
        terms = extract_bold_terms(text)
        self.assertIn("Gradient Descent", terms)
        self.assertIn("Stochastic Gradient Descent", terms)

    def test_spec_and_prompt_formatting(self):
        spec = ContentSpec(
            topic="Attention Mechanisms",
            content_type="technical guide",
            audience="engineers",
            tone="rigorous and engaging",
            word_count=3500,
            purpose="Deep dive into scaled dot-product attention.",
        )
        prompt = spec.build_master_prompt()
        self.assertIn("Topic: Attention Mechanisms", prompt)
        self.assertIn("Target length: 3500 words", prompt)
        self.assertIn("Content type: technical guide", prompt)
        self.assertIn("HARD RULES (never violate)", prompt)
        self.assertIn("DEPTH (every major section must include at least 4 of these)", prompt)

    def test_parse_outline_response(self):
        raw_outline = """
## Section 1: Introduction and Context
- Word Budget: 350
- Key Points: Hook, historical background from RNNs, problem statement
- Example: Translation bottleneck in seq2seq
- Depth Elements: Mechanism, comparison, trade-offs

## Section 2: Mathematical Foundations
- Word Budget: 500
- Key Points: Scaled dot-product formula, query-key-value projections
- Example: Calculating attention scores for a 4-token sequence
- Depth Elements: Mechanism, formula derivation, concrete example

## COVERAGE CHECKLIST
- Scaled Dot-Product: Section 2
- Multi-Head Attention: Section 3
- Computational Complexity: Section 4
"""
        outline = parse_outline_response(raw_outline, target_total_words=3000)
        self.assertEqual(len(outline.sections), 2)
        self.assertEqual(outline.sections[0].title, "Introduction and Context")
        self.assertEqual(outline.sections[0].budget, 350)
        self.assertIn("seq2seq", outline.sections[0].example)
        self.assertEqual(outline.sections[1].budget, 500)
        self.assertEqual(len(outline.coverage_checklist), 3)

    def test_parse_outline_fallback(self):
        empty_outline = "Unstructured text without any section markers."
        outline = parse_outline_response(empty_outline, target_total_words=3000)
        self.assertGreaterEqual(len(outline.sections), 4)
        total_budget = sum(s.budget for s in outline.sections)
        self.assertGreaterEqual(total_budget, 2000)

    def test_quality_gates_detection(self):
        spec = ContentSpec(topic="Test Topic", word_count=50)

        # 1. Test clean text meeting length
        clean_text = "## Introduction\n\n" + "word " * 50
        qr = audit_quality_gates(clean_text, spec)
        self.assertTrue(qr.within_length_tolerance)
        self.assertEqual(qr.verify_markers_count, 0)
        self.assertEqual(len(qr.detected_fillers), 0)
        self.assertTrue(qr.passed_all_gates)

        # 2. Test forbidden filler phrases detection
        dirty_text = "## Section 1\nIn today's world, it is important to note that neural nets delve into data. " + "word " * 40
        qr_dirty = audit_quality_gates(dirty_text, spec)
        self.assertGreaterEqual(len(qr_dirty.detected_fillers), 2)
        self.assertFalse(qr_dirty.passed_all_gates)

        # 3. Test unverified [VERIFY] detection
        verify_text = "The algorithm was invented in [VERIFY: 2017] by researchers. " + "word " * 45
        qr_verify = audit_quality_gates(verify_text, spec)
        self.assertEqual(qr_verify.verify_markers_count, 1)
        self.assertFalse(qr_verify.passed_all_gates)

    def test_pipeline_orchestration_mock(self):
        """Simulate end-to-end pipeline run with a mock LLM callable."""
        call_log = []

        def mock_llm(prompt: str, temperature: float, max_tokens: int) -> str:
            call_log.append({"temperature": temperature, "max_tokens": max_tokens, "prompt_snippet": prompt[:80]})

            # Stage 1: Outline
            if "Create a detailed outline" in prompt:
                return """
## Section 1: Foundations
- Word Budget: 50
- Key Points: Basic concepts and definitions
- Example: Small matrix
- Depth Elements: Mechanism, definition

## Section 2: Architecture & Analysis
- Word Budget: 50
- Key Points: Scaling laws and trade-offs
- Example: Production latency
- Depth Elements: Trade-offs, comparison

## COVERAGE CHECKLIST
- Foundations: Section 1
- Architecture: Section 2
"""
            # Stage 2: Section writing
            if "CURRENT ASSIGNMENT" in prompt:
                # Provide text with defined term and sufficient words
                return (
                    "## Section Content\n\n"
                    "We begin by examining the core mechanism of **Transformer Architecture**. "
                    "This system processes vectors through multi-layer perceptrons and attention blocks. "
                    "In practice, the operational throughput scales with hardware tensor parallelism. "
                    "A typical example is processing batches of sequences with low latency. "
                    "However, quadratic memory complexity presents clear trade-offs under high context windows. "
                    "Next, we transition smoothly to architectural nuances.\n\n"
                    "Finally, this concludes the core analytical breakdown."
                )

            # Stage 3: Audit
            if "rigorous technical editor" in prompt:
                return "CRITIC AUDIT: 0 factual errors found. Accuracy: 9/10, Depth: 9/10. Ready for final pass."

            # Stage 4: Revision
            if "senior editor" in prompt:
                return (
                    "## Foundations\n\n"
                    "We begin by examining the core mechanism of **Transformer Architecture**. "
                    "This system processes vectors through multi-layer perceptrons and attention blocks. "
                    "In practice, the operational throughput scales with hardware tensor parallelism. "
                    "A typical example is processing batches of sequences with low latency. "
                    "However, quadratic memory complexity presents clear trade-offs under high context windows. "
                    "Next, we transition smoothly to architectural nuances.\n\n"
                    "## Architecture & Analysis\n\n"
                    "Continuing our analysis, we observe that tensor operations remain memory-bound. "
                    "We conclude with key technical takeaways and concrete next steps."
                )

            # Summarization helper
            if "Summarize the key information" in prompt:
                return "The section defines transformer architecture and highlights quadratic memory trade-offs."

            return "Mock response text."

        spec = ContentSpec(
            topic="Transformers",
            word_count=100,
            audience="engineers",
            purpose="Explain mechanism and trade-offs.",
        )

        pipeline = LongformWriterPipeline(llm_callable=mock_llm)
        progress_events = []

        def on_progress(msg, pct):
            progress_events.append((pct, msg))

        result = pipeline.generate(spec, progress_callback=on_progress, run_revision=True)

        self.assertIn("outline", result)
        self.assertIn("initial_draft", result)
        self.assertIn("audit_report", result)
        self.assertIn("final_content", result)
        self.assertIn("quality_report", result)

        self.assertEqual(len(result["outline"].sections), 2)
        self.assertGreater(len(progress_events), 0)
        self.assertTrue(result["quality_report"].total_words > 0)

        # Check that Stage 3 used cold temperature 0.2
        audit_calls = [c for c in call_log if c["temperature"] == 0.2]
        self.assertGreaterEqual(len(audit_calls), 1)

    def test_single_prompt_template_generation(self):
        spec = ContentSpec(
            topic="Deep Learning Optimization",
            word_count=2500,
        )
        pipeline = LongformWriterPipeline(llm_callable=lambda p, t, m: "")
        single_p = pipeline.generate_single_prompt_template(spec)
        self.assertIn("STEP 1 (outline)", single_p)
        self.assertIn("STEP 2 (draft)", single_p)
        self.assertIn("STEP 3 (self-review)", single_p)
        self.assertIn("STEP 4 (final)", single_p)
        self.assertIn("Target length: 2500 words", single_p)


if __name__ == "__main__":
    unittest.main()
