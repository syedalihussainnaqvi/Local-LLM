"""Long-Form Content Generation Pipeline.

General-purpose prompt and orchestration system for long, accurate,
well-structured content. Implements the 4-stage pipeline with memory tracking,
per-section word budgeting, retry expansion loops, fact-check audit, and
post-generation quality verification.
"""

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("longform_writer")

# ==============================================================================
# PROMPT TEMPLATES (Parts 1, 2, 3)
# ==============================================================================

MASTER_SYSTEM_PROMPT_TEMPLATE = """You are a senior content writer with 15 years of experience, working together
with a domain expert (engineer / researcher / analyst) who fact-checks every
claim. Your job is to produce publication-ready writing that is accurate,
deep, well structured, and engaging.

TASK
Topic: {TOPIC}
Content type: {TYPE}
Target audience: {AUDIENCE}
Tone: {TONE}
Target length: {WORD_COUNT} words (acceptable range: 95% to 110% of target)
Purpose: {PURPOSE}
{STYLE_INSTRUCTION}

HARD RULES (never violate)

1. LENGTH
   - Hit the target length. Do not stop early.
   - If a section feels finished but is under its word budget, add depth
     (see rule 4). Never add filler.

2. ACCURACY
   - State only facts you are confident are correct.
   - If unsure about a name, number, date, formula, or technical term, do NOT
     guess. Either omit it, describe it in general terms, or mark it [VERIFY].
   - Never invent algorithms, studies, statistics, quotes, people, or sources.
   - Do not contradict yourself. Keep definitions and counts consistent across
     sections (if you say there are three types, cover all three).

3. NO REPETITION
   - Define each concept exactly once, where it belongs.
   - Later mentions may refer back briefly but must add new information.
   - Do not repeat the same idea under a different heading.
   - Do not start more than two consecutive sentences or paragraphs with the
     same word or phrase.

4. DEPTH (every major section must include at least 4 of these)
   a. A clear definition in plain language
   b. A mechanism: HOW or WHY it works, step by step
   c. A concrete real-world example with specifics
   d. A comparison with a related concept
   e. Advantages AND limitations / trade-offs
   f. Common mistakes or misconceptions
   g. A short analogy for non-experts
   h. Current state or recent developments (only if you are confident)

5. STRUCTURE
   - Flow: Hook introduction -> Background/context -> Core concepts (simple
     to advanced) -> Applications/examples -> Challenges and limitations ->
     Future outlook -> Conclusion.
   - Use a logical heading hierarchy: H1 title, H2 sections, H3 subsections.
     Sibling headings must be the same kind of thing.
   - Each section opens with one sentence stating its purpose and ends with a
     transition to the next.
   - Use paragraphs by default. Use bullet lists only for genuinely list-like
     content (steps, parallel items), never as a substitute for explanation.
     Every bullet list must be followed by a paragraph analyzing it.

6. STYLE
   - Vary sentence length. Mix short, medium, and long sentences.
   - Prefer specific, concrete wording over vague wording.
       Bad:  "improves performance"
       Good: "reduces training time by caching computed features"
   - Avoid filler phrases such as: "In today's world", "It is important to
     note", "plays a crucial role", "rapidly growing field", "delve into",
     "in conclusion, it is clear that".
   - Bold key terms on first definition only.
   - Use active voice by default.

7. OPENING AND CLOSING
   - Opening: start with a concrete hook (a relatable scenario, a surprising
     fact you are sure about, or a problem statement), not a dictionary
     definition.
   - Closing: synthesize the main ideas (do not re-list them), state the key
     takeaway, and give 3 to 5 specific next steps or further-learning
     directions.

8. OUTPUT FORMAT
   - Markdown.
   - No preamble such as "Sure, here is..." and no closing commentary.
   - Output only the content."""


STAGE1_OUTLINE_PROMPT = """You are a content strategist. Create a detailed outline for the following piece.

Topic: {TOPIC}
Audience: {AUDIENCE}
Total length: {WORD_COUNT} words
Purpose: {PURPOSE}

Requirements:
1. Produce 6 to 10 H2 sections. Give each a word budget so the total equals {WORD_COUNT}. (Example: introduction 250, core sections 350-500 each, conclusion 250.)
2. Under each H2, list 2 to 4 H3 subsections.
3. For each section list: (a) key points to cover, (b) one concrete example to include, (c) the depth elements it will use.
4. Create a COVERAGE CHECKLIST: the 10 to 15 concepts a reader would expect to find on this topic. Confirm each one appears in exactly one section.
5. Flag any concept that appears in more than one section and assign it to a single home.
6. Check that the outline has no overlap between sections and no missing fundamentals.

Format each section strictly like this:
## Section [N]: [Title]
- Word Budget: [Number]
- Key Points: [List of points]
- Example: [Concrete example]
- Depth Elements: [List of depth elements]

Followed by:
## COVERAGE CHECKLIST
- [Concept 1]: [Section assigned]
...

Output only the outline and coverage checklist."""


STAGE2_SECTION_PROMPT = """{MASTER_PROMPT}

CURRENT ASSIGNMENT
Write ONLY section {SECTION_NUM} of {TOTAL_SECTIONS}: "{SECTION_TITLE}".

Full outline (for context only; do not write other sections):
{OUTLINE_TEXT}

Points to cover in this section:
{SECTION_POINTS}

Word budget: {SECTION_BUDGET} words (minimum 90% of that number).

MEMORY (do not repeat any of this)
Concepts already defined: {DEFINED_TERMS}
Summary of previous sections:
{PREVIOUS_SUMMARIES}
Last paragraph written:
{LAST_PARAGRAPH}

INSTRUCTIONS
- Begin with a smooth transition from the last paragraph.
- Cover every point listed, with depth (mechanism, example, trade-off).
- Do not write an introduction or conclusion for the whole piece unless this section is that one.
- End with a transition sentence leading into the next section: "{NEXT_SECTION_TITLE}".
- Output only this section's text in Markdown."""


STAGE3_AUDIT_PROMPT = """You are a rigorous technical editor and fact-checker. Review the draft below.
Do NOT rewrite it. Produce an audit report only.

Draft:
{DRAFT_TEXT}

Check and report on:
1. FACTUAL ERRORS: List every claim that is incorrect, doubtful, or unverifiable. Quote the sentence, explain the problem, and give the corrected version. Pay special attention to named algorithms, formulas, dates, statistics, and definitions.
2. INTERNAL CONTRADICTIONS: Places where the text disagrees with itself (counts, categories, definitions).
3. REPETITION: Concepts, definitions, or sentences that appear more than once. Say where, and which instance to keep.
4. MISSING TOPICS: Fundamentals a reader would expect that are absent.
5. SHALLOW SECTIONS: Sections that only define without explaining mechanism, example, or trade-offs.
6. STYLE ISSUES: Repeated sentence openers, filler phrases, overlong bullet lists, weak transitions.
7. STRUCTURE ISSUES: Heading levels that are inconsistent or out of order.
8. LENGTH: Estimated word count per section versus its budget.

Finish with a prioritized fix list (Critical / Major / Minor) and a score from 0 to 10 for: accuracy, depth, structure, style, instruction-following.
Output only the audit report."""


STAGE4_REVISION_PROMPT = """You are a senior editor. Revise the draft by applying EVERY item in the audit report below.

Rules:
- Fix all factual errors using the corrections provided. If a correction is uncertain, remove the claim.
- Remove duplicated content, keeping the best instance.
- Expand shallow sections using mechanism, example, and trade-off until they meet their word budget.
- Add any missing topics in the correct section.
- Rewrite weak openers, filler phrases, and abrupt transitions.
- Do not shorten the document. Final length must be {WORD_COUNT} words (+/-5%).
- Preserve everything that was already correct and well written.

Draft:
{DRAFT_TEXT}

Audit report:
{AUDIT_REPORT}

Output only the revised full text in Markdown."""


SINGLE_PROMPT_TEMPLATE = """{MASTER_PROMPT}

Work in this exact order and show your work only where marked:

STEP 1 (outline): Write an outline with 8 sections and a word budget per section totaling {WORD_COUNT}. Show it.

STEP 2 (draft): Write the full piece section by section, meeting each budget.
After each section, add a line "[Section N: approx. X words]". Do not stop until all sections are done.

STEP 3 (self-review): Silently check for factual errors, repetition, contradictions, and missing topics. Fix them in the final version.

STEP 4 (final): Output the final, clean article without the word-count markers.

If you are about to finish before reaching {WORD_COUNT} words, add more depth to the thinnest sections instead of ending."""


# Common filler phrases prohibited by Rule 6
FORBIDDEN_FILLER_PHRASES = [
    r"\bin today's world\b",
    r"\bit is important to note\b",
    r"\bit should be noted\b",
    r"\bplays a crucial role\b",
    r"\bplays a key role\b",
    r"\brapidly growing field\b",
    r"\bdelve into\b",
    r"\bin conclusion, it is clear that\b",
]


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class ContentSpec:
    """Specification parameters for a long-form content generation task."""
    topic: str
    content_type: str = "article"
    audience: str = "professionals"
    tone: str = "technical and engaging"
    word_count: int = 3000
    purpose: str = "Provide a deep, actionable understanding of the topic with mechanisms, examples, and trade-offs."
    style_sample: Optional[str] = None

    def build_master_prompt(self) -> str:
        style_inst = f"\nSTYLE SAMPLE:\nMatch the style of this sample: {self.style_sample}\n" if self.style_sample else ""
        return MASTER_SYSTEM_PROMPT_TEMPLATE.format(
            TOPIC=self.topic,
            TYPE=self.content_type,
            AUDIENCE=self.audience,
            TONE=self.tone,
            WORD_COUNT=self.word_count,
            PURPOSE=self.purpose,
            STYLE_INSTRUCTION=style_inst,
        )


@dataclass
class SectionSpec:
    """Individual section specification parsed from the outline."""
    index: int
    title: str
    budget: int
    points: str = ""
    example: str = ""
    depth_elements: str = ""


@dataclass
class Outline:
    """Structured outline with sections and coverage checklist."""
    raw_text: str
    sections: List[SectionSpec] = field(default_factory=list)
    coverage_checklist: List[str] = field(default_factory=list)


@dataclass
class DraftMemory:
    """Cross-section memory state tracking."""
    defined_terms: Set[str] = field(default_factory=set)
    summaries: List[str] = field(default_factory=list)
    last_para: str = ""


@dataclass
class QualityReport:
    """Automated rule-based quality evaluation against the 8 hard rules."""
    total_words: int
    target_words: int
    length_ratio: float
    within_length_tolerance: bool
    verify_markers_count: int
    verify_snippets: List[str]
    detected_fillers: List[str]
    defined_terms_count: int
    section_count: int
    passed_all_gates: bool
    warnings: List[str]


# ==============================================================================
# UTILITIES
# ==============================================================================

def count_words(text: str) -> int:
    """Accurately count words in a markdown or plain text string."""
    if not text:
        return 0
    # Clean markdown headers and extra symbols for accurate word counting
    cleaned = re.sub(r"[#*_`~>\[\]\(\)]+", " ", text)
    return len(cleaned.split())


def extract_bold_terms(text: str) -> List[str]:
    """Extract terms wrapped in markdown bold (**term**) indicating new definitions."""
    matches = re.findall(r"\*\*([^*]+)\*\*", text)
    return [m.strip() for m in matches if len(m.strip()) > 1 and len(m.strip().split()) <= 4]


def parse_outline_response(raw_outline: str, target_total_words: int) -> Outline:
    """Parse raw LLM outline response into structured SectionSpec objects."""
    sections: List[SectionSpec] = []
    checklist: List[str] = []

    # Check if there are any H2 sections in the outline
    if not re.search(r"(?m)^##\s+", raw_outline):
        sec_blocks = []
    else:
        # Split on ## headings, skipping any preamble before the first ##
        parts = re.split(r"(?m)^##\s+", raw_outline)
        sec_blocks = parts[1:] if len(parts) > 1 else parts
    sec_idx = 1

    for block in sec_blocks:
        if not block.strip():
            continue
        first_line, *rest_lines = block.strip().split("\n")
        title_line = first_line.strip()
        body = "\n".join(rest_lines)

        # Check if this block is the coverage checklist
        if "coverage checklist" in title_line.lower():
            for line in rest_lines:
                line_clean = line.strip().lstrip("-*0123456789. ")
                if line_clean:
                    checklist.append(line_clean)
            continue

        # Extract budget
        budget_match = re.search(r"word\s*budget[:\s]*(\d+)", body, re.IGNORECASE)
        if budget_match:
            budget = int(budget_match.group(1))
        else:
            budget = max(250, target_total_words // 8)

        # Extract points / details
        points_match = re.search(r"key\s*points?[:\s]*(.+?)(?=(?:example|depth elements|##|\Z))", body, re.IGNORECASE | re.DOTALL)
        points = points_match.group(1).strip() if points_match else body[:300].strip()

        example_match = re.search(r"example[:\s]*(.+?)(?=(?:depth elements|key points|##|\Z))", body, re.IGNORECASE | re.DOTALL)
        example = example_match.group(1).strip() if example_match else ""

        depth_match = re.search(r"depth\s*elements?[:\s]*(.+?)(?=(?:key points|example|##|\Z))", body, re.IGNORECASE | re.DOTALL)
        depth = depth_match.group(1).strip() if depth_match else ""

        # Clean title
        clean_title = re.sub(r"^section\s*[\d:]+\s*", "", title_line, flags=re.IGNORECASE).strip()
        if not clean_title:
            clean_title = f"Section {sec_idx}"

        sections.append(
            SectionSpec(
                index=sec_idx,
                title=clean_title,
                budget=budget,
                points=points,
                example=example,
                depth_elements=depth,
            )
        )
        sec_idx += 1

    # Fallback if parsing produced no sections
    if not sections:
        avg_budget = target_total_words // 6
        fallback_titles = [
            "Introduction and Background Context",
            "Core Principles and Mechanisms",
            "Technical Deep-Dive and Architecture",
            "Real-World Implementation and Examples",
            "Trade-Offs, Limitations, and Common Pitfalls",
            "Future Directions and Conclusion",
        ]
        sections = [
            SectionSpec(
                index=i + 1,
                title=t,
                budget=avg_budget,
                points=f"In-depth analysis of {t} with concrete mechanisms and examples.",
            )
            for i, t in enumerate(fallback_titles)
        ]

    return Outline(raw_text=raw_outline, sections=sections, coverage_checklist=checklist)


def audit_quality_gates(text: str, spec: ContentSpec) -> QualityReport:
    """Run automated verification checks on the final content."""
    wc = count_words(text)
    ratio = wc / max(1, spec.word_count)
    within_length = 0.90 <= ratio <= 1.15

    # Check for [VERIFY] tags
    verify_matches = re.findall(r"\[VERIFY[^\]]*\]", text, re.IGNORECASE)

    # Check for forbidden filler phrases
    fillers_found = []
    for pattern in FORBIDDEN_FILLER_PHRASES:
        if re.search(pattern, text, re.IGNORECASE):
            fillers_found.append(pattern.replace(r"\b", ""))

    # Extract bold terms
    bold_terms = extract_bold_terms(text)

    # Check section count by H2
    h2_count = len(re.findall(r"(?m)^##\s+", text))

    warnings = []
    if not within_length:
        warnings.append(f"Word count {wc} is outside acceptable range (target: {spec.word_count}, range: {int(spec.word_count*0.95)}-{int(spec.word_count*1.10)}).")
    if verify_matches:
        warnings.append(f"Found {len(verify_matches)} unverified markers: {verify_matches[:3]}")
    if fillers_found:
        warnings.append(f"Found {len(fillers_found)} forbidden filler phrases: {fillers_found}")
    if h2_count < 3:
        warnings.append(f"Low section count: only {h2_count} H2 headings detected.")

    passed = within_length and len(verify_matches) == 0 and len(fillers_found) == 0

    return QualityReport(
        total_words=wc,
        target_words=spec.word_count,
        length_ratio=round(ratio, 3),
        within_length_tolerance=within_length,
        verify_markers_count=len(verify_matches),
        verify_snippets=verify_matches,
        detected_fillers=fillers_found,
        defined_terms_count=len(bold_terms),
        section_count=h2_count,
        passed_all_gates=passed,
        warnings=warnings,
    )


# ==============================================================================
# PIPELINE ORCHESTRATOR
# ==============================================================================

class LongformWriterPipeline:
    """Orchestrates the 4-stage generation pipeline with memory, length-control retry loop, and audit."""

    def __init__(self, llm_callable: Optional[Callable[[str, float, int], str]] = None):
        """Initialize pipeline with an LLM caller function.

        llm_callable signature: (prompt: str, temperature: float, max_tokens: int) -> str.
        If None, attempts to use the project's local LLMEngine.
        """
        self._llm = llm_callable or self._default_engine_callable

    @staticmethod
    def _default_engine_callable(prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """Call the project's fine-tuned LLMEngine."""
        from llm_engine import get_llm_engine
        engine = get_llm_engine()
        if not engine.is_loaded:
            engine.load_model()
        res = engine.generate_answer(
            question=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return res["reply"]

    def stage1_create_outline(self, spec: ContentSpec) -> Outline:
        """Stage 1: Generate structured outline with word budgets and coverage checklist."""
        prompt = STAGE1_OUTLINE_PROMPT.format(
            TOPIC=spec.topic,
            AUDIENCE=spec.audience,
            WORD_COUNT=spec.word_count,
            PURPOSE=spec.purpose,
        )
        raw_outline = self._llm(prompt, 0.6, 2048)
        return parse_outline_response(raw_outline, spec.word_count)

    def stage2_write_section(
        self,
        spec: ContentSpec,
        section: SectionSpec,
        total_sections: int,
        outline_text: str,
        memory: DraftMemory,
        next_section_title: str,
        max_retries: int = 3,
    ) -> str:
        """Stage 2: Write a single section, enforcing word budget via retry expansion loop."""
        master_prompt = spec.build_master_prompt()
        points_to_cover = section.points
        if section.example:
            points_to_cover += f"\nConcrete Example to include: {section.example}"
        if section.depth_elements:
            points_to_cover += f"\nDepth Elements: {section.depth_elements}"

        defined_terms_str = ", ".join(sorted(memory.defined_terms)) if memory.defined_terms else "None yet."
        prev_summaries_str = "\n".join(memory.summaries) if memory.summaries else "First section."
        last_para_str = memory.last_para if memory.last_para else "First section: start with a compelling hook."

        target_tokens = max(512, int(section.budget * 1.6) + 200)

        current_text = ""
        for attempt in range(max_retries):
            prompt = STAGE2_SECTION_PROMPT.format(
                MASTER_PROMPT=master_prompt,
                SECTION_NUM=section.index,
                TOTAL_SECTIONS=total_sections,
                SECTION_TITLE=section.title,
                OUTLINE_TEXT=outline_text,
                SECTION_POINTS=points_to_cover,
                SECTION_BUDGET=section.budget,
                DEFINED_TERMS=defined_terms_str,
                PREVIOUS_SUMMARIES=prev_summaries_str,
                LAST_PARAGRAPH=last_para_str,
                NEXT_SECTION_TITLE=next_section_title,
            )

            current_text = self._llm(prompt, 0.65, target_tokens)
            words = count_words(current_text)

            # Check if section reaches at least 90% of budget
            if words >= int(section.budget * 0.90):
                logger.info(f"Section {section.index} reached budget: {words}/{section.budget} words.")
                break

            logger.warning(
                f"Section {section.index} attempt {attempt + 1} under budget: {words}/{section.budget} words. Retrying with expansion directives..."
            )
            points_to_cover += (
                f"\n[EXPANSION REQUIRED: The draft was only {words} words. Expand by adding detailed step-by-step mechanisms, "
                f"an in-depth concrete case study, comparative trade-offs, and common failure modes until you reach {section.budget} words.]"
            )

        return current_text.strip()

    def stage3_critic_and_factcheck(self, draft_text: str) -> str:
        """Stage 3: Run cold temperature fact-check and critique pass (temperature 0.2)."""
        prompt = STAGE3_AUDIT_PROMPT.format(DRAFT_TEXT=draft_text)
        audit_report = self._llm(prompt, 0.2, 3072)
        return audit_report.strip()

    def stage4_revision(self, draft_text: str, audit_report: str, spec: ContentSpec) -> str:
        """Stage 4: Senior editorial revision incorporating the audit findings."""
        prompt = STAGE4_REVISION_PROMPT.format(
            WORD_COUNT=spec.word_count,
            DRAFT_TEXT=draft_text,
            AUDIT_REPORT=audit_report,
        )
        revised = self._llm(prompt, 0.5, int(spec.word_count * 1.6) + 500)
        return revised.strip()

    def summarize_section(self, section_text: str) -> str:
        """Generate a 2-sentence summary of a section for memory tracking."""
        prompt = f"Summarize the key information and takeaways of the following section in exactly 2 clear sentences:\n\n{section_text[:2000]}"
        return self._llm(prompt, 0.3, 200).strip()

    def generate(
        self,
        spec: ContentSpec,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        run_revision: bool = True,
    ) -> Dict[str, Any]:
        """Execute full 4-stage pipeline.

        Returns:
            Dict containing:
                - outline: Outline object
                - initial_draft: string
                - audit_report: string
                - final_content: string
                - quality_report: QualityReport object
        """
        def report(msg: str, pct: float):
            logger.info(f"[{pct:.0f}%] {msg}")
            if progress_callback:
                progress_callback(msg, pct)

        # Stage 1: Outline
        report("Stage 1/4: Generating outline with word budgets and coverage checklist...", 10.0)
        outline = self.stage1_create_outline(spec)
        total_sections = len(outline.sections)
        logger.info(f"Generated outline with {total_sections} sections.")

        # Stage 2: Section by section writing
        report(f"Stage 2/4: Drafting {total_sections} sections with memory & length enforcement...", 25.0)
        memory = DraftMemory()
        draft_sections: List[str] = []

        for i, sec in enumerate(outline.sections):
            next_title = outline.sections[i + 1].title if i + 1 < total_sections else "Conclusion & Final Next Steps"
            pct = 25.0 + (50.0 * (i / total_sections))
            report(f"Drafting section {sec.index}/{total_sections}: {sec.title} (Budget: {sec.budget} words)...", pct)

            sec_text = self.stage2_write_section(
                spec=spec,
                section=sec,
                total_sections=total_sections,
                outline_text=outline.raw_text,
                memory=memory,
                next_section_title=next_title,
            )
            draft_sections.append(sec_text)

            # Update memory
            new_terms = extract_bold_terms(sec_text)
            memory.defined_terms.update(new_terms)

            summary = self.summarize_section(sec_text)
            memory.summaries.append(f"Section {sec.index} ({sec.title}): {summary}")

            paragraphs = [p.strip() for p in sec_text.split("\n\n") if p.strip()]
            if paragraphs:
                memory.last_para = paragraphs[-1]

        initial_draft = "\n\n".join(draft_sections)

        # Stage 3: Audit pass
        report("Stage 3/4: Executing fact-check and editorial audit pass (temp 0.2)...", 80.0)
        audit_report = self.stage3_critic_and_factcheck(initial_draft)

        # Stage 4: Revision
        if run_revision:
            report("Stage 4/4: Applying editorial audit revisions and finalizing content...", 90.0)
            final_content = self.stage4_revision(initial_draft, audit_report, spec)
        else:
            final_content = initial_draft

        # Run automated quality checks
        report("Finalizing: Running quality gate verification...", 98.0)
        quality_report = audit_quality_gates(final_content, spec)
        report(f"Complete! Generated {quality_report.total_words} words (Target: {spec.word_count}).", 100.0)

        return {
            "spec": spec,
            "outline": outline,
            "initial_draft": initial_draft,
            "audit_report": audit_report,
            "final_content": final_content,
            "quality_report": quality_report,
        }

    def generate_single_prompt_template(self, spec: ContentSpec) -> str:
        """Produce the Part 3 single-prompt template for manual or single-turn chat usage."""
        master = spec.build_master_prompt()
        return SINGLE_PROMPT_TEMPLATE.format(
            MASTER_PROMPT=master,
            WORD_COUNT=spec.word_count,
        )


# ==============================================================================
# CLI INTERFACE
# ==============================================================================

def main():
    """Command-line interface to generate long-form articles."""
    parser = argparse.ArgumentParser(description="Long-Form Content Generation Pipeline")
    parser.add_argument("--topic", type=str, required=True, help="Topic of the article/document")
    parser.add_argument("--words", type=int, default=2500, help="Target word count (e.g. 2500)")
    parser.add_argument("--audience", type=str, default="professionals", help="Target audience")
    parser.add_argument("--tone", type=str, default="technical, engaging, authoritative", help="Writing tone")
    parser.add_argument("--type", type=str, default="article", help="Content type (article, report, guide, etc.)")
    parser.add_argument("--purpose", type=str, default="", help="What reader should understand afterward")
    parser.add_argument("--output", type=str, default="", help="Path to save markdown output")
    parser.add_argument("--no-revision", action="store_true", help="Skip stage 4 revision pass")
    parser.add_argument("--export-single-prompt", action="store_true", help="Print single-turn prompt only")

    args = parser.parse_args()

    purpose = args.purpose or f"Provide comprehensive mastery and practical implementation details of {args.topic}."
    spec = ContentSpec(
        topic=args.topic,
        content_type=args.type,
        audience=args.audience,
        tone=args.tone,
        word_count=args.words,
        purpose=purpose,
    )

    pipeline = LongformWriterPipeline()

    if args.export_single_prompt:
        prompt_text = pipeline.generate_single_prompt_template(spec)
        print(prompt_text)
        return

    print(f"\n🚀 Starting Long-Form Pipeline for '{spec.topic}' (~{spec.word_count} words)...")

    def progress(msg: str, pct: float):
        print(f"[{pct:3.0f}%] {msg}")

    result = pipeline.generate(spec, progress_callback=progress, run_revision=not args.no_revision)

    final_text = result["final_content"]
    qr: QualityReport = result["quality_report"]

    print("\n" + "=" * 60)
    print("QUALITY GATE EVALUATION")
    print("=" * 60)
    print(f"Target Words:       {qr.target_words}")
    print(f"Achieved Words:     {qr.total_words} ({qr.length_ratio * 100:.1f}%)")
    print(f"Tolerance Pass:     {'✅ YES' if qr.within_length_tolerance else '❌ NO'}")
    print(f"Verify Markers:     {qr.verify_markers_count}")
    print(f"Forbidden Fillers:  {len(qr.detected_fillers)}")
    print(f"H2 Sections:        {qr.section_count}")
    print(f"Overall Gate Pass:  {'✅ PASSED' if qr.passed_all_gates else '⚠️ WARNINGS PRESENT'}")
    if qr.warnings:
        for w in qr.warnings:
            print(f"  - Warning: {w}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(final_text)
        print(f"\n💾 Saved full article to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("PREVIEW (First 500 words)")
        print("=" * 60)
        preview = " ".join(final_text.split()[:500]) + "..."
        print(preview)


if __name__ == "__main__":
    main()
