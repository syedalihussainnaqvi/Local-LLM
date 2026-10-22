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


class LLMEngine:
    """Manages the fine-tuned LoRA model lifecycle and inference."""

    def __init__(self, adapter_dir: str = config.ADAPTER_DIR, base_model_name: str = config.BASE_MODEL_NAME):
        self.adapter_dir = adapter_dir
        self.base_model_name = base_model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lock = threading.Lock()
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

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate_stream(
        self,
        question: str,
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate tokens in a streaming fashion."""
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded yet. Call load_model() first.")

        prompt_text = self.format_prompt(
            question=question,
            context=context,
            history=history,
            system_prompt=system_prompt,
        )

        inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            temperature=max(0.01, temperature),
            top_p=top_p,
            do_sample=(temperature > 0.05),
            repetition_penalty=1.05,
            use_cache=True,
            pad_token_id=pad_id,
        )

        start_time = time.time()

        def _run_gen():
            with torch.inference_mode():
                self.model.generate(**generation_kwargs)

        thread = threading.Thread(target=_run_gen)
        thread.start()

        full_reply = ""
        token_count = 0

        for new_text in streamer:
            token_count += 1
            full_reply += new_text
            yield {"token": new_text}

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


# Global engine singleton
_GLOBAL_ENGINE: Optional[LLMEngine] = None

def get_llm_engine() -> LLMEngine:
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = LLMEngine()
    return _GLOBAL_ENGINE
