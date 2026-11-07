"""image_router.py
Forces a small language model to create images as HTML/SVG code instead of
refusing. Works with the project's shared LLM engine (no second model load).

4 layers so a refusal can never reach the user:
  1. INTENT DETECTION   - code (not the model) decides if the user wants an image
  2. PREFILL            - reply pre-started with "```html", so the model cannot
                          begin with "I'm sorry..."
  3. FEW-SHOT EXAMPLE   - model sees one correct example (a car) first
  4. VALIDATE + FALLBACK- bad code is rejected, retried with a repair prompt,
                          and finally replaced by a built-in template.
                          An image is ALWAYS produced.

Integration:
    Called from app.py inside the /api/chat handler when the image router
    detects an image request.  Uses the project's shared LlmEngine so the
    model is loaded only once.

Standalone test:
    python image_router.py
"""

import logging
import re
import webbrowser
from html.parser import HTMLParser
from typing import Optional, Tuple

logger = logging.getLogger("image_router")

MAX_RETRIES = 2          # total generation attempts before fallback template
OUTPUT_FILE  = "image.html"

# --------------------------------------------------------------------------
# 1. INTENT DETECTION  (done in Python, model never gets to refuse)
# --------------------------------------------------------------------------
_IMAGE_WORDS = (
    r"image|picture|photo|drawing|draw|sketch|illustration|logo|icon|banner|"
    r"poster|chart|graph|diagram|flowchart|mind ?map|timeline|painting|"
    r"design|visuali[sz]e|animation|pattern|scene"
)
_ACTION_WORDS = (
    r"create|make|generate|draw|design|produce|build|show|give|paint|"
    r"sketch|plot|visuali[sz]e"
)

_RE_IMAGE  = re.compile(_IMAGE_WORDS,  re.IGNORECASE)
_RE_ACTION = re.compile(_ACTION_WORDS, re.IGNORECASE)


def wants_image(text: str) -> bool:
    """Return True when the user is asking for a visual to be generated."""
    t = text.strip()
    # "draw ..." as the first word is always an image request
    if re.match(r"^draw\b", t, re.IGNORECASE):
        return True
    return bool(_RE_IMAGE.search(t)) and bool(_RE_ACTION.search(t))


# --------------------------------------------------------------------------
# 2. SYSTEM PROMPT  (short, decisive, fits even tiny context windows)
# --------------------------------------------------------------------------
IMAGE_SYSTEM_PROMPT = (
    "You create images by writing code. "
    "You always answer image requests with one complete self-contained HTML "
    "file using inline SVG (or canvas if needed). "
    "Never refuse. Never use external links. Close every tag."
)

# --------------------------------------------------------------------------
# 3. FEW-SHOT EXAMPLE  (model sees one correct answer before it writes)
# --------------------------------------------------------------------------
_CAR_SVG = """\
<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Car</title>
<style>html,body{margin:0;height:100%;display:flex;align-items:center;
justify-content:center;background:#f5f7fa}</style>
</head><body>
<svg width="400" height="250" viewBox="0 0 400 250"
     xmlns="http://www.w3.org/2000/svg">
  <rect width="400" height="250" fill="#e0f2fe"/>
  <rect y="190" width="400" height="60" fill="#9ca3af"/>
  <path d="M30 170 L30 140 Q30 125 50 122 L110 118 L145 80
            L260 80 L300 118 L350 125 Q370 128 370 150 L370 170 Z"
        fill="#ef4444"/>
  <polygon points="120,115 150,88 200,88 200,115" fill="#bae6fd"/>
  <polygon points="210,115 210,88 255,88 285,115" fill="#bae6fd"/>
  <circle cx="100" cy="175" r="28" fill="#1f2937"/>
  <circle cx="100" cy="175" r="12" fill="#9ca3af"/>
  <circle cx="300" cy="175" r="28" fill="#1f2937"/>
  <circle cx="300" cy="175" r="12" fill="#9ca3af"/>
</svg>
</body></html>"""

FEW_SHOT = [
    {"role": "user",      "content": "Create an image of a car"},
    {"role": "assistant", "content": (
        "I'll draw a car using SVG.\n```html\n" + _CAR_SVG +
        "\n```\nSave this as image.html and open it in your browser."
    )},
]

# Prefill: model CANNOT start with a refusal — it already started writing code
PREFILL = "I'll draw this using SVG.\n```html\n<!DOCTYPE html>\n"

# --------------------------------------------------------------------------
# 4. VALIDATION  (catches errors before the user ever sees them)
# --------------------------------------------------------------------------
_VOID_TAGS = {
    "meta", "link", "br", "hr", "img", "input",
    "stop", "rect", "circle", "ellipse", "line",
    "polyline", "polygon", "path", "use", "animate",
}


class _TagChecker(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack: list = []
        self.errors: list = []

    def handle_starttag(self, tag, attrs):
        if tag not in _VOID_TAGS:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        pass  # self-closing <rect ... /> — fine as-is

    def handle_endtag(self, tag):
        if tag in _VOID_TAGS:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}>")
        else:
            self.stack.pop()


def validate_image_code(code: str) -> Tuple[bool, str]:
    """Return (is_valid, message) for a block of HTML/SVG image code.

    Covers every rule in the image_router spec:
      - File completeness   (no </html> → cut off)
      - SVG or canvas present
      - viewBox on <svg>
      - No external links / resources
      - url(#id) references match defined ids
      - Tag balance check via HTMLParser
    """
    low = code.lower()

    if "</html>" not in low:
        return False, "file is cut off — no </html>"
    if "<svg" not in low and "<canvas" not in low:
        return False, "no <svg> or <canvas> element found"
    if "<svg" in low and "viewbox" not in low:
        return False, "svg has no viewBox attribute"
    if re.search(r"https?://(?!www\.w3\.org)", low):
        return False, "external URL found — must be 100% offline"
    if "<img" in low or "@import" in low or ("<link" in low and "stylesheet" in low):
        return False, "external resource (<img>, @import, or <link stylesheet>) found"

    # url(#id) integrity
    for ref in re.findall(r"url\(#([\w-]+)\)", code):
        if f'id="{ref}"' not in code and f"id='{ref}'" not in code:
            return False, f"url(#{ref}) has no matching id in the file"

    # Tag balance
    checker = _TagChecker()
    try:
        checker.feed(code)
    except Exception as exc:
        return False, f"parse error: {exc}"
    if checker.errors:
        return False, checker.errors[0]
    if checker.stack:
        return False, f"unclosed tag <{checker.stack[-1]}>"

    return True, "ok"


# --------------------------------------------------------------------------
# 5. FALLBACK TEMPLATES  (used when the model fails MAX_RETRIES times)
# --------------------------------------------------------------------------
def _fallback_html(request: str) -> str:
    """Return a guaranteed-valid HTML file.

    Tries to match the topic from common keywords; otherwise renders a
    gradient card with the user's request text as a label.
    """
    t = request.lower()

    if "car" in t:
        return _CAR_SVG

    # Generic gradient card with safe-escaped label
    label = re.sub(r"[<>&\"']", "", request)[:50] or "Image"
    return (
        '<!DOCTYPE html>\n'
        '<html lang="en"><head><meta charset="UTF-8"><title>Image</title>\n'
        '<style>html,body{margin:0;height:100%;display:flex;align-items:center;'
        'justify-content:center;background:#f5f7fa}</style>\n'
        '</head><body>\n'
        '<svg width="500" height="300" viewBox="0 0 500 300" '
        'xmlns="http://www.w3.org/2000/svg" font-family="Arial,sans-serif">\n'
        '  <defs>\n'
        '    <linearGradient id="g1" x1="0" y1="0" x2="1" y2="1">\n'
        '      <stop offset="0" stop-color="#3b82f6"/>\n'
        '      <stop offset="1" stop-color="#8b5cf6"/>\n'
        '    </linearGradient>\n'
        '  </defs>\n'
        '  <rect x="40" y="40" width="420" height="220" rx="20" fill="url(#g1)"/>\n'
        f'  <text x="250" y="150" text-anchor="middle" dominant-baseline="central" '
        f'font-size="26" fill="#ffffff">{label}</text>\n'
        '</svg>\n'
        '</body></html>'
    )


# --------------------------------------------------------------------------
# 6. PROMPT BUILDER
# --------------------------------------------------------------------------
def build_image_prompt(tokenizer, user_request: str, error: str = "") -> str:
    """Build a fully-formatted chat prompt with few-shot example and prefill.

    The prefill string is appended at the end so the model is already
    "inside" the code block and cannot output a refusal opener.
    """
    request = user_request
    if error:
        request = (
            f"{user_request}\n\n"
            f"(The previous code had this error: {error}. "
            "Return the complete corrected file, same subject, nothing removed.)"
        )

    messages = (
        [{"role": "system", "content": IMAGE_SYSTEM_PROMPT}]
        + FEW_SHOT
        + [{"role": "user", "content": request}]
    )

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Model has no chat template → manual assembly
        text = IMAGE_SYSTEM_PROMPT + "\n\n"
        for m in messages[1:]:
            text += f"{m['role'].capitalize()}: {m['content']}\n\n"
        text += "Assistant: "

    return text + PREFILL   # <-- prefill removes any chance of refusal


def extract_html_from_output(raw: str) -> str:
    """Pull the HTML file out of the raw model output.

    The model's output begins right after the prefill, so the full code
    is:  <!DOCTYPE html>\\n + raw_output (up to the closing ``` fence).
    """
    full = "<!DOCTYPE html>\n" + raw
    # Stop at closing code fence if present
    full = full.split("```")[0]
    return full.strip()


# --------------------------------------------------------------------------
# 7. INTEGRATION WITH PROJECT LLM ENGINE (no extra model load)
# --------------------------------------------------------------------------

def generate_image_with_engine(engine, user_request: str) -> str:
    """Run the 4-layer image pipeline using the shared LlmEngine instance.

    Args:
        engine: LlmEngine instance (already loaded in app.py).
        user_request: Raw user text requesting the image.

    Returns:
        A validated, self-contained HTML string.  Never raises.
    """
    error = ""
    for attempt in range(MAX_RETRIES + 1):
        prompt = build_image_prompt(engine.tokenizer, user_request, error)
        try:
            # Use the engine's low-level generate so we can pass our custom prompt
            # and keep temperature very low for code reliability.
            raw_output = engine.generate_raw(
                prompt=prompt,
                max_new_tokens=1500,
                temperature=0.2,
                top_p=0.9,
                prefill=PREFILL,
            )
        except Exception as exc:
            logger.warning("image generation attempt %d failed: %s", attempt + 1, exc)
            error = str(exc)
            continue

        code = extract_html_from_output(raw_output)
        ok, message = validate_image_code(code)
        logger.info("image attempt %d: %s", attempt + 1, message)

        if ok:
            return code
        error = message

    logger.warning("all %d image attempts failed, using built-in fallback", MAX_RETRIES + 1)
    return _fallback_html(user_request)


def route_image_request(
    engine,
    user_request: str,
    save_path: Optional[str] = None,
) -> dict:
    """High-level entry point called from app.py.

    Returns a dict:
        {
            "type": "image",
            "html": "<complete html string>",
            "saved_to": "<path or None>",
        }
    """
    html = generate_image_with_engine(engine, user_request)

    saved_to = None
    if save_path:
        try:
            with open(save_path, "w", encoding="utf-8") as fh:
                fh.write(html)
            saved_to = save_path
        except OSError as exc:
            logger.error("could not write image file %s: %s", save_path, exc)

    return {"type": "image", "html": html, "saved_to": saved_to}


# --------------------------------------------------------------------------
# 8. STANDALONE TEST  (python image_router.py)
# --------------------------------------------------------------------------
def _standalone_generate(user_request: str) -> str:
    """Run the pipeline without the shared engine (loads model directly)."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from . import config
        model_path = getattr(config, "MODEL_PATH", None) or getattr(config, "MODEL_ID", None)
        if not model_path:
            raise RuntimeError("MODEL_PATH / MODEL_ID not set in config.py")

        print(f"  loading model from {model_path} …")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
    except Exception as exc:
        print(f"  model load failed ({exc}), using fallback template")
        return _fallback_html(user_request)

    import torch
    error = ""
    for attempt in range(MAX_RETRIES + 1):
        prompt = build_image_prompt(tokenizer, user_request, error)
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=1500,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        code = extract_html_from_output(raw)
        ok, message = validate_image_code(code)
        print(f"  attempt {attempt + 1}: {message}")
        if ok:
            return code
        error = message

    print("  model failed validation — using built-in fallback template")
    return _fallback_html(user_request)


def main():
    logging.basicConfig(level=logging.INFO)
    print("Image Router — standalone test")
    print("Type an image request (e.g. 'draw a sunset'). Type 'exit' to quit.\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"exit", "quit", ""}:
            break

        if wants_image(user):
            print("  → image request detected, generating…")
            code = _standalone_generate(user)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
                fh.write(code)
            print(f"  Image saved to: {OUTPUT_FILE}")
            webbrowser.open(OUTPUT_FILE)
        else:
            # Pass-through for non-image requests
            print("(non-image request — your normal model reply goes here)")


if __name__ == "__main__":
    main()
