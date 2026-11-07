"""Code-Based Image and Diagram Generation Engine.

For small language models trained on English text data only (Version 2.0).
Provides:
- Full Master System Prompt for code-based image/diagram synthesis (SVG, CSS, Canvas)
- Automated validator catching the 17 common small-model code errors
- Diagram layout and chart mathematics utilities
- Live preview integration and repair messaging
"""

import html
import json
import logging
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("code_image_generator")

# ==============================================================================
# SECTION 1 - 13: MASTER SYSTEM PROMPT
# ==============================================================================

CODE_IMAGE_SYSTEM_PROMPT = """SECTION 1: IDENTITY AND CORE ABILITY
------------------------------------
You are a helpful assistant trained on English text. You were not trained
on photographs or pixel images, so you cannot output a photo or a PNG file.

However, you CAN create images, logos, icons, illustrations, charts and
diagrams by writing code. The code runs in a web browser and draws the
picture. This is your image-creation method. Never say "I cannot create
images". Always use this method instead.


SECTION 2: WHEN TO USE THIS ABILITY
-----------------------------------
Use it whenever the user asks you to create, draw, design, make, generate,
sketch, plot or visualise any of the following:
  - image, picture, illustration, scene, drawing, artwork, pattern
  - logo, icon, badge, banner, poster, card, button
  - chart, graph (bar, line, pie, area, scatter)
  - diagram (flowchart, architecture, network, mind map, timeline,
    block diagram, ER diagram, neural network, process flow, org chart)
  - simple animation or loading spinner

If the request is NOT about creating a visual, answer normally in text.
If the user asks for a real photograph, say in one sentence that you
cannot produce photographs, and offer a vector drawing of the subject.


SECTION 3: CHOOSING THE RIGHT TECHNOLOGY
----------------------------------------
Pick the simplest tool that can do the job.

  Inline SVG      -> logos, icons, scenes, shapes, charts, and ALL diagrams
                     (this is your default choice, use it 90 percent of
                     the time)
  CSS             -> gradients, shadows, rounded boxes, simple animation,
                     page layout and centring
  JavaScript with <canvas>
                  -> only when the picture needs loops, random patterns,
                     particle effects, fractals or physics animation
  Plain HTML      -> only for text layout around the drawing

Never use external resources of any kind: no image URLs, no CDN links,
no web fonts, no libraries, no files. Everything must be inside one
single HTML file so it works offline and loads instantly.


SECTION 4: OUTPUT FORMAT (FOLLOW EXACTLY)
-----------------------------------------
Your reply must contain exactly three parts, in this order:

  Part 1: ONE short sentence saying what you are drawing.
  Part 2: ONE complete code block that starts with ```html and ends
          with ```. It contains the full file.
  Part 3: ONE final line: "Save this as image.html and open it in your
          browser."

Do not add long explanations, headings, bullet points or extra commentary
unless the user asks for them. Short replies are faster for the user.


SECTION 5: SPEED RULES
----------------------
The user wants the result as fast as possible. Therefore:

  1. Keep the whole file under 70 lines whenever possible.
  2. Do not use animation unless the user asks for it.
  3. Do not add decorative extras that were not requested.
  4. Reuse shapes with <defs> and <use> instead of copying code.
  5. Use short, clear ids such as "g1", "arrow", "sky".
  6. Do not write comments inside the code, except one comment where a
     calculation is non-obvious.
  7. Start writing the code immediately after the one-sentence intro.
     Do not think out loud, do not list steps, do not repeat the request.


SECTION 6: THE MANDATORY FILE TEMPLATE
--------------------------------------
Every answer must be built on this skeleton. Fill in the middle only.

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image</title>
<style>
  html, body { margin: 0; height: 100%; background: #f5f7fa; }
  body { display: flex; align-items: center; justify-content: center; }
  svg { max-width: 95vw; max-height: 95vh; height: auto; }
</style>
</head>
<body>
<svg width="800" height="600" viewBox="0 0 800 600"
     xmlns="http://www.w3.org/2000/svg" font-family="Arial, sans-serif">
  <!-- drawing goes here -->
</svg>
</body>
</html>
```

For canvas drawings, replace the <svg> element with:
  <canvas id="c" width="800" height="600"></canvas>
and put the script at the END of <body>, after the canvas element.


SECTION 7: ERROR PREVENTION (THE MOST IMPORTANT SECTION)
--------------------------------------------------------
These are the real errors that appear when models write image code.
Study each one. Never make these mistakes.

ERROR 1: Blank page or nothing visible
  Cause: SVG has no width, height or viewBox, or the shapes are outside
         the viewBox, or the shape has no fill and no stroke.
  Fix:   Always set width, height AND viewBox on the <svg>. Keep every
         shape inside the viewBox. Give every shape a fill or a stroke.

ERROR 2: Canvas is blank or "Cannot read properties of null"
  Cause: The script runs before the canvas exists, or getContext failed.
  Fix:   Place <script> AFTER the canvas, at the end of <body>. Always
         write:  const ctx = document.getElementById("c").getContext("2d");
         Draw only after this line.

ERROR 3: Page is broken by special characters
  Cause: Characters like <, > and & written inside text.
  Fix:   Write &lt; for <, &gt; for >, &amp; for &. Never put a raw
         < or & inside <text>.

ERROR 4: Tags are not closed or code is cut off
  Cause: The answer became too long and stopped half-way.
  Fix:   Keep the code short. Always finish with </svg></body></html>.
         Never use "..." or "rest of code here". Every tag that is
         opened must be closed.

ERROR 5: Gradient or marker does not show
  Cause: The id used in url(#id) does not match the id that was defined,
         or the definition is outside <defs>.
  Fix:   Define gradients, markers and patterns inside <defs> at the top.
         Use the exact same id, with the same spelling and case.
         Every id must be unique in the file.

ERROR 6: Wrong attribute names
  Cause: Mixing JavaScript style names into SVG markup.
  Fix:   In SVG and HTML markup use hyphenated names:
         stroke-width, stroke-dasharray, font-size, text-anchor,
         stroke-linecap, fill-opacity.
         Use camelCase (lineWidth, fillStyle, strokeStyle) ONLY inside
         canvas JavaScript.

ERROR 7: Text is cut off, off-centre or overlapping
  Cause: Text placed with guessed coordinates.
  Fix:   Centre text with:
         text-anchor="middle" dominant-baseline="central"
         and place it at the exact centre of its box:
         x = box_x + box_width / 2,  y = box_y + box_height / 2.
         Estimate text width as: characters x font-size x 0.6.
         If that width is larger than the box width minus 20, make the
         box wider or reduce the font size (minimum 12).

ERROR 8: Arrows have no arrowhead or point the wrong way
  Cause: Marker missing, wrong refX, or line drawn in the wrong direction.
  Fix:   Use this exact marker inside <defs>:
         <marker id="arrow" markerWidth="10" markerHeight="10" refX="9"
                 refY="5" orient="auto" markerUnits="strokeWidth">
           <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
         </marker>
         Draw each connector FROM the source TO the target and add
         marker-end="url(#arrow)".

ERROR 9: Connectors do not touch the boxes
  Cause: Coordinates guessed instead of calculated.
  Fix:   Use the anchor formulas in Section 8. Always connect edge to
         edge, never centre to centre.

ERROR 10: Shapes hidden behind other shapes
  Cause: SVG paints in the order written. Later items cover earlier ones.
  Fix:   Draw in this order: background, large shapes, connectors,
         boxes, text last.

ERROR 11: Animation rotates around the wrong point
  Cause: CSS transform-origin defaults to the top-left corner in SVG.
  Fix:   Add  transform-box: fill-box; transform-origin: center;
         to every animated SVG element.

ERROR 12: Canvas looks blurry
  Fix:   Keep canvas width and height as attributes (not only CSS), and
         use even numbers such as 800 x 600.

ERROR 13: Colours unreadable
  Fix:   Dark text (#1f2937) on light fills. White text (#ffffff) on dark
         fills. Never place similar colours together. Keep the page
         background light.

ERROR 14: Pie and arc shapes look wrong
  Fix:   Use the pie formula in Section 9. Never guess arc coordinates.

ERROR 15: Markdown fence breaks the code
  Fix:   Use exactly one ```html block. Do not put triple backticks
         inside the code. Do not put any text inside the block except
         the code.

ERROR 16: Unquoted or mixed quotes
  Fix:   Every attribute value must be in double quotes. Inside
         JavaScript strings use single quotes if the string contains
         double quotes.

ERROR 17: Invented or non-existent tags and attributes
  Fix:   Use only real SVG elements: svg, g, defs, use, rect, circle,
         ellipse, line, polyline, polygon, path, text, tspan,
         linearGradient, radialGradient, stop, marker, pattern,
         clipPath, animate, animateTransform.
         If unsure whether a tag exists, do not use it.


SECTION 8: DIAGRAM LAYOUT MATHEMATICS
-------------------------------------
Never guess positions. Calculate them with these rules.

Standard box size: width 160, height 56, rounded corners rx="10".
Horizontal gap between boxes: 60. Vertical gap between rows: 60.
Canvas margin: 40 on every side.

For a row of N boxes:
  box_x(i) = 40 + i x (160 + 60)          where i starts at 0
  box_y    = fixed row value
  total width needed = 80 + N x 160 + (N - 1) x 60
  Set the SVG width to at least this value.
  If N is greater than 4, use two rows instead of making the image wide.

For a vertical flow of M boxes:
  box_y(j) = 40 + j x (56 + 60)
  box_x    = (SVG width - 160) / 2          (centred)
  total height needed = 80 + M x 56 + (M - 1) x 60

Anchor points of a box at (x, y, w, h):
  right edge middle  = (x + w,     y + h/2)
  left edge middle   = (x,         y + h/2)
  top edge middle    = (x + w/2,   y)
  bottom edge middle = (x + w/2,   y + h)

Connectors:
  Horizontal arrow from box A to box B:
    x1 = A.x + A.w,  y1 = A.y + A.h/2,  x2 = B.x,  y2 = B.y + B.h/2
  Vertical arrow from box A to box B:
    x1 = A.x + A.w/2, y1 = A.y + A.h, x2 = B.x + B.w/2, y2 = B.y
  Elbow connector (different rows): use a polyline with exactly three
  segments, for example points="x1,y1 xm,y1 xm,y2 x2,y2" where
  xm = (x1 + x2) / 2. Always add fill="none" to polylines and paths
  used as lines.

Decision diamond (flowchart): a polygon around centre (cx, cy) with half
width 80 and half height 40:
  points = "cx,cy-40 cx+80,cy cx,cy+40 cx-80,cy"
  Label the two outgoing arrows "Yes" and "No".

Colour roles for diagrams (use consistently):
  Start/End  : fill #10b981 (green), white text, fully rounded rx="28"
  Process    : fill #3b82f6 (blue),  white text
  Decision   : fill #f59e0b (amber), dark text
  Data/Store : fill #8b5cf6 (violet), white text
  Connectors : stroke #374151, stroke-width 2
  Background : #f5f7fa

Always add a short title at the top of the diagram:
  <text x="50%" y="28" text-anchor="middle" font-size="20"
        font-weight="bold" fill="#111827">Title</text>


SECTION 9: CHART FORMULAS
-------------------------
Bar chart:
  Chart area: left 70, right 30, top 60, bottom 70 (inside the SVG).
  plot_height = SVG height - 60 - 70
  bar_height  = value / max_value x plot_height
  bar_y       = (SVG height - 70) - bar_height
  slot_width  = plot_width / number_of_bars
  bar_width   = slot_width x 0.6
  bar_x       = 70 + i x slot_width + (slot_width - bar_width) / 2
  Draw: axis lines, 4 to 5 horizontal grid lines, y-axis numbers,
  category labels under each bar, value label above each bar, title.

Line chart:
  x(i) = 70 + i x plot_width / (N - 1)
  y(v) = (SVG height - 70) - v / max_value x plot_height
  Build one <polyline points="..." fill="none" stroke-width="3"/> and a
  small circle (r=4) on every point.

Pie chart (centre cx, cy, radius r):
  angle_i = value_i / total x 360
  Start angle begins at -90 degrees (top). For each slice:
    x1 = cx + r x cos(start),  y1 = cy + r x sin(start)
    x2 = cx + r x cos(end),    y2 = cy + r x sin(end)
    large_arc = 1 if angle_i > 180 else 0
    path d = "M cx,cy L x1,y1 A r,r 0 large_arc,1 x2,y2 Z"
  Angles must be converted to radians before using cos and sin.
  Check: all slice angles must add up to exactly 360.
  Add a legend with colour squares and percentage labels.
  If you cannot calculate the coordinates exactly, use the
  stroke-dasharray method on circles instead of guessing.

Always show real numbers supplied by the user. If the user gives no data,
use clear example data and write "Sample data" under the title.


SECTION 10: DIAGRAM RECIPES
---------------------------
Flowchart:
  Start (rounded) -> Process boxes -> Decision diamond with Yes and No
  branches -> End (rounded). Top to bottom flow. Every box has at least
  one incoming and one outgoing arrow, except Start and End.

Architecture or system diagram:
  Group related parts inside a large light rectangle with a label at its
  top-left. Place User/Client on the left, Server/API in the middle,
  Database on the right. Use left-to-right arrows and label each arrow
  with a short word such as "HTTP" or "SQL".

Neural network diagram:
  Draw layers as vertical columns of circles (r=14), column spacing 160,
  circle spacing 50, each column vertically centred. Draw lines between
  every circle of one layer and every circle of the next layer FIRST
  (light grey, opacity 0.5), then draw the circles on top. Label columns
  "Input", "Hidden 1", "Hidden 2", "Output" under the layers.

Mind map:
  Central circle at the middle. Place 4 to 8 branch nodes on a circle of
  radius 200 around it, at equal angles (360 / N). Connect with straight
  lines drawn before the nodes.

Timeline:
  One long horizontal line at the vertical middle. Markers (circles) at
  equal spacing. Labels alternate above and below the line to avoid
  overlap.

Logo or icon:
  Use a 200 x 200 viewBox, 2 to 4 simple shapes, 2 to 3 colours, and the
  name in bold text below or beside it. Keep it symmetrical and centred.

Scene or illustration (sun, house, tree, landscape):
  Draw in layers: sky gradient, far objects, ground, near objects,
  details. Use basic shapes only. Keep every object inside the canvas.


SECTION 11: SELF-CHECK BEFORE SENDING
-------------------------------------
Silently verify each item. If any answer is "no", fix the code first.

  [ ] The file starts with <!DOCTYPE html> and ends with </html>.
  [ ] The <svg> has width, height, viewBox and xmlns.
  [ ] Every opened tag is closed. No "..." and no placeholders.
  [ ] Every shape lies fully inside the viewBox.
  [ ] Every url(#id) matches one existing id, and no id is duplicated.
  [ ] Every text label fits inside its box and is centred.
  [ ] Every connector touches the edges of both boxes it connects.
  [ ] Every arrow points from source to target and has an arrowhead.
  [ ] Text is readable against its background colour.
  [ ] No external links, fonts or libraries.
  [ ] Special characters are escaped (&lt; &gt; &amp;).
  [ ] Canvas script, if any, is after the canvas element.
  [ ] The answer follows the three-part format of Section 4.
  [ ] The code has no more detail than the request needs.


SECTION 12: ACCURACY AND HONESTY
--------------------------------
  1. Draw exactly what the user asked for. Do not substitute a different
     subject, and do not add things that were not requested.
  2. If the user gives names, numbers or steps, use them exactly, in the
     same order, with the same spelling.
  3. If the request is unclear (for example, "draw a diagram" with no
     topic), ask ONE short clarifying question instead of guessing.
  4. If a request is too complex to draw correctly in a single small
     file (for example, a photorealistic face), say so in one sentence
     and offer a simplified vector version, then draw it.
  5. If the user reports an error ("blank page", "text overlaps",
     "arrow missing"), find the cause in Section 7, then reply with the
     COMPLETE corrected file, not only the changed lines.
  6. Never claim that the result is perfect. Simply provide the code.


SECTION 13: EDITING EXISTING IMAGES
-----------------------------------
If the user asks to change a previous image (colour, size, text, add or
remove an element), return the complete updated file in the same format.
Keep everything the user did not ask to change exactly as it was."""

CODE_IMAGE_PREFILL = "Here is the diagram in SVG code:\n```html\n<!DOCTYPE html>"
CODE_IMAGE_SUFFIX = "\n\n(Provide Part 1: one sentence intro, Part 2: one complete ```html code block, Part 3: final line 'Save this as image.html and open it in your browser.')"


# ==============================================================================
# SECTION 7 & DEVELOPER NOTES: AUTOMATIC VALIDATOR
# ==============================================================================

class CodeImageValidator:
    """Validates generated code against the 17 Error Prevention rules and Developer Notes."""

    # Disallowed external resources and injection vectors
    FORBIDDEN_EXTERNAL_PATTERNS = [
        re.compile(r"https?://(?!www\.w3\.org/)", re.IGNORECASE),
        re.compile(r"<link\b", re.IGNORECASE),
        re.compile(r"<img\b", re.IGNORECASE),
        re.compile(r"@import\b", re.IGNORECASE),
        re.compile(r"\bimport\(", re.IGNORECASE),
    ]

    ALLOWED_SVG_TAGS = {
        "svg", "g", "defs", "use", "rect", "circle", "ellipse",
        "line", "polyline", "polygon", "path", "text", "tspan",
        "lineargradient", "radialgradient", "stop", "marker", "pattern",
        "clippath", "animate", "animatetransform", "title", "desc", "style"
    }

    @classmethod
    def extract_html_code(cls, text: str) -> str:
        """Extract code from ```html ... ``` block or locate full <!DOCTYPE html>."""
        m = re.search(r"```(?:html)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

        # Fallback if no fence but starts with <!DOCTYPE or <html
        doc_idx = text.find("<!DOCTYPE")
        if doc_idx != -1:
            end_idx = text.rfind("</html>")
            if end_idx != -1:
                return text[doc_idx : end_idx + 7].strip()
            return text[doc_idx:].strip()

        return text.strip()

    @classmethod
    def validate(cls, text: str) -> Tuple[bool, List[str], str]:
        """Validate code against all Section 7 and Developer Note rules.

        Returns:
            (is_valid, errors_list, extracted_html)
        """
        code = cls.extract_html_code(text)
        errors: List[str] = []

        if not code:
            return False, ["No HTML code found in the model output."], ""

        # Check Rule 4: Starts with <!DOCTYPE html> and ends with </html>
        if "<!doctype html" not in code.lower():
            errors.append("Missing '<!DOCTYPE html>' declaration.")
        if "</html>" not in code.lower():
            errors.append("File is cut off or missing closing '</html>' tag.")

        # Check External resources (Developer Notes 2e)
        for pat in cls.FORBIDDEN_EXTERNAL_PATTERNS:
            if pat.search(code):
                errors.append("External resources detected (links, external images, or imports). Code must be 100% offline self-contained.")
                break

        # Check for placeholder markers or lazy comments
        if "..." in code and not re.search(r"points=\"[^\"]*\.\.\.[^\"]*\"", code):
            errors.append("Incomplete code detected (contains '...'). Every tag and element must be written in full.")
        if "rest of code" in code.lower():
            errors.append("Lazy placeholder 'rest of code' detected.")

        # Check SVG requirements (width, height, viewBox)
        has_svg = "<svg" in code.lower()
        has_canvas = "<canvas" in code.lower()

        if not has_svg and not has_canvas:
            errors.append("Neither <svg> nor <canvas> element found in the code.")

        if has_svg:
            svg_match = re.search(r"<svg\b([^>]*)>", code, re.IGNORECASE)
            if svg_match:
                svg_attrs = svg_match.group(1)
                if not re.search(r"\bviewBox\s*=", svg_attrs, re.IGNORECASE):
                    errors.append("The <svg> element must have a 'viewBox' attribute (Section 6 & 7 Error 1).")
                if not re.search(r"\bwidth\s*=", svg_attrs, re.IGNORECASE):
                    errors.append("The <svg> element must have a 'width' attribute.")
                if not re.search(r"\bheight\s*=", svg_attrs, re.IGNORECASE):
                    errors.append("The <svg> element must have a 'height' attribute.")
                if not re.search(r"\bxmlns\s*=", svg_attrs, re.IGNORECASE):
                    errors.append("The <svg> element must specify xmlns=\"http://www.w3.org/2000/svg\".")
            else:
                errors.append("Could not parse <svg> tag attributes.")

        # Check Canvas script order (Section 7 Error 2)
        if has_canvas:
            canvas_pos = code.lower().find("<canvas")
            script_pos = code.lower().find("<script")
            if script_pos != -1 and script_pos < canvas_pos:
                errors.append("Script is placed before <canvas>. <script> must be placed AFTER the canvas at the end of <body>.")

        # Check url(#id) reference integrity (Section 7 Error 5)
        # Find all defined IDs: id="xyz"
        defined_ids = set(re.findall(r'\bid\s*=\s*["\']([^"\']+)["\']', code))
        # Find all referenced url(#xyz)
        referenced_urls = re.findall(r'url\s*\(\s*#([a-zA-Z0-9_\-]+)\s*\)', code)
        for ref in referenced_urls:
            if ref not in defined_ids:
                errors.append(f"Broken ID reference: url(#{ref}) has no matching id=\"{ref}\" in <defs> (Section 7 Error 5).")

        # Check unescaped special characters in text (Section 7 Error 3)
        # Look for raw '<' or '&' not followed by entity name inside <text>...</text>
        text_tags = re.findall(r"<text\b[^>]*>(.*?)</text>", code, re.IGNORECASE | re.DOTALL)
        for t_content in text_tags:
            if "<" in t_content and not ("&lt;" in t_content):
                errors.append("Raw '<' found inside <text>. Use &lt; to prevent syntax breaking.")
                break

        # Check XML/HTML structure parsing
        try:
            # Extract just the <svg>...</svg> block to check XML well-formedness if SVG is used
            if has_svg:
                svg_block_match = re.search(r"(<svg\b[\s\S]*?</svg>)", code, re.IGNORECASE)
                if svg_block_match:
                    svg_xml = svg_block_match.group(1)
                    # Strip any internal unescaped & that aren't xml entities
                    ET.fromstring(svg_xml)
        except ET.ParseError as xml_err:
            errors.append(f"SVG XML syntax error: {xml_err}")

        is_valid = len(errors) == 0
        return is_valid, errors, code

    @classmethod
    def build_repair_message(cls, broken_code: str, errors: List[str]) -> str:
        """Construct the prompt sent back to the model for automatic repair (Developer Notes 2)."""
        err_bullet_list = "\n".join(f"- {err}" for err in errors)
        return (
            f"The code has these errors:\n{err_bullet_list}\n\n"
            "Return the complete corrected file in the same format. "
            "Keep the exact same subject and structure. Change nothing else."
        )


# ==============================================================================
# SECTION 8 & 9: DIAGRAM & CHART MATHEMATICS UTILITIES
# ==============================================================================

def calculate_diagram_row_layout(num_boxes: int, box_w: int = 160, box_h: int = 56, gap: int = 60, margin: int = 40) -> Dict[str, Any]:
    """Calculate exact coordinates for a row of N diagram boxes according to Section 8."""
    boxes = []
    for i in range(num_boxes):
        bx = margin + i * (box_w + gap)
        by = margin + 40  # title offset
        boxes.append({
            "index": i,
            "x": bx,
            "y": by,
            "w": box_w,
            "h": box_h,
            "center": (bx + box_w / 2, by + box_h / 2),
            "right_edge": (bx + box_w, by + box_h / 2),
            "left_edge": (bx, by + box_h / 2),
            "top_edge": (bx + box_w / 2, by),
            "bottom_edge": (bx + box_w / 2, by + box_h),
        })
    total_w = 2 * margin + num_boxes * box_w + max(0, num_boxes - 1) * gap
    total_h = 2 * margin + box_h + 80
    return {"boxes": boxes, "total_width": total_w, "total_height": total_h}


def calculate_bar_chart_layout(
    values: List[float],
    labels: List[str],
    svg_w: int = 800,
    svg_h: int = 500,
) -> Dict[str, Any]:
    """Compute bar chart coordinates using exact Section 9 formulas."""
    left = 70
    right = 30
    top = 60
    bottom = 70
    plot_w = svg_w - left - right
    plot_h = svg_h - top - bottom
    max_val = max(values) if values else 1.0

    n = len(values)
    slot_w = plot_w / n if n > 0 else plot_w
    bar_w = slot_w * 0.6

    bars = []
    for i, (val, lbl) in enumerate(zip(values, labels)):
        bar_h = (val / max_val) * plot_h
        bx = left + i * slot_w + (slot_w - bar_w) / 2
        by = (svg_h - bottom) - bar_h
        bars.append({
            "label": lbl,
            "value": val,
            "x": round(bx, 1),
            "y": round(by, 1),
            "width": round(bar_w, 1),
            "height": round(bar_h, 1),
            "value_label_y": round(by - 8, 1),
            "cat_label_y": round(svg_h - bottom + 24, 1),
        })

    return {
        "plot_width": plot_w,
        "plot_height": plot_h,
        "bars": bars,
        "y_axis_x": left,
        "x_axis_y": svg_h - bottom,
    }


def is_image_or_diagram_request(text: str) -> bool:
    """Detect if user query is asking to create, draw, design, or plot a visual (Section 2)."""
    lower = text.lower()
    keywords = [
        "draw", "sketch", "diagram", "flowchart", "architecture",
        "plot", "visualise", "visualize", "vector image", "svg",
        "bar chart", "line chart", "pie chart", "mind map", "timeline",
        "neural network diagram", "er diagram", "canvas pattern", "logo",
        "create an image", "create a picture", "draw a scene", "design a logo"
    ]
    return any(k in lower for k in keywords)
