# !/usr/bin/env python3
"""Build a static HTML page from the Planck teaching guide markdown.

Math notation:
  Display  math -- delimited by \\[ ... \\] Inline math -- delimited by \\( ...
  \\)

MathJax  3  is loaded from the jsDelivr CDN and renders all LaTeX at page load.
Math  regions are extracted before any HTML escaping so that backslashes, angle
brackets, and other LaTeX syntax are preserved verbatim.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


# ── Math extraction ───────────────────────────────────────────────────────
#
# Pull  every  \[...\]  and \(...\) region out of the markdown text and replace
# each  with  a  short  alphanumeric  token  before any HTML processing. Tokens
# contain  only [A-Z0-9] so html.escape() leaves them untouched. After the full
# HTML  body  is assembled, substitute the tokens back with the original LaTeX,
# which MathJax typesets in the browser.

def _extract_math(text: str) -> tuple[str, dict[str, str]]:
    """Replace math regions with placeholder tokens.

    Returns the modified text and a mapping {token: raw_latex}.
    """
    store: dict[str, str] = {}
    counter = [0]

    def _token() -> str:
        tok = f"XMATHX{counter[0]}XMATHX"
        counter[0] += 1
        return tok

    # Display math: \[ ... \] (may span multiple lines)
    def _sub_display(m: re.Match) -> str:
        tok = _token()
        store[tok] = m.group(0)
        return tok

    text = re.sub(r"\\\[.*?\\\]", _sub_display, text, flags=re.DOTALL)

    # Inline math: \( ... \)
    def _sub_inline(m: re.Match) -> str:
        tok = _token()
        store[tok] = m.group(0)
        return tok

    text = re.sub(r"\\\(.*?\\\)", _sub_inline, text, flags=re.DOTALL)

    return text, store


def _restore_math(html_text: str, store: dict[str, str]) -> str:
    """Put math back, wrapped in a <span> so MathJax can locate it."""
    for tok, latex in store.items():
        if latex.startswith(r"\["):
            replacement = (
                f'<span class="math-display">{latex}</span>'
            )
        else:
            replacement = (
                f'<span class="math-inline">{latex}</span>'
            )
        html_text = html_text.replace(tok, replacement)
    return html_text


# ── Inline Markdown formatting ────────────────────────────────────────────

def inline_format(text: str) -> str:
    """Apply inline markdown (code, bold, italic, links)."""
    text = html.escape(text)
    text = re.sub(
        r"!\[([^\]]*)\]\(([^)]+)\)",
        lambda m: (
            f'<img src="{html.escape(m.group(2), quote=True)}" '
            f'alt="{html.escape(m.group(1), quote=True)}">'
        ),
        text,
    )
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text
    )
    return text


# ── Markdown renderer ─────────────────────────────────────────────────────

TocEntry = tuple[int, str, str]


def render_markdown(
    markdown_text: str,
) -> tuple[str, list[TocEntry]]:
    """Convert Markdown (math tokens already substituted) to HTML.

    Returns  (html_body,  toc)  where  each  toc entry is (level, heading_text,
    anchor).
    """
    lines = markdown_text.splitlines()
    output: list[str] = []
    toc: list[TocEntry] = []
    in_code = False
    in_list = False
    in_ordered_list = False
    in_table = False
    pending_header: list[str] | None = None  # first row before separator
    paragraph: list[str] = []

    def slugify(t: str) -> str:
        # Strip  math placeholder tokens before building the URL anchor so that
        # headings   containing   inline   math   don't  produce  XMATHX…XMATHX
        # fragments.
        t = re.sub(r"XMATHX\d+XMATHX", "", t)
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", t.strip().lower()).strip("-")
        return slug or "section"

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            joined = " ".join(
                part.strip() for part in paragraph if part.strip()
            )
            if joined:
                output.append(f"<p>{inline_format(joined)}</p>")
            paragraph = []

    def close_list() -> None:
        nonlocal in_list, in_ordered_list
        if in_list:
            output.append("</ul>")
            in_list = False
        if in_ordered_list:
            output.append("</ol>")
            in_ordered_list = False

    def close_table() -> None:
        nonlocal in_table, pending_header
        if in_table:
            output.append("</tbody></table></div>")
            in_table = False
        if pending_header is not None:
            # Headerless single row — emit as plain table
            output.append('<div class="table-wrap"><table><tbody>')
            row = "".join(
                f"<td>{inline_format(c)}</td>" for c in pending_header
            )
            output.append(f"<tr>{row}</tr>")
            output.append("</tbody></table></div>")
            pending_header = None

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        # ── Fenced code block ─────────────────────────────────────────
        if line.startswith("```"):
            flush_paragraph()
            close_list()
            if not in_code:
                in_code = True
                code_lang = line[3:].strip()
                cls = (
                    f' class="language-{html.escape(code_lang)}"'
                    if code_lang else ""
                )
                output.append(f"<pre><code{cls}>")
            else:
                in_code = False
                output.append("</code></pre>")
            continue

        if in_code:
            output.append(html.escape(line))
            continue

        # ── Blank line ────────────────────────────────────────────────
        if not line.strip():
            flush_paragraph()
            close_list()
            close_table()
            continue

        # ── Headings ──────────────────────────────────────────────────
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            flush_paragraph()
            close_list()
            close_table()
            level = len(heading.group(1))
            text = heading.group(2).strip()
            anchor = slugify(text)
            toc.append((level, text, anchor))
            tag = inline_format(text)
            output.append(
                f'<h{level} id="{anchor}">{tag}</h{level}>'
            )
            continue

        # ── Horizontal rule ───────────────────────────────────────────
        if re.match(r"^---+\s*$", line):
            flush_paragraph()
            close_list()
            close_table()
            output.append("<hr>")
            continue

        # ── Unordered list ────────────────────────────────────────────
        if re.match(r"^- ", line):
            flush_paragraph()
            close_table()
            if in_ordered_list:
                output.append("</ol>")
                in_ordered_list = False
            if not in_list:
                output.append("<ul>")
                in_list = True
            output.append(
                f"<li>{inline_format(line[2:].strip())}</li>"
            )
            continue

        # ── Ordered list ──────────────────────────────────────────────
        m_ol = re.match(r"^(\d+)\.\s+(.*)", line)
        if m_ol:
            flush_paragraph()
            close_table()
            if in_list:
                output.append("</ul>")
                in_list = False
            if not in_ordered_list:
                output.append("<ol>")
                in_ordered_list = True
            output.append(
                f"<li>{inline_format(m_ol.group(2).strip())}</li>"
            )
            continue

        # ── List item continuation (indented line) ────────────────────
        if (in_list or in_ordered_list) and re.match(r"^ {2,}", line):
            flush_paragraph()
            close_table()
            if output and output[-1].endswith("</li>"):
                inner = output[-1][4:-5]  # strip <li> and </li>
                output[-1] = f"<li>{inner} {inline_format(line.strip())}</li>"
            continue

        # ── Table row ─────────────────────────────────────────────────
        if line.startswith("|") and line.endswith("|"):
            flush_paragraph()
            close_list()
            cells = [c.strip() for c in line.strip("|").split("|")]

            # Separator row (|---|---|): emit buffered header, open tbody
            if all(re.fullmatch(r"[-: ]+", c) for c in cells):
                if pending_header is not None:
                    output.append('<div class="table-wrap"><table>')
                    output.append("<thead><tr>")
                    for c in pending_header:
                        output.append(
                            f"<th>{inline_format(c)}</th>"
                        )
                    output.append("</tr></thead>")
                    output.append("<tbody>")
                    pending_header = None
                    in_table = True
                continue

            if in_table:
                # Body row
                row = "".join(
                    f"<td>{inline_format(c)}</td>" for c in cells
                )
                output.append(f"<tr>{row}</tr>")
            else:
                # First data row — buffer it; next line may be separator
                pending_header = cells
            continue

        # Non-table line: close any open table
        close_table()
        paragraph.append(line)

    flush_paragraph()
    close_list()
    close_table()

    return "\n".join(output), toc


# ── Table of contents ─────────────────────────────────────────────────────

def build_toc(toc: list[TocEntry]) -> str:
    items: list[str] = []
    for level, text, anchor in toc:
        if level > 3:
            continue
        cls = f"toc-level-{level}"
        label = html.escape(text)
        items.append(
            f'<li class="{cls}">'
            f'<a href="#{anchor}">{label}</a></li>'
        )
    return "<ul>\n" + "\n".join(items) + "\n</ul>"


# ── HTML page template ────────────────────────────────────────────────────

def page_template(title: str, toc_html: str, content_html: str) -> str:
    et = html.escape(title)
    # CSS  lines  that exceed 79 chars are intentional inside the HTML template
    # string and cannot be wrapped without breaking the CSS.
    return f"""<!DOCTYPE html>
<html lang="en"> <head>
  <meta  charset="utf-8">  <meta  name="viewport"  content="width=device-width,
  initial-scale=1"> <title>{et}</title>

  <!--  MathJax  3:  render  \\(  ...  \\)  inline  and \\[ ... \\] display -->
  <script>
    MathJax = {{
      tex: {{
        inlineMath:  [['\\\\(',  '\\\\)']],  displayMath: [['\\\\[', '\\\\]']],
        packages: {{'[+]': ['ams']}}, tags: 'none'
      }}, options: {{
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
      }}, startup: {{ typeset: true }}
    }};
  </script> <script async
    src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml-full.js">
  </script>

  <style>
    :root {{
      --bg:  #f6f1e8;  --panel:  #fffdf8;  --ink:  #1d2b36;  --muted:  #566472;
      --accent:  #9a3d22; --rule: #dbcdb9; --code-bg: #f2ece2; --shadow: 0 18px
      60px rgba(42, 33, 20, 0.08);
    }} * {{ box-sizing: border-box; }} body {{
      margin:  0; font-family: Georgia, "Iowan Old Style", "Palatino Linotype",
      serif; color: var(--ink);
      background:
        radial-gradient(
          circle at top left, rgba(154,61,34,0.08), transparent 22rem
        ), linear-gradient(180deg, #fbf7f0 0%, var(--bg) 100%);
      line-height: 1.7;
    }} .page {{
      max-width: 1280px; margin: 0 auto; padding: 32px 20px 56px;
    }} .hero {{
      background: linear-gradient(
        135deg, rgba(154,61,34,0.95), rgba(123,53,31,0.92)
      );  color:  #fff9f2; border-radius: 24px; padding: 28px 34px; box-shadow:
      var(--shadow); margin-bottom: 24px;
    }} .hero h1 {{
      margin: 0 0 8px; font-size: clamp(2rem, 3vw, 3.2rem); line-height: 1.1;
    }} .hero p {{
      margin:  0;  max-width:  100%;  color: rgba(255,249,242,0.92); font-size:
      1.05rem;
    }} .layout {{
      display:  grid;  grid-template-columns:  300px minmax(0, 1fr); gap: 24px;
      align-items: start;
    }} .sidebar, .content {{
      background:  var(--panel);  border: 1px solid var(--rule); border-radius:
      22px; box-shadow: var(--shadow);
    }} .sidebar {{
      position:  sticky; top: 20px; max-height: calc(100vh - 40px); overflow-y:
      auto; padding: 22px 20px;
    }} .sidebar h2 {{
      margin:   0   0   12px;  font-size:  0.9rem;  text-transform:  uppercase;
      letter-spacing: 0.09em; color: var(--accent);
    }}  .sidebar  ul {{ list-style: none; padding: 0; margin: 0; }} .sidebar li
    {{ margin: 0; padding: 0; }} .sidebar a {{
      display:  block;  padding:  4px  0; color: var(--muted); text-decoration:
      none; font-size: 0.9rem; line-height: 1.4;
    }} .sidebar a:hover {{ color: var(--accent); }} .toc-level-1 a {{
      font-weight:   600;   padding-top:  7px;  color:  var(--ink);  font-size:
      0.93rem;
    }} .toc-level-2 a {{ padding-left: 12px; }} .toc-level-3 a {{ padding-left:
    24px;  font-size:  0.85rem;  }} .content {{ padding: 36px 40px; }} .content
    h1, .content h2, .content h3, .content h4 {{
      line-height: 1.2; color: #13202b; scroll-margin-top: 24px;
    }} .content h1 {{
      font-size:   2.1rem;   padding-bottom:  10px;  border-bottom:  1px  solid
      var(--rule); margin-top: 0;
    }} .content h2 {{
      font-size:    1.55rem;    margin-top:    2.8rem;   padding-bottom:   4px;
      border-bottom: 1px solid var(--rule);
    }} .content h3 {{
      font-size: 1.18rem; margin-top: 2rem; color: var(--accent);
    }} .content h4 {{
      font-size:   1rem;   margin-top:   1.5rem;   text-transform:   uppercase;
      letter-spacing: 0.06em; color: var(--muted);
    }} .content hr {{
      border: none; border-top: 1px solid var(--rule); margin: 2.5rem 0;
    }} .content p, .content li {{
      font-size: 1.02rem; text-align: justify; hyphens: auto;
    }}  .content  ul,  .content  ol  {{ padding-left: 1.6rem; }} .content li {{
    margin-bottom: 0.25rem; }} .content code {{
      background:  var(--code-bg);  border:  1px  solid #e0d7c9; border-radius:
      6px;  padding:  0.08rem  0.38rem; font-family: "SFMono-Regular", "Menlo",
      "Consolas", monospace; font-size: 0.91em;
    }} .content pre {{
      background:  #201a17;  color: #f8efe5; border-radius: 14px; padding: 20px
      22px; overflow-x: auto; line-height: 1.5;
    }} .content pre code {{
      background:   transparent;   border:   0;  padding:  0;  color:  inherit;
      font-size: 0.93rem;
    }} .content table {{
      width:  100%;  border-collapse:  collapse;  margin:  1.4rem 0; font-size:
      0.95rem;
    }} .content th {{
      border:  1px  solid  var(--rule);  padding:  10px 14px; text-align: left;
      background:  rgba(154,61,34,0.08);  font-weight: 600; font-size: 0.93rem;
      white-space: nowrap;
    }} .content td {{
      border: 1px solid var(--rule); padding: 10px 14px; vertical-align: top;
    }} .content tr:nth-child(even) td {{
      background: rgba(219,205,185,0.18);
    }} .content a {{ color: var(--accent); }} .content img {{
      display:  block;  max-width:  100%;  height:  auto;  margin: 1.2rem auto;
      border:  1px solid var(--rule); border-radius: 16px; background: #fffdf8;
      box-shadow: 0 10px 28px rgba(42, 33, 20, 0.06); padding: 10px;
    }} .math-display {{
      display: block; overflow-x: auto; padding: 0.5rem 0; text-align: center;
    }}  .math-inline  {{ white-space: nowrap; }} /* ── Scrollable table wrapper
    ─────────────────────────────── */ .table-wrap {{
      overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 1.4rem 0;
    }}  .table-wrap  table  {{ margin: 0; min-width: 480px; }} /* ── Mobile TOC
    toggle button ─────────────────────────────── */ .toc-toggle {{
      display:  none;  width:  100%; background: var(--accent); color: #fff9f2;
      border:  none;  border-radius:  12px;  padding:  10px  16px; font-family:
      inherit;   font-size:   0.95rem;   font-weight:   600;  cursor:  pointer;
      text-align: left; margin-bottom: 12px;
    }}  /* ── Tablet breakpoint (≤ 980 px) ────────────────────────── */ @media
    (max-width: 980px) {{
      .layout  {{  grid-template-columns: 1fr; }} .sidebar {{ position: static;
      max-height:  none;  }}  .content {{ padding: 24px 18px; }} .toc-toggle {{
      display: block; }} .sidebar-nav {{ display: none; }} .sidebar-nav.open {{
      display: block; }} .sidebar h2 {{ display: none; }}
    }}  /* ── Phone breakpoint (≤ 600 px) ─────────────────────────── */ @media
    (max-width: 600px) {{
      .page  {{  padding:  12px  10px  40px;  }}  .hero  {{ padding: 18px 18px;
      border-radius:  16px;  }}  .hero  h1  {{  font-size:  clamp(1.6rem,  7vw,
      2.4rem);  }}  .hero p {{ font-size: 0.95rem; }} .content {{ padding: 18px
      14px;  border-radius:  16px; }} .sidebar {{ border-radius: 16px; padding:
      16px 14px; }} .content p, .content li {{
        text-align: left; -webkit-hyphens: none; hyphens: none;
      }} .content h1 {{ font-size: 1.7rem; }} .content h2 {{ font-size: 1.3rem;
      margin-top: 2rem; }} .content h3 {{ font-size: 1.1rem; }} .content pre {{
      padding: 14px 14px; border-radius: 10px; }} .sidebar a {{ padding: 7px 0;
      font-size:   0.97rem;   }}  .toc-level-2  a  {{  padding-left:  14px;  }}
      .toc-level-3 a {{ padding-left: 26px; font-size: 0.9rem; }}
    }}
  </style>
</head> <body>
  <div class="page">
    <section class="hero">
      <h1>{et}</h1>  <p>A  teaching-first,  theory-linked  walkthrough  of  the
      Planck codebase
         and  its  quantum  chemistry  algorithms  — from Gaussian integrals to
         CASSCF and vibrational analysis.</p>
    </section> <div class="layout">
      <aside class="sidebar">
        <button             class="toc-toggle"            aria-expanded="false"
        aria-controls="toc-nav">
          Contents &#9662;
        </button> <h2>Contents</h2> <nav id="toc-nav" class="sidebar-nav">
          {toc_html}
        </nav>
      </aside> <main class="content">
        {content_html}
      </main>
    </div>
  </div> <script>
    (function () {{
      var    btn    =    document.querySelector('.toc-toggle');   var   nav   =
      document.getElementById('toc-nav');    if    (!btn   ||   !nav)   return;
      btn.addEventListener('click', function () {{
        var            open           =           nav.classList.toggle('open');
        btn.setAttribute('aria-expanded', String(open)); btn.innerHTML = open ?
        'Contents &#9652;' : 'Contents &#9662;';
      }});
    }})();
  </script>
</body> </html>
"""


# ── Entry point ───────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a static HTML page from the teaching guide markdown"
        )
    )
    parser.add_argument("--input", default="docs/PLANCK_TEACHING_GUIDE.md")
    parser.add_argument("--output", default="docs/index.html")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / args.input
    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    markdown_text = input_path.read_text(encoding="utf-8")

    # 1. Extract math regions → tokens
    markdown_no_math, math_store = _extract_math(markdown_text)

    # 2. Render the rest of the markdown to HTML
    content_html, toc = render_markdown(markdown_no_math)

    # 3. Build TOC and assemble the full page (math tokens still present)
    toc_html = build_toc(toc)
    title = "Planck"
    page = page_template(title, toc_html, content_html)

    # 4.  Restore math tokens across the entire page in one pass so that tokens
    # appearing  in  heading text (and therefore in TOC sidebar links) are also
    # replaced — not just tokens in the main content body.
    page = _restore_math(page, math_store)

    output_path.write_text(page, encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
