#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "section"


def inline_format(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text


def render_markdown(markdown_text: str) -> tuple[str, list[tuple[int, str, str]]]:
    lines = markdown_text.splitlines()
    output: list[str] = []
    toc: list[tuple[int, str, str]] = []
    in_code = False
    code_lang = ""
    in_list = False
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            text = " ".join(part.strip() for part in paragraph if part.strip())
            if text:
                output.append(f"<p>{inline_format(text)}</p>")
            paragraph = []

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            output.append("</ul>")
            in_list = False

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        if line.startswith("```"):
            flush_paragraph()
            close_list()
            if not in_code:
                in_code = True
                code_lang = line[3:].strip()
                cls = f' class="language-{html.escape(code_lang)}"' if code_lang else ""
                output.append(f"<pre><code{cls}>")
            else:
                in_code = False
                output.append("</code></pre>")
            continue

        if in_code:
            output.append(html.escape(line))
            continue

        if not line.strip():
            flush_paragraph()
            close_list()
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            flush_paragraph()
            close_list()
            level = len(heading.group(1))
            text = heading.group(2).strip()
            anchor = slugify(text)
            toc.append((level, text, anchor))
            output.append(
                f'<h{level} id="{anchor}">{inline_format(text)}</h{level}>'
            )
            continue

        if line.startswith("- "):
            flush_paragraph()
            if not in_list:
                output.append("<ul>")
                in_list = True
            output.append(f"<li>{inline_format(line[2:].strip())}</li>")
            continue

        if re.match(r"^\d+\.\s+", line):
            flush_paragraph()
            close_list()
            output.append(f"<p>{inline_format(line)}</p>")
            continue

        if line.startswith("|") and line.endswith("|"):
            flush_paragraph()
            close_list()
            cells = [inline_format(cell.strip()) for cell in line.strip("|").split("|")]
            if all(re.fullmatch(r"[-: ]+", cell.replace("&amp;", "&")) for cell in cells):
                continue
            row = "".join(f"<td>{cell}</td>" for cell in cells)
            if not output or not output[-1].startswith("<table"):
                output.append("<table>")
                output.append("<tbody>")
            output.append(f"<tr>{row}</tr>")
            continue

        if output and output[-1] == "</tbody>" and not line.startswith("|"):
            output.append("</table>")

        paragraph.append(line)

    flush_paragraph()
    close_list()

    if output and output[-1] == "</tbody>":
        output.append("</table>")

    html_body = "\n".join(output).replace("<table>\n<tbody>\n", "<table>\n<tbody>\n")
    html_body = html_body.replace("</tbody>\n<table>", "</tbody>\n</table>\n<table>")
    html_body = html_body.replace("<table>\n<tbody>", "<table>\n<tbody>")
    if "<table>" in html_body and "</tbody>" not in html_body:
        html_body += "\n</tbody>\n</table>"
    if "<tbody>" in html_body and "</table>" not in html_body:
        html_body += "\n</table>"

    return html_body, toc


def build_toc(toc: list[tuple[int, str, str]]) -> str:
    items = []
    for level, text, anchor in toc:
        cls = f"toc-level-{level}"
        items.append(
            f'<li class="{cls}"><a href="#{anchor}">{html.escape(text)}</a></li>'
        )
    return "<ul>\n" + "\n".join(items) + "\n</ul>"


def page_template(title: str, toc_html: str, content_html: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --panel: #fffdf8;
      --ink: #1d2b36;
      --muted: #566472;
      --accent: #9a3d22;
      --rule: #dbcdb9;
      --code-bg: #f2ece2;
      --shadow: 0 18px 60px rgba(42, 33, 20, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(154, 61, 34, 0.08), transparent 22rem),
        linear-gradient(180deg, #fbf7f0 0%, var(--bg) 100%);
      line-height: 1.65;
    }}
    .page {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(154, 61, 34, 0.95), rgba(123, 53, 31, 0.92));
      color: #fff9f2;
      border-radius: 24px;
      padding: 28px 30px;
      box-shadow: var(--shadow);
      margin-bottom: 24px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: clamp(2rem, 3vw, 3.2rem);
      line-height: 1.1;
    }}
    .hero p {{
      margin: 0;
      max-width: 65ch;
      color: rgba(255, 249, 242, 0.92);
    }}
    .layout {{
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
      gap: 24px;
      align-items: start;
    }}
    .sidebar, .content {{
      background: var(--panel);
      border: 1px solid var(--rule);
      border-radius: 22px;
      box-shadow: var(--shadow);
    }}
    .sidebar {{
      position: sticky;
      top: 20px;
      padding: 22px 20px;
    }}
    .sidebar h2 {{
      margin: 0 0 12px;
      font-size: 1rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--accent);
    }}
    .sidebar ul {{
      list-style: none;
      padding: 0;
      margin: 0;
    }}
    .sidebar li {{
      margin: 0;
      padding: 0;
    }}
    .sidebar a {{
      display: block;
      padding: 6px 0;
      color: var(--muted);
      text-decoration: none;
    }}
    .sidebar a:hover {{
      color: var(--accent);
    }}
    .toc-level-2 a {{ padding-left: 12px; }}
    .toc-level-3 a {{ padding-left: 24px; font-size: 0.96rem; }}
    .content {{
      padding: 34px;
    }}
    .content h1, .content h2, .content h3, .content h4 {{
      line-height: 1.2;
      color: #13202b;
      scroll-margin-top: 24px;
    }}
    .content h1 {{
      font-size: 2.1rem;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--rule);
    }}
    .content h2 {{
      font-size: 1.55rem;
      margin-top: 2.3rem;
      padding-top: 0.2rem;
    }}
    .content h3 {{
      font-size: 1.18rem;
      margin-top: 1.8rem;
    }}
    .content p, .content li {{
      font-size: 1.02rem;
    }}
    .content code {{
      background: var(--code-bg);
      border: 1px solid #e0d7c9;
      border-radius: 6px;
      padding: 0.08rem 0.35rem;
      font-family: "SFMono-Regular", "Menlo", "Consolas", monospace;
      font-size: 0.94em;
    }}
    .content pre {{
      background: #201a17;
      color: #f8efe5;
      border-radius: 16px;
      padding: 18px;
      overflow-x: auto;
    }}
    .content pre code {{
      background: transparent;
      border: 0;
      padding: 0;
      color: inherit;
    }}
    .content table {{
      width: 100%;
      border-collapse: collapse;
      margin: 1.2rem 0;
      font-size: 0.96rem;
    }}
    .content td {{
      border: 1px solid var(--rule);
      padding: 10px 12px;
      vertical-align: top;
    }}
    .content a {{
      color: var(--accent);
    }}
    @media (max-width: 980px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .sidebar {{
        position: static;
      }}
      .content {{
        padding: 24px 18px;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>{html.escape(title)}</h1>
      <p>A teaching-first, theory-linked walkthrough of the Planck codebase and its current quantum chemistry methods.</p>
    </section>
    <div class="layout">
      <aside class="sidebar">
        <h2>Contents</h2>
        {toc_html}
      </aside>
      <main class="content">
        {content_html}
      </main>
    </div>
  </div>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a static HTML page from the teaching guide markdown")
    parser.add_argument("--input", default="docs/PLANCK_TEACHING_GUIDE.md")
    parser.add_argument("--output", default="docs/site/index.html")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / args.input
    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    markdown_text = input_path.read_text(encoding="utf-8")
    content_html, toc = render_markdown(markdown_text)
    toc_html = build_toc(toc)
    title = "Planck Teaching Guide"
    output_path.write_text(page_template(title, toc_html, content_html), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
