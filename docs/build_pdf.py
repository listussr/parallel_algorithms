"""Сборка docs/*.md в один PDF: Markdown -> HTML (формулы KaTeX) -> PDF (headless Chrome).

Использование (из корня репозитория):  python docs/build_pdf.py
Нужны: Python-пакет markdown, Google Chrome или Microsoft Edge, доступ к cdn.jsdelivr.net
и fonts.googleapis.com (KaTeX и шрифты подгружаются при печати).
Результат: docs/lu_gaxpy.pdf
"""
import base64, html, os, re, subprocess, sys, tempfile
import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
OUT = os.path.join(DOCS, "lu_gaxpy.pdf")
REPO = "https://github.com/listussr/parallel_algorithms/blob/main/"
FILES = ["01_sequential.md", "02_vectorization.md", "03_autopar.md", "04_mpi.md"]
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]
KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/"


def slug(text):
    """Якорь заголовка по правилам GitHub: строчные, без пунктуации, пробелы -> '-'."""
    text = re.sub(r"<[^>]+>", "", text).strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def protect_math(md, store):
    """Заменяет $$...$$ и $...$ вне кода на заглушки, чтобы Markdown их не испортил."""
    def put(tex, display):
        store.append((tex.strip(), display))
        return f"MATHPH{len(store) - 1:05d}X"

    def text_part(s):
        s = re.sub(r"\$\$(.+?)\$\$", lambda m: "\n\n" + put(m.group(1), True) + "\n\n", s, flags=re.S)
        return re.sub(r"(?<![\\$])\$(?!\s)([^$\n]+?)(?<!\s)\$", lambda m: put(m.group(1), False), s)

    out = []
    for i, block in enumerate(re.split(r"(^```.*?^```)", md, flags=re.S | re.M)):
        if i % 2:                                   # блок кода - без изменений
            out.append(block)
            continue
        for j, span in enumerate(re.split(r"(`[^`\n]+`)", block)):
            out.append(span if j % 2 else text_part(span))
    return "".join(out)


def restore_math(h, store):
    def sub(m):
        tex, display = store[int(m.group(1))]
        tag = "div" if display else "span"
        return f'<{tag} class="tex" data-display="{int(display)}">{html.escape(tex)}</{tag}>'
    h = re.sub(r"<p>\s*MATHPH(\d{5})X\s*</p>", sub, h)
    return re.sub(r"MATHPH(\d{5})X", sub, h)


def fix_links(h, doc_id):
    """Ссылки между файлами docs -> якоря внутри PDF, на исходники -> GitHub."""
    ids = {f: f"d{n}" for n, f in enumerate(FILES, 1)}

    def href(m):
        url = m.group(1)
        if url.startswith(("http://", "https://")):
            return m.group(0)
        path, _, frag = url.partition("#")
        if not path:
            return f'href="#{doc_id}-{frag}"'
        name = os.path.basename(path)
        if name in ids:
            return f'href="#{ids[name]}' + (f"-{frag}" if frag else "") + '"'
        target = os.path.normpath(os.path.join("docs", path)).replace("\\", "/")
        return f'href="{REPO}{target}"'
    return re.sub(r'href="([^"]+)"', href, h)


def embed_images(h):
    def img(m):
        path = os.path.normpath(os.path.join(DOCS, m.group(1)))
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f'src="data:image/png;base64,{data}"'
    return re.sub(r'src="([^"]+\.png)"', img, h)


def convert(name, n):
    with open(os.path.join(DOCS, name), encoding="utf-8") as f:
        md_text = f.read()
    store = []
    md_text = protect_math(md_text, store)
    doc_id = f"d{n}"
    conv = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists", "toc"],
                             extension_configs={"toc": {"slugify": lambda t, sep: f"{doc_id}-" + slug(t)}})
    h = conv.convert(md_text)
    h = restore_math(h, store)
    h = fix_links(h, doc_id)
    h = embed_images(h)
    # заголовок документа (h1) получает id всего раздела
    h = re.sub(r'<h1 id="[^"]*">', f'<h1 id="{doc_id}">', h, count=1)
    toc = [(lvl, ident, re.sub(r"<[^>]+>", "", txt))
           for lvl, ident, txt in re.findall(r'<h([12]) id="([^"]+)">(.*?)</h\1>', h)]
    toc = [(lvl, ident, restore_toc_math(txt, store)) for lvl, ident, txt in toc]
    return h, toc


def restore_toc_math(txt, store):
    return re.sub(r'MATHPH(\d{5})X', lambda m: html.escape(store[int(m.group(1))][0]), txt)


CSS = r"""
@page { size: A4; margin: 20mm 18mm 22mm 18mm;
        @bottom-center { content: counter(page); font: 9pt 'PT Sans', sans-serif; color: #888; } }
@page :first { @bottom-center { content: none; } }
:root { --accent: #1f4e79; --accent2: #2e75b6; --line: #d0d7de; --soft: #f4f7fb; }
html { font-family: 'PT Serif', Georgia, serif; font-size: 10.5pt; line-height: 1.5; color: #1b1b1b; }
body { margin: 0; }
h1, h2, h3, h4 { font-family: 'PT Sans', 'Segoe UI', sans-serif; color: var(--accent);
                 break-after: avoid; line-height: 1.25; }
h1 { font-size: 21pt; margin: 0 0 6mm; padding-bottom: 3mm; border-bottom: 2.5px solid var(--accent); }
h2 { font-size: 14.5pt; margin: 8mm 0 3mm; padding-bottom: 1.5mm; border-bottom: 1px solid var(--line); }
h3 { font-size: 12pt; margin: 6mm 0 2mm; color: var(--accent2); }
p { margin: 2mm 0; text-align: justify; hyphens: auto; }
a { color: var(--accent2); text-decoration: none; }
hr { display: none; }
ul, ol { margin: 2mm 0; padding-left: 7mm; }
li { margin: 1mm 0; }
code { font-family: 'Cascadia Mono', 'PT Mono', Consolas, monospace; font-size: 9pt; background: var(--soft);
       padding: 0.3mm 1mm; border-radius: 2px; }
pre { background: var(--soft); border: 1px solid var(--line); border-left: 3px solid var(--accent2);
      border-radius: 3px; padding: 3mm 4mm; font-size: 8.6pt; line-height: 1.35;
      break-inside: avoid; white-space: pre-wrap; }
pre code { background: none; padding: 0; font-family: 'Cascadia Mono', Consolas, monospace; font-variant-ligatures: none; }
table { border-collapse: collapse; margin: 3mm auto; font-size: 9.2pt; break-inside: avoid; }
th, td { border: 1px solid var(--line); padding: 1.2mm 2.5mm; }
th { background: var(--accent); color: #fff; font-family: 'PT Sans', sans-serif; font-weight: 700; }
tr:nth-child(even) td { background: #f7f9fc; }
blockquote { margin: 4mm 0; padding: 2mm 5mm; background: #fff8e6; border-left: 3px solid #e0a800;
             break-inside: avoid; }
img { display: block; max-width: 82%; margin: 4mm auto; break-inside: avoid; }
div.tex { margin: 3mm 0; overflow: hidden; break-inside: avoid; }
.katex { font-size: 1.08em; }
.katex-display { margin: 0; }
.doc { break-before: page; }
.cover { height: 250mm; display: flex; flex-direction: column; justify-content: center;
         text-align: center; font-family: 'PT Sans', sans-serif; }
.cover .band { background: var(--accent); color: #fff; padding: 14mm 10mm; margin: 0 -18mm; }
.cover .title { font-size: 28pt; font-weight: 700; margin-bottom: 4mm; }
.cover .sub { font-size: 14pt; opacity: .9; }
.cover .meta { margin-top: 18mm; font-size: 12pt; color: #444; line-height: 1.8; }
.cover .formula { margin-top: 14mm; font-size: 16pt; color: var(--accent); }
.toc { break-before: page; }
.toc ol { list-style: none; padding: 0; }
.toc li.l1 { margin-top: 5mm; font-family: 'PT Sans', sans-serif; font-size: 12.5pt; font-weight: 700; }
.toc li.l2 { margin: 1mm 0 1mm 8mm; font-size: 10.5pt; }
.toc a { color: #1b1b1b; }
.toc li.l1 a { color: var(--accent); }
"""


def build_html():
    parts, toc_items = [], []
    for n, name in enumerate(FILES, 1):
        h, toc = convert(name, n)
        parts.append(f'<section class="doc">{h}</section>')
        toc_items += toc
    toc_html = "".join(
        f'<li class="l{lvl}"><a href="#{ident}">{txt}</a></li>' for lvl, ident, txt in toc_items)
    toc_html = re.sub(r"\$([^$]+)\$", r"\1", toc_html)
    cover = f"""
<div class="cover">
  <div class="band">
    <div class="title">Блочное LU-разложение через GAXPY</div>
    <div class="sub">Курс «Параллельные алгоритмы» · вариант 5</div>
  </div>
  <div class="formula"><span class="tex" data-display="1">A = L\\,U</span></div>
  <div class="meta">
    Последовательный алгоритм · Векторизация · Автоматическое распараллеливание · MPI<br>
    Intel Core i5-1135G7 · MinGW GCC 9.2 · MS-MPI
  </div>
</div>
<nav class="toc"><h1>Содержание</h1><ol>{toc_html}</ol></nav>"""
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>Блочное LU-разложение через GAXPY</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=PT+Mono&family=PT+Sans:wght@400;700&family=PT+Serif:ital,wght@0,400;0,700;1,400&display=block&subset=cyrillic">
<link rel="stylesheet" href="{KATEX}katex.min.css">
<script src="{KATEX}katex.min.js"></script>
<style>{CSS}</style></head><body>
{cover}
{''.join(parts)}
<script>
document.querySelectorAll('.tex').forEach(function (el) {{
  try {{
    katex.render(el.textContent, el, {{displayMode: el.dataset.display === '1',
                                       throwOnError: false, strict: false}});
  }} catch (e) {{ el.style.color = 'red'; }}
}});
</script>
</body></html>"""


def fix_outline(page_html):
    """Chrome задваивает в закладках заголовки, стоящие в начале страницы.
    Закладки пересобираются с теми же страницами, но с текстом заголовков из HTML."""
    from pypdf import PdfReader, PdfWriter
    titles = [html.unescape(re.sub(r"<[^>]+>", "", t)).strip()
              for t in re.findall(r"<h[1-6][^>]*>(.*?)</h[1-6]>", page_html, flags=re.S)]
    reader = PdfReader(OUT)
    items = []                                      # (уровень, страница) в порядке документа

    def walk(nodes, level):
        for node in nodes:
            if isinstance(node, list):
                walk(node, level + 1)
            else:
                items.append((level, reader.get_destination_page_number(node)))
    walk(reader.outline, 0)
    if len(items) != len(titles):
        print("закладки оставлены как есть: число заголовков не совпало")
        return
    writer = PdfWriter(clone_from=reader)
    writer._root_object.pop("/Outlines", None)
    parents = {}
    for (level, page), title in zip(items, titles):
        parents[level] = writer.add_outline_item(title, page, parent=parents.get(level - 1))
    writer.page_mode = "/UseOutlines"
    writer.add_metadata({"/Title": "Блочное LU-разложение через GAXPY"})
    with open(OUT, "wb") as f:
        writer.write(f)


def main():
    browser = next((b for b in BROWSERS if os.path.exists(b)), None)
    if not browser:
        sys.exit("Не найден Chrome или Edge")
    html_path = os.path.join(tempfile.gettempdir(), "lu_gaxpy_docs.html")
    page_html = build_html()
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(page_html)
    profile = tempfile.mkdtemp(prefix="pdfprofile")
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-first-run",
                    f"--user-data-dir={profile}", "--no-pdf-header-footer",
                    "--generate-pdf-document-outline", "--virtual-time-budget=20000",
                    f"--print-to-pdf={OUT}", "file:///" + html_path.replace("\\", "/")],
                   check=True, timeout=180, capture_output=True)
    fix_outline(page_html)
    print("PDF:", OUT)


if __name__ == "__main__":
    main()
