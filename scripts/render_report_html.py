"""Render a report as a portable HTML file with embedded local figures."""
import argparse
import base64
import html
from pathlib import Path
import re
from urllib.parse import urlsplit

from markdown_it import MarkdownIt


def render_report_html(source, output=None):
    source = Path(source).resolve()
    output = Path(output) if output else source.with_suffix('.html')
    root = Path(__file__).resolve().parents[1]
    text = source.read_text(encoding='utf-8')
    body = MarkdownIt('commonmark', {'html': False}).enable('table').render(text)

    def image(match):
        relative = match.group(1)
        if urlsplit(relative).scheme:
            return match.group(0)
        path = (source.parent / relative).resolve()
        path.relative_to(root)
        mime = 'image/png' if path.suffix == '.png' else 'image/svg+xml'
        data = base64.b64encode(path.read_bytes()).decode('ascii')
        return f'src="data:{mime};base64,{data}"'

    body = re.sub(r'src="([^"]+)"', image, body)

    def link(match):
        relative = match.group(1)
        if urlsplit(relative).scheme or relative.startswith('#'):
            return match.group(0)
        path = (source.parent / relative).resolve()
        repository_path = path.relative_to(root).as_posix()
        return 'href="https://github.com/sumitasthana/PATH/blob/main/' + html.escape(repository_path, quote=True) + '"'

    body = re.sub(r'href="([^"]+)"', link, body)
    headings = []

    def heading(match):
        label = match.group(1)
        identifier = re.sub(r'[^a-z0-9]+', '-', label.lower()).strip('-')
        headings.append((identifier, label))
        return f'<h2 id="{identifier}">{label}</h2>'

    body = re.sub(r'<h2>(.*?)</h2>', heading, body)
    body = body.replace('<table>', '<div class="table-scroll"><table>')
    body = body.replace('</table>', '</table></div>')
    navigation = ' '.join(f'<a href="#{identifier}">{label}</a>' for identifier, label in headings)
    page = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paired study draft report</title>
<style>
* { box-sizing: border-box; }
body { margin: 0; background: #f3f5f7; color: #202a35; font: 16px/1.65 "Segoe UI", Arial, sans-serif; }
main { max-width: 1180px; margin: 24px auto; padding: 32px 40px; background: white; border-radius: 8px; }
h1 { font-size: 30px; line-height: 1.25; color: #17365d; }
h2 { margin-top: 38px; padding-top: 12px; border-top: 1px solid #dde3ea; font-size: 23px; color: #17365d; }
a { color: #155a9a; text-underline-offset: 3px; }
nav { padding: 12px 0; display: flex; flex-wrap: wrap; gap: 8px 18px; font-size: 14px; }
.notice { padding: 12px 16px; background: #fff5d6; border-left: 4px solid #b78819; }
.table-scroll { overflow-x: auto; margin: 18px 0; }
table { border-collapse: collapse; width: 100%; font-size: 14px; line-height: 1.5; }
th, td { padding: 10px 12px; border: 1px solid #d9e1e9; min-width: 85px; vertical-align: top; }
th { background: #17365d; color: white; }
tbody tr:nth-child(even) { background: #f5f8fb; }
code { background: #edf1f5; padding: 2px 4px; border-radius: 3px; overflow-wrap: anywhere; font-size: 13px; }
img { max-width: 100%; height: auto; }
li { margin: 8px 0; }
@media (max-width: 650px) { main { margin: 0; padding: 20px 16px; } h1 { font-size: 26px; } }
@media print { body { background: white; } main { margin: 0; padding: 0; max-width: none; } nav { display: none; } .table-scroll { overflow: visible; } table { font-size: 10px; } th, td { padding: 5px; min-width: 0; } h2 { break-after: avoid; } tr, img { break-inside: avoid; } }
</style>
</head>
<body><main>
<div class="notice"><strong>Draft report.</strong> Remaining results are marked as pending. Tables scroll horizontally on smaller screens. You can print this page or save it as PDF from your browser.</div>
<nav aria-label="Report sections">NAVIGATION</nav>
BODY
</main></body></html>
'''.replace('NAVIGATION', navigation).replace('BODY', body)
    output.write_text(page, encoding='utf-8')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(render_report_html(args.source, args.output))
