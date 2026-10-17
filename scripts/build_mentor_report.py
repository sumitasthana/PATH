"""Build the concise public report from verified paired-study evidence."""
import html
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / 'docs/reports/evidence/paired-study-2026-10-10.json').read_text(encoding='utf-8'))
rows = data['selected_comparisons']
assert len(rows) == 6 and all(all(r['pair_checks'].values()) for r in rows)
get = lambda seed, target: next(r for r in rows if r['seed'] == seed and r['forget_task'] == str(target))
def table(headers, records):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{html.escape(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in record) + '</tr>' for record in records) + '</tbody></table></div>'
def label(x, y, value, size=16, anchor='middle'):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}">{html.escape(str(value))}</text>'
chart = ''
for tick in range(0, 101, 20):
    y = 270 - tick * 2
    chart += f'<line x1="65" x2="910" y1="{y}" y2="{y}" stroke="#d9e1ea"/>' + label(50, y + 5, tick, 13, 'end')
for seed in range(3):
    a, b = get(seed, 3), get(seed, 14)
    for i, value in enumerate([a['new_task_a'], a['new_task_b'], b['new_task_b']]):
        x = 130 + seed * 280 + i * 60
        chart += f'<rect x="{x}" y="{270-value*2}" width="46" height="{value*2}" fill="{["#64748b", "#235fa6", "#b75b20"][i]}"/>' + label(x + 23, 260 - value * 2, f'{value:.1f}%', 16)
    chart += label(210 + seed * 280, 302, f'Seed {seed}', 18)
chart = '<svg viewBox="0 0 960 325" role="img" aria-label="Task 15 accuracy: seed 0, 50.8, 50.8, 48.2 percent; seed 1, 44.0, 47.2, 47.8 percent; seed 2, 34.4, 47.4, 32.6 percent."><title>Task-15 accuracy (%) across three seeds</title>' + label(20, 30, 'Accuracy (%)', 14, 'start') + chart + '</svg>'
results = table(['Seed', 'Learn 15 only', 'Forget 3 → Learn 15', 'Forget 14 → Learn 15'], [[seed, f'{get(seed,3)["new_task_a"]:.1f}%', f'{get(seed,3)["new_task_b"]:.1f}%', f'{get(seed,14)["new_task_b"]:.1f}%'] for seed in range(3)])
averages = []
for target in [3, 14]:
    selected = [get(seed, target) for seed in range(3)]
    values = [r['new_task_difference'] for r in selected]
    averages.append([f'Forget {target} → Learn 15', ', '.join(f'{v:+.1f}' for v in values), f'{statistics.mean(values):+.2f}', f'{statistics.stdev(values):.2f}', f'{statistics.mean(r["retained_mean_difference"] for r in selected):+.2f}'])
summary = table(['Request sequence', 'Task-15 changes: seeds 0, 1, 2', 'Mean task-15 change', 'Task-15 SD', 'Mean other-task change'], averages)
forgotten = table(['Task', 'Before forgetting: seeds 0, 1, 2', 'After forgetting', 'After learning 15'], [[target, ', '.join(f'{get(seed,target)["starting_target_accuracy"]:.1f}%' for seed in range(3)), '10% in every seed', '10% in every seed'] for target in [3, 14]])
damage = table(['Seed', 'Forget 3: largest other-task loss', 'Forget 14: largest other-task loss'], [[seed] + [f'Task {min(get(seed,target)["retained_differences"], key=get(seed,target)["retained_differences"].get)}: {get(seed,target)["largest_paired_retained_drop"]:.1f} points' for target in [3, 14]] for seed in range(3)])
settings = table(['Data and model', 'Learning', 'Forgetting'], [['Tiny ImageNet; 10 classes/task; 5,000 train + 500 validation images/task; generated ResNet-50', '5 epochs; batch 64; Adam, learning rate 0.0001; protection weight 0.1', '100 steps; 10 noise samples/step; Adam, learning rate 0.0001; noise weight 0.01']])
page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Completed paired continual-unlearning study: three seeds, three request sequences, accuracy results and limits."><title>Accuracy after forgetting: three-seed study | CARK</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f9;color:#203448;font:16px/1.55 system-ui,Arial}}main{{max-width:1050px;margin:24px auto;background:white;padding:32px 42px;border-radius:10px}}h1{{font-size:34px;line-height:1.15}}h2{{font-size:23px;margin:30px 0 10px}}p{{margin:12px 0}}a{{color:#235fa6}}.muted,figcaption{{color:#536579;font-size:14px}}.finding{{padding:15px 18px;background:#edf3fa;border-left:4px solid #235fa6}}.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:14px;margin:12px 0}}th,td{{text-align:left;padding:11px;border:1px solid #d7e0e9;vertical-align:top}}th{{background:#edf2f7}}svg{{width:100%;height:auto}}svg text{{fill:#203448;font-family:system-ui,Arial}}figure{{margin:15px 0}}.legend{{display:flex;flex-wrap:wrap;gap:20px;font-size:14px}}.swatch{{display:inline-block;width:12px;height:12px;margin-right:5px}}.sequence{{padding:12px;background:#f5f7fa;font-weight:600}}@media(max-width:650px){{main{{margin:0;padding:22px 16px}}h1{{font-size:28px}}th,td{{min-width:95px}}}}@media print{{body{{background:white}}main{{margin:0;padding:0}}table{{font-size:11px}}figure,tr{{break-inside:avoid}}h2{{break-after:avoid}}}}
</style></head><body><main><nav><a href="index.html">← CARK reports</a> · <a href="paired-study-presentation.html">Presentation and experiment tree</a></nav>
<h1>Accuracy after forgetting: three seeds, three request sequences</h1><p class="muted">Completed study · 10 October 2026 · One source learning order · Seeds 0, 1, 2</p>
<p class="finding"><strong>Main finding:</strong> the model learns task 15 after forgetting. Its accuracy depends on the seed and request sequence. The forgotten tasks stay at chance accuracy, but some other tasks lose accuracy.</p>
<h2>Design</h2><p>For each seed, we copy the same source model into three branches: <strong>learn 15 only</strong> (baseline), <strong>forget 3 then learn 15</strong>, and <strong>forget 14 then learn 15</strong>. A seed sets random choices, including initialization.</p><div class="sequence">Source: Learn 3 → 0 → 9 → 5 → 17 → 1 → 7 → 14</div><p class="muted">Task 3 was learned first; task 14 was learned last. Compare branches within the same seed.</p>
{settings}
<h2>1. Learning task 15</h2><p>Accuracy is the percentage of validation images classified correctly. Higher is better.</p><div class="legend"><span><i class="swatch" style="background:#64748b"></i>Learn 15 only</span><span><i class="swatch" style="background:#235fa6"></i>Forget 3 → Learn 15</span><span><i class="swatch" style="background:#b75b20"></i>Forget 14 → Learn 15</span></div><figure>{chart}<figcaption>Each forgetting branch is compared with the baseline from the same seed.</figcaption></figure>{results}
{summary}<p class="muted">All changes are percentage points: branch minus baseline. Positive means higher accuracy. Means use three seeds. SD (standard deviation) measures variation between seeds; it is not a confidence interval. Other-task means exclude the forgotten task.</p><p><strong>Forget 3:</strong> the task-15 score matches or exceeds the baseline in every seed. The +5.40-point average includes a large +13.0-point seed-2 effect. <strong>Forget 14:</strong> gains and losses average to −0.20 points.</p>
<h2>2. Forgetting persists; other tasks can suffer</h2>{forgotten}<p class="muted">Each task has 10 classes, so chance accuracy is 10%. All six comparisons meet the 12% forgetting threshold; no saved evaluation during task-15 learning exceeds it.</p>{damage}<p>Average changes on other tasks are −1.29 points after forgetting task 3 and −1.02 after forgetting task 14. Individual losses reach <strong>14.2 points</strong>. A small average does not mean every task is preserved.</p>
<h2>Conclusion and limits</h2><p>The results support forgetting that persists during one later task, with continued ability to learn. They do not show uniform accuracy preservation. Only three seeds, one source order, and one incoming task were tested. Task identity and learning position change together. Chance accuracy does not prove information erasure, privacy, or resistance to recovery.</p>
<p class="muted">12/12 jobs · 6/6 comparisons · 39/39 requests complete. Eleven verified session logs total 15 h 10 m 20 s of runner time, including evaluation and transfers; this is not billed GPU time. This diagnostic study uses protection weight 0.1 rather than the repository's paper-based default of 0.01.</p>
<p class="muted"><a href="https://github.com/sumitasthana/PATH/blob/main/docs/reports/evidence/paired-study-2026-10-10.json">Verified scores and settings</a> · <a href="https://github.com/sumitasthana/PATH/blob/main/docs/reports/Paired-study-2026-10-10-draft.md">Detailed report</a>. All recorded pairing checks pass. Historical results are excluded from these averages.</p>
</main></body></html>'''
assert '\u2014' not in page
(ROOT / 'site/paired-study.html').write_text(page, encoding='utf-8')
# Pages is configured for main:/, so keep the shared root URL available too.
root_page = page.replace('href="index.html"', 'href="site/index.html"').replace('href="paired-study-presentation.html"', 'href="site/paired-study-presentation.html"')
(ROOT / 'paired-study.html').write_text(root_page, encoding='utf-8')
(ROOT / 'site/paired-study-presentation.html').write_text((ROOT / 'docs/reports/Paired-study-2026-10-10-presentation.html').read_text(encoding='utf-8'), encoding='utf-8')
print('Built site/paired-study.html and its linked presentation.')
