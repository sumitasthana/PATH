"""Build a static research report and export figures from committed evidence."""
import argparse
import hashlib
import html
import io
import json
import re
from pathlib import Path
from string import Template

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/reports/trajectory/evidence'
SITE = ROOT / 'site'
BLUE, ORANGE, TEAL, INK = '#245d99', '#b85b22', '#267768', '#253349'


def read(name):
    return json.loads((EVIDENCE / name).read_text(encoding='utf-8'))


def table(headers, rows):
    head = ''.join('<th scope="col">'+html.escape(str(x))+'</th>' for x in headers)
    body = []
    for row in rows:
        body.append('<tr><th scope="row">'+html.escape(str(row[0]))+'</th>'+''.join(
            '<td>'+html.escape(str(x))+'</td>' for x in row[1:])+'</tr>')
    return '<div class="table-scroll"><table><thead><tr>'+head+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></div>'


def style_axis(ax, xlabel=None, ylabel=None):
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#b4bcc5')
    ax.tick_params(colors=INK, labelsize=11)
    if xlabel: ax.set_xlabel(xlabel, color=INK, labelpad=10)
    if ylabel: ax.set_ylabel(ylabel, color=INK, labelpad=10)
    ax.grid(axis='y', color='#e6e9ed', linewidth=.8)
    ax.set_axisbelow(True)


def export(fig, name, title, assets):
    plt.rcParams['svg.hashsalt'] = name
    stream = io.StringIO()
    fig.savefig(stream, format='svg', metadata={'Creator': None, 'Date': None, 'Title': title}, facecolor='white')
    plt.close(fig)
    svg = stream.getvalue()
    svg = svg[svg.index('<svg '):]
    for identifier in re.findall(r'\bid="([^"]+)"',svg):
        unique = name+'-'+identifier
        svg = svg.replace('id="'+identifier+'"','id="'+unique+'"')
        svg = svg.replace('href="#'+identifier+'"','href="#'+unique+'"')
        svg = svg.replace('url(#'+identifier+')','url(#'+unique+')')
    svg = svg.replace('<svg ', '<svg role="img" aria-label="'+html.escape(title, quote=True)+'" ', 1)
    svg = '\n'.join(line.rstrip() for line in svg.splitlines()) + '\n'
    assets['assets/'+name+'.svg'] = svg.encode()
    return svg


def build():
    names = ['20261007_beta_batch_comparison.json','20261007_beta01_repeat_comparison.json',
             '20261007_sequence_comparison.json','20261007_paired_comparison.json',
             '20261007_pair_learning_only_report.json','20261007_pair_unlearn_then_learn_report.json']
    batch, repeat, sequence, pair, a, b = [read(name) for name in names]
    assert pair['status'] == sequence['status'] == a['status'] == b['status'] == 'complete'
    assert all(pair['pair_checks'].values())
    assert pair['selection']['source_sha256'] == sequence['lessons'][-1]['checkpoint_sha256']
    assert a['final_accuracies'] == pair['learning_only_final_accuracies']
    assert b['final_accuracies'] == pair['unlearn_then_learn_final_accuracies']
    for field in ('initial_model_sha256','starting_accuracies','learning_rng_sha256',
                  'learning_cuda_rng_sha256','new_task_code_sha256'):
        assert a[field] == b[field]
    assert [r['indices'] for r in a['learning_trace']] == [r['indices'] for r in b['learning_trace']]
    assert [r['input_sha256'] for r in a['learning_trace']] == [r['input_sha256'] for r in b['learning_trace']]
    assert len(a['learning_trace']) == len(b['learning_trace']) == 395
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none',
        'axes.labelsize':12,'axes.titleweight':'bold','axes.titlesize':13})
    assets, values = {}, {}
    config, environment = a['config'], a['environment']
    assert config['hidden'] == [128, 256, 512]
    assert config['code_dim'] == 32 and config['chunks'] == 200
    diagram_values = {'code_dim': config['code_dim'], 'pair_dim': 2 * config['code_dim'],
        'hidden': ' → '.join(map(str, config['hidden'])), 'chunks': config['chunks'],
        'generated_count': f"{environment['generated_parameters']:,}",
        'weight_chunks': environment['chunks_per_head']['weights'],
        'bn_chunks': environment['chunks_per_head']['batchnorm'],
        'residual_chunks': environment['chunks_per_head']['residual']}
    diagram = Template((ROOT/'scripts/templates/architecture.svg').read_text(encoding='utf-8')).substitute(diagram_values)
    assets['assets/07-architecture.svg'] = diagram.encode()
    assets['assets/08-architecture-unlearning.svg'] = diagram.replace('data-mode="learning"', 'data-mode="unlearning"', 1).encode()
    values['architecture_diagram'] = diagram
    values['generated_parameters'] = f"{environment['generated_parameters']:,}"
    values['hypernetwork_parameters'] = f"{environment['hypernetwork_parameters']:,}"

    fig, ax = plt.subplots(figsize=(9,4), layout='constrained')
    tasks = ['3','0','9','5','17']
    x = np.arange(len(tasks))
    first = [repeat['first_final_accuracies'][t] for t in tasks]
    second = [repeat['repeat_final_accuracies'][t] for t in tasks]
    for offset, scores, color, label in ((-.19,first,BLUE,'First run'),(.19,second,ORANGE,'Fresh-session repeat')):
        bars=ax.bar(x+offset,scores,.36,color=color,label=label)
        ax.bar_label(bars,fmt='%.1f',padding=4,fontsize=11)
    ax.set_xticks(x,['Task '+t for t in tasks])
    ax.set_ylim(0,70)
    ax.legend(loc='upper left',frameon=False,ncols=2,fontsize=10)
    style_axis(ax,ylabel='Validation accuracy (%)')
    values['repeat_chart']=export(fig,'01-repeat','Final validation scores on all five tasks in two task-17 training runs',assets)

    trials=batch['trials']
    fig, axes=plt.subplots(1,2,figsize=(9.5,4.5),layout='constrained')
    y=np.arange(5)
    labels=[f"{t['beta']:g}"+('*' if t['reused'] else '') for t in trials]
    colors=[BLUE if not t['reused'] else '#8794a3' for t in trials]
    for ax, column, title, limits in ((axes[0],'new_task_accuracy','Learning task 17',(0,65)),
                                    (axes[1],'mean_old_change','Change in the original four tasks',(-30,5))):
        numbers=[t[column] for t in trials]
        bars=ax.barh(y,numbers,color=colors,height=.58)
        ax.set_yticks(y,labels)
        ax.invert_yaxis()
        ax.set_xlim(*limits)
        ax.set_title(title,fontsize=12,pad=15)
        ax.axvline(0,color='#8693a0',linewidth=.9)
        ax.set_ylabel('Beta (protection strength)')
        ax.set_xlabel('Validation accuracy (%)' if column=='new_task_accuracy' else 'Mean accuracy change (points)')
        ax.spines[['top','right','left']].set_visible(False)
        ax.grid(axis='x',color='#e6e9ed')
        ax.set_axisbelow(True)
        for bar,value in zip(bars,numbers):
            ax.text(value+(.8 if value>=0 else -.7),bar.get_y()+bar.get_height()/2,
                f'{value:.1f}' if column=='new_task_accuracy' else f'{value:+.2f}',
                ha='left' if value>=0 else 'right',va='center',fontsize=11)
    values['beta_chart']=export(fig,'02-beta','Task 17 learning and old-task retention across five beta values',assets)

    stages=sequence['scores_by_stage']
    initial=sequence['selection']['initial_tasks']
    means=[sum(row['accuracies'][t] for t in initial)/len(initial) for row in stages]
    target=[row['accuracies']['17'] for row in stages]
    fig,ax=plt.subplots(figsize=(8.5,4),layout='constrained')
    for numbers,color,label,marker in ((means,BLUE,'Mean of the same five original tasks','o'),(target,ORANGE,'Task 17','s')):
        ax.plot(range(4),numbers,color=color,marker=marker,linewidth=2.3,label=label)
        for i,v in enumerate(numbers): ax.annotate(f'{v:.2f}' if numbers is means else f'{v:.1f}',(i,v),xytext=(0,9),textcoords='offset points',ha='center',fontsize=10)
    ax.set_xticks(range(4),['Start','After learning 1','After learning 7','After learning 14'])
    ax.set_ylim(0,65)
    ax.legend(loc='lower left',frameon=False,fontsize=10)
    style_axis(ax,ylabel='Validation accuracy (%)')
    values['sequence_chart']=export(fig,'03-sequence','Original five-task mean and task-17 accuracy after learning tasks 1, 7, and 14',assets)

    fig,ax=plt.subplots(figsize=(8.5,4),layout='constrained')
    for report,color,label,marker in ((a,BLUE,'A: learning only','o'),(b,ORANGE,'B: unlearn 3, then learn 15','s')):
        numbers=[report['starting_accuracies']['3'],report['before_learning_accuracies']['3']]+[r['accuracies']['3'] for r in report['epochs']]
        ax.plot(range(7),numbers,color=color,label=label,marker=marker,linewidth=2.3)
        ax.annotate(f'{numbers[-1]:.1f}%',(6,numbers[-1]),xytext=(8,4),textcoords='offset points',color=color,fontsize=12)
    ax.axhline(10,color=INK,linestyle='--',linewidth=1,label='Chance: 10%')
    ax.set_xticks(range(7),['Start','Before\ntask 15','Epoch 1','Epoch 2','Epoch 3','Epoch 4','Epoch 5'])
    ax.set_xlim(-.15,6.7)
    ax.set_ylim(0,50)
    ax.legend(loc='upper left',frameon=False,fontsize=10)
    style_axis(ax,ylabel='Task 3 validation accuracy (%)')
    values['forget_chart']=export(fig,'04-forgetting','Task 3 stays at chance during one later lesson',assets)

    retained=pair['retained_tasks']
    deltas=[pair['retained_changes']['paired_final_difference'][t] for t in retained]
    fig,ax=plt.subplots(figsize=(8.5,4.2),layout='constrained')
    bars=ax.barh(range(7),deltas,color=[BLUE if v>=0 else ORANGE for v in deltas],height=.6)
    ax.set_yticks(range(7),['Task '+t for t in retained])
    ax.invert_yaxis()
    ax.axvline(0,color=INK,linewidth=1)
    ax.set_xlim(-8,12)
    ax.set_xlabel('Final score difference: B minus A (percentage points)')
    ax.spines[['top','right','left']].set_visible(False)
    ax.grid(axis='x',color='#e6e9ed')
    ax.set_axisbelow(True)
    for bar,v in zip(bars,deltas): ax.text(v+(.25 if v>=0 else -.25),bar.get_y()+bar.get_height()/2,f'{v:+.1f}',ha='left' if v>=0 else 'right',va='center',fontsize=11)
    values['retained_chart']=export(fig,'05-retained','Retained tasks respond differently to the paired unlearning branch',assets)

    scored=[r for r in b['forget_trace'] if 'accuracies' in r]
    fig,ax=plt.subplots(figsize=(8.5,4),layout='constrained')
    steps=[r['step'] for r in scored]
    numbers=[sum(r['accuracies'][t] for t in retained)/len(retained) for r in scored]
    ax.plot(steps,numbers,color=BLUE,marker='o',label='Mean of the seven retained tasks',linewidth=2)
    ax.plot(steps,[r['accuracies']['3'] for r in scored],color=ORANGE,marker='s',label='Task 3 (forget target)',linewidth=2)
    ax.axhline(10,color=INK,linestyle='--',linewidth=.9,label='Chance: 10%')
    ax.set_xlim(-2,104)
    ax.set_ylim(0,60)
    ax.legend(loc='upper right',frameon=False,fontsize=10)
    style_axis(ax,xlabel='Unlearning update',ylabel='Validation accuracy (%)')
    values['unlearning_chart']=export(fig,'06-unlearning','Retained scores initially fall and then mostly recover during unlearning',assets)

    values['repeat_table']=table(['Task','First run (%)','Repeat (%)','Repeat minus first (points)'],[
        ['Task '+t,f'{f:.1f}',f'{s:.1f}',f'{s-f:+.1f}'] for t,f,s in zip(tasks,first,second)])
    values['beta_table']=table(['Beta','Task 17 (%)','Mean old-task change (points)','Largest old-task drop (points)'],[
        [f"{t['beta']:g}"+('*' if t['reused'] else ''),f"{t['new_task_accuracy']:.1f}",f"{t['mean_old_change']:+.2f}",f"{t['largest_old_drop']:.1f}"] for t in trials])
    all_tasks=[*initial,*sequence['selection']['new_tasks']]
    values['sequence_table']=table(['Stage']+['Task '+t+' (%)' for t in all_tasks],[
        [row['stage'].capitalize()]+[f"{row['accuracies'][t]:.1f}" if t in row['accuracies'] else 'Not learned' for t in all_tasks] for row in stages])
    values['pair_table']=table(['Task','Start (%)','B: after unlearning 3 (%)','A: learn 15 (%)','B: unlearn 3, then learn 15 (%)','B minus A (points)'],[
        ['Task '+t+(' (target)' if t=='3' else ' (new)' if t=='15' else ''),
         f"{pair['starting_accuracies'][t]:.1f}" if t!='15' else 'Not learned',
         f"{pair['after_unlearning_accuracies'][t]:.1f}" if t!='15' else 'Not learned',
         f"{pair['learning_only_final_accuracies'][t]:.1f}",f"{pair['unlearn_then_learn_final_accuracies'][t]:.1f}",
         f"{pair['unlearn_then_learn_final_accuracies'][t]-pair['learning_only_final_accuracies'][t]:+.1f}"] for t in [*pair['starting_accuracies'],'15']])
    spill=sum(abs(pair['after_unlearning_accuracies'][t]-pair['starting_accuracies'][t]) for t in retained)
    values['spill']=f'{spill:.1f}'
    values['source_list']='<ul>'+''.join('<li><a href="https://github.com/sumitasthana/PATH/blob/main/docs/reports/trajectory/evidence/'+name+'">'+html.escape(label)+'</a></li>' for name,label in zip(names,[
        'Five-value beta batch','Beta 0.1 repeat comparison','Learning sequence: tasks 1, 7, and 14',
        'Paired U3 and L15 comparison','Branch A: learning-only report','Branch B: unlearn-then-learn report']))+'</ul>'
    import statistics
    study = json.loads((ROOT/'docs/reports/evidence/paired-study-2026-10-10.json').read_text(encoding='utf-8'))
    completed = study['selected_comparisons']
    assert len(completed) == 6 and all(all(row['pair_checks'].values()) for row in completed)
    get = lambda seed, target: next(row for row in completed if row['seed']==seed and row['forget_task']==str(target))
    accuracy = table(['Seed', 'Learn 15 only', 'Forget 3 ? Learn 15', 'Forget 14 ? Learn 15'], [[seed, f"{get(seed,3)['new_task_a']:.1f}%", f"{get(seed,3)['new_task_b']:.1f}%", f"{get(seed,14)['new_task_b']:.1f}%"] for seed in range(3)])
    effects = table(['Sequence', 'Task-15 changes: seeds 0, 1, 2', 'Mean task-15 change', 'Mean other-task change'], [[f'Forget {target} ? Learn 15', ', '.join(f"{get(seed,target)['new_task_difference']:+.1f}" for seed in range(3)), f"{statistics.mean(get(seed,target)['new_task_difference'] for seed in range(3)):+.2f}", f"{statistics.mean(get(seed,target)['retained_mean_difference'] for seed in range(3)):+.2f}"] for target in [3,14]])
    values['latest_study'] = f'''<section id="paired-expanded">
      <p class="section-number">Latest results ? 10 October 2026</p>
      <h2>Completed extension: three seeds and two forgetting targets</h2>
      <p>Each seed first learns <strong>3 ? 0 ? 9 ? 5 ? 17 ? 1 ? 7 ? 14</strong>. We copy its source model into three independent continuations: learn 15 only, forget 3 then learn 15, and forget 14 then learn 15. This tests different request sequences after one source order.</p>
      {accuracy}
      {effects}
      <p class="small">Changes are percentage points: forgetting branch minus the matching learning-only baseline. Other-task means exclude the requested forgotten task and average seven tasks per seed. Each sequence mean uses three seeds.</p>
      <div class="note"><p><strong>Main finding:</strong> both forgotten tasks reach 10% accuracy after forgetting and remain there after task-15 learning in every seed. Forgetting task 3 matches or improves task-15 accuracy, but the +5.40-point mean includes a large +13.0-point seed-2 effect. Forgetting task 14 has mixed effects.</p></div>
      <h3>Do earlier tasks retain their accuracy?</h3>
      <p>We evaluate all previously learned tasks except the requested forgotten task. During forgetting, drops are measured against the model before forgetting. After learning task 15, retention is measured against the same seed's learning-only baseline. For forget 3, the retained tasks are 0, 9, 5, 17, 1, 7, and 14. For forget 14, they are 3, 0, 9, 5, 17, 1, and 7.</p>
      <p>Five of six forgetting branches have lower mean retained-task accuracy. Individual final losses range up to <strong>14.2 points</strong>. Accuracy is therefore not uniformly preserved, even when the requested task stays at chance.</p>
      <p class="small">12 jobs, six comparisons, and 39 requests complete. All recorded pairing checks pass. One source order, one incoming task, and three seeds limit generalization. Task identity and learning position change together. No erasure, privacy, or recovery claim is established. The new source models use uniform settings; the earlier pilot below is not included in these averages.</p>
      <p><a href="paired-study.html">Concise report with charts, settings, and seed-level losses</a> ? <a href="paired-study-presentation.html">Four-slide presentation and interactive experiment tree</a> ? <a href="https://github.com/sumitasthana/PATH/blob/main/docs/reports/evidence/paired-study-2026-10-10.json">Verified evidence</a></p>
    </section>'''
    template=Template((ROOT/'scripts/templates/experiment_report.html').read_text(encoding='utf-8'))
    assets['index.html']=template.substitute(values).encode()
    assets['.nojekyll']=b''
    chart_data={'repeat':{'tasks':tasks,'first':first,'repeat':second}, 'beta_trials':trials,
        'sequence':{'stages':stages,'fixed_tasks':initial,'means':means},
        'pair':pair,'unlearning_scored_updates':scored}
    assets['data/chart-data.json']=(json.dumps(chart_data,indent=2)+'\n').encode()
    manifest={'date':'2026-10-07','sources':[{'file':name,'canonical_json_sha256':hashlib.sha256(
        json.dumps(read(name),sort_keys=True,separators=(',',':')).encode()).hexdigest()} for name in names],
        'definition':'Canonical JSON SHA256 ignores formatting differences; figures use these saved measurements.'}
    assets['data/source-manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    return assets


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Verify generated HTML and figures match evidence.')
    args=parser.parse_args()
    assets=build()
    for name,content in assets.items():
        target=SITE/name
        if args.check:
            if not target.is_file() or target.read_bytes().replace(b'\r\n',b'\n') != content.replace(b'\r\n',b'\n'):
                raise SystemExit('Generated report differs: '+name)
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(content)
    print(('Checked' if args.check else 'Built')+f' HTML report, six charts, learning and unlearning diagrams, and source data ({len(assets)} files).')


if __name__=='__main__': main()
