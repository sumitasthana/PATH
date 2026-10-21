"""Publish the committed experiment page and figures without touching working edits."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'https://github.com/sumitasthana/PATH.git'
WIKI = 'https://github.com/sumitasthana/PATH.wiki.git'
PAGE = Path('docs/wiki/Experiment-trajectory.md')
# Other hand-written pages to publish beside the record, each with the heading
# it must start with. The heading check is what stops an unrelated file being
# pushed over a wiki page.
EXTRA_PAGES = {Path('docs/wiki/Mentor-QnA.md'): '# Mentor QnA\n'}
LOGS = ROOT / 'outputs/wiki_publication'


def run(*args, cwd=None):
    environment = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='never')
    result = subprocess.run(args, cwd=cwd, env=environment, capture_output=True,
                            text=True, encoding='utf-8', timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or 'Command failed.')
    return result.stdout.strip()


def copy_record(source, wiki):
    """Publish the experiment index, dated records, extra pages, and SVG evidence.

    Every page is read and checked before any of them is written, so a page
    that fails its check leaves the live wiki untouched.
    """
    source, wiki = Path(source), Path(wiki)
    text = (source / PAGE).read_text(encoding='utf-8')
    if not text.startswith('# Experiment trajectory\n') or '\u2014' in text:
        raise ValueError('The experiment page failed its heading or prose check.')
    records = {PAGE.name: text}
    for path in sorted((source / PAGE.parent).glob('Experiment-trajectory-*.md')):
        dated_text = path.read_text(encoding='utf-8')
        if not dated_text.startswith('# Experiment trajectory: ') or '\u2014' in dated_text:
            raise ValueError(f'The dated experiment page failed its prose check: {path.name}')
        records[path.name] = dated_text
    for page, heading in EXTRA_PAGES.items():
        extra_text = (source / page).read_text(encoding='utf-8')
        if not extra_text.startswith(heading) or '\u2014' in extra_text:
            raise ValueError(f'The page failed its heading or prose check: {page.name}')
        records[page.name] = extra_text
    for name, content in records.items():
        (wiki / name).write_text(content, encoding='utf-8')
    names = list(records)
    figures = source / 'docs/reports/trajectory/figures'
    for figure in sorted(figures.rglob('*.svg')):
        relative = Path('figures') / figure.relative_to(figures)
        destination = wiki / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(figure, destination)
        names.append(relative.as_posix())
    return names


def remove_checkout(scratch, temporary_root):
    scratch, temporary_root = Path(scratch).resolve(), Path(temporary_root).resolve()
    scratch.relative_to(temporary_root)
    if scratch.parent != temporary_root or not scratch.name.startswith('cark-trajectory-publish-'):
        raise ValueError('Refusing to remove an unexpected temporary path.')
    def readonly_file(function, path, error):
        Path(path).resolve().relative_to(scratch)
        if not isinstance(error, PermissionError):
            raise error
        os.chmod(path, stat.S_IWRITE)
        function(path)
    shutil.rmtree(scratch, onexc=readonly_file)


def publish(repo=ROOT, *, push=False):
    repo = Path(repo).resolve()
    name, email = (run('git', 'config', key, cwd=repo) for key in ('user.name', 'user.email'))
    if not name or not email:
        raise ValueError('Configure the user Git identity before publishing.')
    temporary_root = Path(tempfile.gettempdir()).resolve()
    scratch = Path(tempfile.mkdtemp(prefix='cark-trajectory-publish-', dir=temporary_root)).resolve()
    scratch.relative_to(temporary_root)
    try:
        source, wiki = scratch / 'source', scratch / 'wiki'
        run('git', 'clone', '--depth', '1', '--branch', 'main', REPOSITORY, str(source))
        run(sys.executable, 'scripts/build_experiment_wiki.py', '--check', cwd=source)
        run('git', 'clone', '--depth', '1', WIKI, str(wiki))
        names = copy_record(source, wiki)
        run('git', 'add', '--', *names, cwd=wiki)
        changed = run('git', 'diff', '--cached', '--name-only', cwd=wiki).splitlines()
        source_commit = run('git', 'rev-parse', 'HEAD', cwd=source)
        if changed and push:
            run('git', 'diff', '--cached', '--check', cwd=wiki)
            run('git', 'config', 'user.name', name, cwd=wiki)
            run('git', 'config', 'user.email', email, cwd=wiki)
            run('git', 'commit', '-m', 'Update experiment trajectory', cwd=wiki)
            run('git', 'push', 'origin', 'HEAD', cwd=wiki)
        return {'source_commit': source_commit, 'changed_files': changed,
                'status': 'published' if changed and push else 'unchanged' if not changed else 'preview'}
    finally:
        # Check the absolute target before deleting this run's temporary checkouts.
        remove_checkout(scratch, temporary_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true', help='Commit and push changes; otherwise preview only.')
    args = parser.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=LOGS / 'publish.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    try:
        result = publish(push=args.publish)
        result['checked_at'] = datetime.now().astimezone().isoformat()
        logging.info('%s', result)
        (LOGS / 'latest.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        if sys.stdout is not None:
            print(json.dumps(result, indent=2))
    except Exception as error:
        logging.exception('Publication failed')
        (LOGS / 'latest.json').write_text(json.dumps({'status': 'failed', 'error': str(error),
            'checked_at': datetime.now().astimezone().isoformat()}, indent=2) + '\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
