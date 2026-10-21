import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('publication', ROOT / 'scripts/publish_experiment_trajectory.py')
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


class PublicationTests(unittest.TestCase):
    def test_cleanup_handles_readonly_git_files_and_rejects_other_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            scratch = temporary_root / 'cark-trajectory-publish-test'
            scratch.mkdir()
            packed = scratch / 'pack.idx'
            packed.write_bytes(b'git pack')
            os.chmod(packed, 0o444)
            publication.remove_checkout(scratch, temporary_root)
            self.assertFalse(scratch.exists())
            with self.assertRaises(ValueError):
                publication.remove_checkout(temporary_root, temporary_root)

    def write_extra_pages(self, source):
        """The hand-written pages published beside the record, with valid content."""
        for page, heading in publication.EXTRA_PAGES.items():
            (source / page).parent.mkdir(parents=True, exist_ok=True)
            (source / page).write_text(heading + '\nAn answer.\n')

    def test_publication_copies_only_record_and_svg_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            source, wiki = Path(directory) / 'source', Path(directory) / 'wiki'
            (source / publication.PAGE).parent.mkdir(parents=True)
            (source / publication.PAGE).write_text('# Experiment trajectory\n\nResult.\n')
            dated = 'Experiment-trajectory-2026-10-07.md'
            (source / publication.PAGE.parent / dated).write_text('# Experiment trajectory: 7 October 2026\n\nResults.\n')
            self.write_extra_pages(source)
            figures = source / 'docs/reports/trajectory/figures/2026-10-04'
            figures.mkdir(parents=True)
            (figures / 'chart.svg').write_text('<svg/>')
            (figures / 'private.json').write_text('{}')
            wiki.mkdir()
            (wiki / 'Home.md').write_text('Keep this page')
            copied = publication.copy_record(source, wiki)
            extras = [page.name for page in publication.EXTRA_PAGES]
            self.assertEqual(copied, ['Experiment-trajectory.md', dated, *extras,
                                      'figures/2026-10-04/chart.svg'])
            self.assertIn('Results.', (wiki / dated).read_text())
            self.assertEqual((wiki / 'Home.md').read_text(), 'Keep this page')
            self.assertFalse((wiki / 'figures/2026-10-04/private.json').exists())
            self.assertEqual((wiki / 'Experiment-trajectory.md').read_text(), '# Experiment trajectory\n\nResult.\n')
            for name in extras:
                self.assertIn('An answer.', (wiki / name).read_text())

    def test_bad_extra_page_does_not_replace_any_wiki_page(self):
        # Every page is checked before any is written. A bad extra page must
        # therefore leave the index alone, not half-publish the update.
        with tempfile.TemporaryDirectory() as directory:
            source, wiki = Path(directory) / 'source', Path(directory) / 'wiki'
            (source / publication.PAGE).parent.mkdir(parents=True)
            (source / publication.PAGE).write_text('# Experiment trajectory\n\nNew index.\n')
            self.write_extra_pages(source)
            spoiled = next(iter(publication.EXTRA_PAGES))
            (source / spoiled).write_text('An unrelated document')
            wiki.mkdir()
            (wiki / publication.PAGE.name).write_text('Existing index')
            with self.assertRaises(ValueError):
                publication.copy_record(source, wiki)
            self.assertEqual((wiki / publication.PAGE.name).read_text(), 'Existing index')
            self.assertFalse((wiki / spoiled.name).exists())

    def test_bad_record_is_rejected_before_replacing_wiki_page(self):
        with tempfile.TemporaryDirectory() as directory:
            source, wiki = Path(directory) / 'source', Path(directory) / 'wiki'
            (source / publication.PAGE).parent.mkdir(parents=True)
            (source / publication.PAGE).write_text('An unrelated document')
            wiki.mkdir()
            (wiki / 'Experiment-trajectory.md').write_text('Existing record')
            with self.assertRaises(ValueError):
                publication.copy_record(source, wiki)
            self.assertEqual((wiki / 'Experiment-trajectory.md').read_text(), 'Existing record')

    def test_bad_dated_record_does_not_partially_replace_index(self):
        with tempfile.TemporaryDirectory() as directory:
            source, wiki = Path(directory) / 'source', Path(directory) / 'wiki'
            (source / publication.PAGE).parent.mkdir(parents=True)
            (source / publication.PAGE).write_text('# Experiment trajectory\n\nNew index.\n')
            (source / publication.PAGE.parent / 'Experiment-trajectory-2026-10-07.md').write_text('Invalid dated page')
            wiki.mkdir()
            (wiki / publication.PAGE.name).write_text('Existing index')
            with self.assertRaises(ValueError):
                publication.copy_record(source, wiki)
            self.assertEqual((wiki / publication.PAGE.name).read_text(), 'Existing index')


if __name__ == '__main__':
    unittest.main()
