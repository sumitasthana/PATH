"""Validate archived measurements and their generated wiki representation."""

import copy
import importlib.util
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_wiki", ROOT / "scripts" / "build_experiment_wiki.py")
wiki = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wiki)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.registry = json.loads((wiki.DATA / "registry.json").read_text(encoding="utf-8"))
        self.traces = wiki.read_csv("forgetting_traces.csv")
        self.gradients = wiki.read_csv("gradient_norms.csv")
        self.norms = wiki.read_csv("raw_output_norms.csv")

    def validate(self):
        wiki.validate(self.registry, self.traces, self.gradients, self.norms)

    def test_actual_archive_agrees_with_derived_screens(self):
        self.validate()
        manifest = json.loads((wiki.DATA / "manifest.json").read_text(encoding="utf-8"))
        by_id = {r["id"]: r for r in self.registry["experiments"]}
        for run in manifest["runs"]:
            observations = by_id[run["run"]]["observations"]
            for key in ("passing_steps", "minimum_task3_accuracy_pct", "first_retention_failure_step"):
                self.assertEqual(observations[key], run[key], (run["run"], key))

    def test_changed_drift_cannot_silently_change_interpretation(self):
        self.traces[-1]["task0_absolute_drift_pp"] = "20"
        with self.assertRaisesRegex(ValueError, "Incorrect drift"):
            self.validate()

    def test_invented_pass_is_rejected(self):
        self.traces[-1]["passes_screen"] = "True"
        with self.assertRaisesRegex(ValueError, "Incorrect screen"):
            self.validate()

    def test_missing_step_is_rejected_for_completed_runs(self):
        del self.traces[-2]
        with self.assertRaisesRegex(ValueError, "Missing or unordered"):
            self.validate()

    def test_duplicate_gradient_rows_are_rejected(self):
        self.gradients.append(copy.deepcopy(self.gradients[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate observations"):
            self.validate()

    def test_generated_pages_match_sources_and_internal_links_resolve(self):
        pages = wiki.render(self.registry, self.traces, self.gradients, self.norms)
        expected = {'Home.md', '_Sidebar.md', 'Experiment-trajectory.md',
            'Concepts-and-processes.md', 'Model-diagnostics.md', 'Weekly-Updates.md',
            'An-Unlearning-Framework-for-Continual-Learning.md'}
        expected.update(path.name for path in (ROOT / 'docs/wiki').glob('Experiment-trajectory-*.md'))
        self.assertEqual(set(pages), expected)
        for name, text in pages.items():
            self.assertEqual((ROOT / "docs" / "wiki" / name).read_text(encoding="utf-8"), text, name)
            self.assertNotIn("\u2014", text, name)
            for slug in re.findall(r"https://github.com/sumitasthana/PATH/wiki/([^#)\s]+)", text):
                self.assertIn(slug + ".md", pages, (name, slug))
            for slug in re.findall(r"\]\((Experiment-trajectory[^#)\s]*)(?:#[^)]+)?\)", text):
                self.assertIn(slug + '.md', pages, (name, slug))


if __name__ == "__main__":
    unittest.main()
