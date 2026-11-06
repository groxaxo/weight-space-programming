"""Check the recipe against the actual pinned training API without using a GPU."""
import ast
import importlib.util
import inspect
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("transformers"), "Run with requirements-coder.txt for the training API check")
class TrainingRuntimeTests(unittest.TestCase):
    def test_recipe_keywords_exist_in_installed_training_arguments(self):
        from transformers import TrainingArguments
        tree = ast.parse((Path(__file__).resolve().parents[1] / "scripts/train_coder.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "TrainingArguments"]
        self.assertEqual(len(calls), 1)
        supported = inspect.signature(TrainingArguments).parameters
        unknown = {keyword.arg for keyword in calls[0].keywords} - set(supported)
        self.assertEqual(unknown, set(), f"Unsupported TrainingArguments in this recipe: {unknown}")

    def test_five_percent_warmup_uses_expected_optimizer_steps(self):
        from transformers import TrainingArguments
        with tempfile.TemporaryDirectory() as directory:
            args = TrainingArguments(output_dir=directory, warmup_steps=0.05, use_cpu=True, report_to="none")
            self.assertEqual(args.get_warmup_steps(72), 4)
            self.assertEqual(args.get_warmup_steps(2), 1)


if __name__ == "__main__":
    unittest.main()
