"""Regression for Transformers 5 returning BatchEncoding, not token-ID lists."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import compare_gpu as c


class FakeTokenizer:
    def __init__(self):
        self.thinking = None
        self.special = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        self.thinking = enable_thinking
        if tokenize:
            return {'input_ids': [11, 22, 33], 'attention_mask': [1, 1, 1]}
        return 'rendered prompt'

    def encode(self, text, add_special_tokens):
        self.special = add_special_tokens
        return [11, 22, 33]


class TokenizerPreflightTests(unittest.TestCase):
    def test_explicit_render_then_encode_avoids_batchencoding(self):
        t = FakeTokenizer()
        self.assertEqual(c.prompt_ids(t, 'question', True), [11, 22, 33])
        self.assertTrue(t.thinking)
        self.assertFalse(t.special)

    def test_nonthinking_switch_is_preserved(self):
        t = FakeTokenizer()
        c.prompt_ids(t, 'question', False)
        self.assertFalse(t.thinking)

    def test_non_integer_token_output_is_rejected(self):
        t = FakeTokenizer()
        t.encode = lambda *args, **kwargs: {'input_ids': [11, 22]}
        with self.assertRaises(TypeError):
            c.prompt_ids(t, 'question', True)


if __name__ == '__main__': unittest.main()
