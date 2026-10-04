import unittest

from deployment.formatting import content_token_controls, post_instruction_start


END = 10
SUFFIX = [END, 2, 11, 3, 2, 12, 4, 13, 4]


class FakeTensor:
    def __init__(self, rows, shape=None):
        self.rows = rows
        self.shape = shape if shape is not None else (len(rows), len(rows[0]) if rows else 0)

    def tolist(self):
        return self.rows


class FakeTokenizer:
    def __len__(self):
        return 30

    def convert_tokens_to_ids(self, token):
        return END if token == "<|im_end|>" else None

    def convert_ids_to_tokens(self, token):
        return "<|im_end|>" if token == END else "other"

    def encode(self, text, **kwargs):
        if kwargs != {"add_special_tokens": False}:
            raise AssertionError("Marker tokenization must not add special tokens")
        return [END]


def inputs(ids, mask=None):
    result = {"input_ids": FakeTensor([ids])}
    if mask is not None:
        result["attention_mask"] = FakeTensor([mask])
    return result


class PostInstructionTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_boundary_is_only_final_formatted_prompt_token(self):
        prompt = [11, 5, 2, 6, 7]
        ids = prompt + SUFFIX
        start = post_instruction_start(self.tokenizer, inputs(ids, [1] * len(ids)))
        self.assertEqual(start, len(ids) - 1)
        self.assertEqual(ids[start:], SUFFIX[-1:])

    def test_history_is_excluded_from_current_suffix(self):
        history = [11, 5, 2, 6, END, 2, 11, 3, 2, 7, END, 2]
        final_user = [11, 5, 2, 8, 9]
        ids = history + final_user + SUFFIX
        self.assertEqual(post_instruction_start(self.tokenizer, inputs(ids)), len(ids) - 1)

    def test_user_literal_markers_do_not_move_suffix_into_user_text(self):
        final_user = [11, 5, 2, 6, END, 7, END, 8]
        ids = final_user + SUFFIX
        self.assertEqual(post_instruction_start(self.tokenizer, inputs(ids)), len(ids) - 1)

    def test_missing_marker_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "final user end marker"):
            post_instruction_start(self.tokenizer, inputs([11, 5, 2, 6]))

    def test_empty_assistant_suffix_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "assistant generation suffix"):
            post_instruction_start(self.tokenizer, inputs([11, 5, 2, 6, END]))

    def test_requires_nonempty_batch_one_tensor(self):
        malformed = [
            FakeTensor([], shape=(0, 0)),
            FakeTensor([[]]),
            FakeTensor([[END, 2], [END, 2]]),
            FakeTensor([END, 2], shape=(2,)),
            FakeTensor([[END, 2]], shape=(1, 3)),
        ]
        for value in malformed:
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                post_instruction_start(self.tokenizer, {"input_ids": value})

    def test_invalid_token_ids_are_rejected(self):
        for token in (-1, 30, True, 1.0, "10"):
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, "invalid token IDs"):
                post_instruction_start(self.tokenizer, inputs([token, END, 2]))

    def test_padding_and_mismatched_masks_are_rejected(self):
        for mask in ([0, 1, 1], [1, 1], [1, 1, 1, 1]):
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "unpadded"):
                post_instruction_start(self.tokenizer, inputs([6, END, 2], mask))

    def test_missing_input_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing input_ids"):
            post_instruction_start(self.tokenizer, {})

    def test_marker_must_be_a_known_single_token(self):
        unsupported = [
            {"convert_tokens_to_ids": lambda token: None},
            {"convert_tokens_to_ids": lambda token: 30},
            {"convert_tokens_to_ids": lambda token: True},
            {"convert_ids_to_tokens": lambda token: "<unknown>"},
            {"encode": lambda text, **kwargs: [END, 2]},
        ]
        for changes in unsupported:
            tokenizer = FakeTokenizer()
            for name, replacement in changes.items():
                setattr(tokenizer, name, replacement)
            with self.subTest(changes=list(changes)), self.assertRaisesRegex(ValueError, "one known token"):
                post_instruction_start(tokenizer, inputs([6, END, 2]))


class ContentTokenControlTests(unittest.TestCase):
    def test_chat_eos_and_non_special_thinking_tags_are_excluded(self):
        class Tokenizer:
            all_special_ids = [10]

            def get_vocab(self):
                return {"<|control|>": 11, "<think>": 12, "</think>": 13, "ordinary": 14}

            def encode(self, text, **kwargs):
                return [{"<think>": 12, "</think>": 13}[text]]

            def decode(self, ids, **kwargs):
                return {12: "<think>", 13: "</think>"}[ids[0]]

        excluded, thinking = content_token_controls(Tokenizer(), [15])
        self.assertEqual(set(excluded), {10, 11, 12, 13, 15})
        self.assertEqual(thinking, (12, 13))


if __name__ == "__main__":
    unittest.main()
