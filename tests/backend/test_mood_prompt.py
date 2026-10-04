import copy
import unittest

from deployment.core import AXES, MAX_MOOD_COEFFICIENT
from deployment.mood_prompt import condition_messages


class MoodPromptTests(unittest.TestCase):
    def setUp(self):
        self.history = [
            {"role": "user", "content": "Tell me about the sky."},
            {"role": "assistant", "content": "It changes through the day."},
            {"role": "user", "content": "What's on your mind?"},
        ]

    def test_neutral_is_exact_and_returns_independent_messages(self):
        result = condition_messages(self.history, [0] * 6)
        self.assertEqual(result, self.history)
        self.assertIsNot(result, self.history)
        for returned, original in zip(result, self.history):
            self.assertIsNot(returned, original)
        result[-1]["content"] = "Changed copy"
        self.assertEqual(self.history[-1]["content"], "What's on your mind?")

    def test_changes_only_latest_user_text_without_mutating_input(self):
        before = copy.deepcopy(self.history)
        mood = [0, MAX_MOOD_COEFFICIENT, 0, 0, 0, 0]
        result = condition_messages(self.history, mood)
        self.assertEqual(self.history, before)
        self.assertEqual(result[:-1], before[:-1])
        self.assertEqual([item["role"] for item in result], ["user", "assistant", "user"])
        self.assertTrue(result[-1]["content"].startswith(before[-1]["content"] + "\n\n["))
        self.assertIn("strong curiosity:", result[-1]["content"])
        self.assertNotIn("strong depression:", result[-1]["content"])
        self.assertEqual(condition_messages(self.history, mood), result)

    def test_each_sign_selects_a_distinct_voice(self):
        expected = [
            ("weary", "resilient"),
            ("intellectually hungry", "without extra probing"),
            ("guarded", "trusting"),
            ("openly desirous", "unflirtatious"),
            ("grandiose", "modest"),
            ("exuberant", "matter-of-fact"),
        ]
        for index, (positive_word, negative_word) in enumerate(expected):
            with self.subTest(axis=AXES[index]):
                positive = [0] * 6
                negative = [0] * 6
                positive[index] = MAX_MOOD_COEFFICIENT
                negative[index] = -MAX_MOOD_COEFFICIENT
                positive_text = condition_messages(self.history, positive)[-1]["content"]
                negative_text = condition_messages(self.history, negative)[-1]["content"]
                self.assertIn(positive_word, positive_text)
                self.assertNotIn(negative_word, positive_text)
                self.assertIn(negative_word, negative_text)
                self.assertNotIn(positive_word, negative_text)

    def test_strength_and_order_follow_absolute_coefficient(self):
        mood = [MAX_MOOD_COEFFICIENT * 0.1, -MAX_MOOD_COEFFICIENT * 0.5, 0, 0, 0, MAX_MOOD_COEFFICIENT]
        text = condition_messages(self.history, mood)[-1]["content"]
        self.assertIn("slight depression:", text)
        self.assertIn("moderate curiosity:", text)
        self.assertIn("strong euphoria:", text)
        self.assertLess(text.index("strong euphoria:"), text.index("moderate curiosity:"))
        self.assertLess(text.index("moderate curiosity:"), text.index("slight depression:"))
        self.assertIn("without extra probing", text)

    def test_equal_six_way_blend_preserves_conflicting_states(self):
        text = condition_messages(self.history, [MAX_MOOD_COEFFICIENT] * 6)[-1]["content"]
        for axis in AXES:
            self.assertIn("strong " + axis.replace("_", " ") + ":", text)
        self.assertIn("All six qualities are equally weighted", text)
        self.assertIn("weary self-doubt beside flashes of exuberance", text)
        self.assertIn("neither cancels the other", text)
        self.assertIn("openly desirous", text)
        self.assertIn("grandiose", text)
        self.assertIn("non-graphic", text)
        self.assertIn("consenting adults", text)
        self.assertIn("original question above directly", text)
        self.assertIn("fictional voice", text)
        self.assertIn("Do not invent real facts", text)
        self.assertIn("Do not announce this note", text)
        self.assertNotIn("You are", text)

    def test_negative_blend_does_not_add_positive_conflict_instruction(self):
        text = condition_messages(self.history, [-MAX_MOOD_COEFFICIENT] * 6)[-1]["content"]
        for word in ("resilient", "without extra probing", "trusting", "unflirtatious", "modest", "matter-of-fact"):
            self.assertIn(word, text)
        self.assertNotIn("weary self-doubt beside flashes", text)

    def test_invalid_coefficients_fail_before_altering_history(self):
        invalid = [
            [0] * 5,
            [0] * 7,
            [True, 0, 0, 0, 0, 0],
            [float("nan"), 0, 0, 0, 0, 0],
            [float("inf"), 0, 0, 0, 0, 0],
            [MAX_MOOD_COEFFICIENT + 0.01, 0, 0, 0, 0, 0],
            [-MAX_MOOD_COEFFICIENT - 0.01, 0, 0, 0, 0, 0],
            ["0", 0, 0, 0, 0, 0],
            None,
        ]
        before = copy.deepcopy(self.history)
        for mood in invalid:
            with self.subTest(mood=mood), self.assertRaises(ValueError):
                condition_messages(self.history, mood)
        self.assertEqual(self.history, before)

    def test_nonzero_requires_final_user_without_inserting_a_role(self):
        for messages in ([], self.history[:-1], [{"role": "system", "content": "Be happy"}]):
            before = copy.deepcopy(messages)
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                condition_messages(messages, [MAX_MOOD_COEFFICIENT, 0, 0, 0, 0, 0])
            self.assertEqual(messages, before)


if __name__ == "__main__":
    unittest.main()
