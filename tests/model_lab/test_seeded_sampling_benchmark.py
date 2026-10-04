"""Sampling semantics regressions for the isolated batching prototype."""
import unittest


class SeededSamplingTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            from transformers import GenerationConfig
            from model_lab.seeded_sampling_benchmark import IndependentRowSampler
        except ImportError:
            self.skipTest("Native sampling benchmark uses the isolated Torch environment")
        self.torch = torch
        self.config = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=5)
        self.Sampler = IndependentRowSampler

    def test_row_seeds_are_independent_of_batch_order_and_other_rows(self):
        torch = self.torch
        scores = torch.tensor([[0., 1., 2., 3., 4., 5., 6., 7.], [7., 6., 5., 4., 3., 2., 1., 0.]])
        batch = self.Sampler([101, 202], self.config, "cpu")
        reversed_batch = self.Sampler([202, 101], self.config, "cpu")
        alone = self.Sampler([101], self.config, "cpu")
        for step in range(12):
            input_ids = torch.zeros(2, step + 1, dtype=torch.long)
            chosen = batch(input_ids, scores.clone()).argmax(-1)
            reverse = reversed_batch(input_ids, scores.flip(0).clone()).argmax(-1)
            single = alone(input_ids[:1], scores[:1].clone()).argmax(-1)
            self.assertTrue(torch.equal(chosen, reverse.flip(0)))
            self.assertTrue(torch.equal(single, chosen[:1]))

    def test_draws_match_native_top_k_distribution_and_generator_stream(self):
        torch = self.torch
        from transformers.generation.logits_process import TopKLogitsWarper, LogitNormalization
        scores = torch.linspace(-2., 2., 16).reshape(1, -1)
        sampler = self.Sampler([73101], self.config, "cpu")
        native_generator = torch.Generator(device="cpu").manual_seed(73101)
        ids = torch.zeros(1, 2, dtype=torch.long)
        for step in range(12):
            native_scores = TopKLogitsWarper(5)(ids, scores.clone())
            native_scores = LogitNormalization()(ids, native_scores)
            expected = torch.multinomial(torch.softmax(native_scores, -1), 1, generator=native_generator)
            forced = sampler(ids, scores.clone())
            self.assertTrue(torch.equal(forced.argmax(-1, keepdim=True), expected))
            # HF's duplicate warper/selection sees one-token support.
            repeated = TopKLogitsWarper(5)(ids, forced)
            self.assertTrue(torch.equal(torch.multinomial(torch.softmax(repeated, -1), 1), expected))

    def test_left_padding_preserves_original_ids_and_attention_mask(self):
        from model_lab.seeded_sampling_benchmark import left_padded_inputs
        ids, mask = left_padded_inputs([[11, 12], [21, 22, 23, 24]], 0, "cpu")
        self.assertEqual(ids.tolist(), [[0, 0, 11, 12], [21, 22, 23, 24]])
        self.assertEqual(mask.tolist(), [[0, 0, 1, 1], [1, 1, 1, 1]])

    def test_unsupported_post_draw_transforms_fail_closed(self):
        self.config.min_p = 0.1
        with self.assertRaisesRegex(ValueError, "active min_p"):
            self.Sampler([1], self.config, "cpu")


if __name__ == "__main__":
    unittest.main()
