"""Tensor and pinned-Qwen checks; the lightweight backend suite needs no torch."""

from importlib.util import find_spec
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

from deployment.core import AXES, PLACEHOLDER_NORM
from deployment.steering import MoodSteering


def tensor_model(layer_count=2, hidden_size=4):
    class IdentityLayer(torch.nn.Module):
        def forward(self, hidden_states, position_ids=None):
            return hidden_states

    return SimpleNamespace(
        device=torch.device("cpu"),
        config=SimpleNamespace(text_config=SimpleNamespace(
            num_hidden_layers=layer_count, hidden_size=hidden_size,
        )),
        model=SimpleNamespace(language_model=SimpleNamespace(
            layers=torch.nn.ModuleList([IdentityLayer() for _ in range(layer_count)]),
        )),
    )


def provided_bank():
    # Every axis is nonzero; unequal magnitudes reveal unwanted renormalization.
    direction = torch.tensor([1.0, 2.0, -3.0, 0.25])
    return torch.stack([
        torch.stack([direction * (axis + 1) * (layer + 1) for axis in range(len(AXES))])
        for layer in range(2)
    ])


@unittest.skipIf(torch is None, "torch is optional for the lightweight backend suite")
class TensorSteeringTests(unittest.TestCase):
    def setUp(self):
        self.model = tensor_model()
        self.bank = provided_bank()
        self.steering = MoodSteering(self.model, self.bank)

    def assert_no_hooks(self):
        for layer in self.model.model.language_model.layers:
            self.assertEqual(len(layer._forward_pre_hooks), 0)

    def test_signed_sum_masks_prefill_and_all_cached_tokens(self):
        coefficients = [2, -1, 1, 0, -2, 1]
        expected_offsets = torch.tensor([
            [-1.0, -2.0, 3.0, -0.25], [-2.0, -4.0, 6.0, -0.5],
        ])
        torch.testing.assert_close(self.steering.offsets(coefficients), expected_offsets)
        hidden = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4).to(torch.bfloat16)
        untouched = hidden.clone()
        positions = torch.arange(6).reshape(1, -1)
        with self.steering.apply(coefficients, 4) as applied:
            self.assertTrue(applied)
            for index, layer in enumerate(self.model.model.language_model.layers):
                result = layer(hidden, position_ids=positions)
                expected = hidden.clone()
                expected[:, 4:] = (hidden[:, 4:].float() + expected_offsets[index]).to(hidden.dtype)
                self.assertTrue(torch.equal(result, expected))
                self.assertTrue(torch.equal(result[:, :4], hidden[:, :4]))
                self.assertEqual(result.dtype, torch.bfloat16)
                # Repeating a forward must use positions, not an accumulated token counter.
                self.assertTrue(torch.equal(layer(hidden, position_ids=positions), result))
                cached = hidden[:, :1]
                cached_result = layer(hidden_states=cached, position_ids=torch.tensor([[6]]))
                cached_expected = (cached.float() + expected_offsets[index]).to(cached.dtype)
                self.assertTrue(torch.equal(cached_result, cached_expected))
        self.assertTrue(torch.equal(hidden, untouched))
        self.assert_no_hooks()

    def test_neutral_is_exact_and_registers_no_hooks(self):
        hidden = torch.randn(1, 6, 4, dtype=torch.bfloat16)
        layer = self.model.model.language_model.layers[0]
        with self.steering.apply([0] * len(AXES), 4) as applied:
            self.assertFalse(applied)
            self.assert_no_hooks()
            self.assertIs(layer(hidden), hidden)
        self.assert_no_hooks()

    def test_cleanup_on_generation_error_and_next_request_is_independent(self):
        layer = self.model.model.language_model.layers[0]
        hidden = torch.zeros(1, 3, 4)
        positions = torch.tensor([[0, 1, 2]])
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            with self.steering.apply([1, 0, 0, 0, 0, 0], 1):
                first = layer(hidden, position_ids=positions)
                torch.testing.assert_close(first[:, 1:], self.bank[0, 0].expand(1, 2, -1))
                raise RuntimeError("generation failed")
        self.assert_no_hooks()
        self.assertIs(layer(hidden), hidden)
        with self.steering.apply([-2, 0, 0, 0, 0, 0], 2):
            second = layer(hidden, position_ids=positions)
            self.assertTrue(torch.equal(second[:, :2], hidden[:, :2]))
            torch.testing.assert_close(second[:, 2:], (-2 * self.bank[0, 0]).reshape(1, 1, -1))
        self.assert_no_hooks()

    def test_placeholder_is_seeded_has_fixed_norm_and_leaves_global_rng_alone(self):
        rng = torch.get_rng_state().clone()
        first = MoodSteering(self.model)
        second = MoodSteering(self.model)
        self.assertTrue(torch.equal(first.vectors, second.vectors))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        torch.testing.assert_close(
            first.vectors.norm(dim=-1), torch.full((2, len(AXES)), PLACEHOLDER_NORM),
            rtol=1e-6, atol=1e-6,
        )
        self.assertEqual(first.metadata()["mood_vectors_source"], "random_placeholder")
        self.assertFalse(first.metadata()["mood_vectors_validated"])

    def test_supplied_magnitudes_are_preserved_and_bank_is_copied(self):
        self.assertTrue(torch.equal(self.steering.vectors, self.bank))
        self.assertFalse(torch.equal(
            self.steering.vectors.norm(dim=-1), torch.ones((2, len(AXES))),
        ))
        self.bank.mul_(100)
        torch.testing.assert_close(self.steering.vectors, provided_bank())
        self.assertEqual(self.steering.metadata()["mood_vectors_source"], "provided")

    def test_invalid_bank_is_rejected_before_hooks_can_be_registered(self):
        invalid = [
            self.bank[:, :-1], self.bank.to(torch.int32), self.bank * float("nan"),
            self.bank * float("inf"), torch.zeros_like(self.bank),
        ]
        for bank in invalid:
            with self.subTest(shape=tuple(bank.shape), dtype=bank.dtype):
                with self.assertRaises(ValueError):
                    MoodSteering(self.model, bank)
        self.assert_no_hooks()

    def test_unexpected_decoder_positions_fail_and_hooks_are_removed(self):
        layer = self.model.model.language_model.layers[0]
        hidden = torch.zeros(1, 3, 4)
        for positions in [None, torch.zeros(4, 1, 3), torch.zeros(1, 2)]:
            with self.subTest(position_shape=None if positions is None else tuple(positions.shape)):
                with self.assertRaisesRegex(ValueError, "Decoder text positions"):
                    with self.steering.apply([1, 0, 0, 0, 0, 0], 2):
                        layer(hidden, position_ids=positions)
                self.assert_no_hooks()


@unittest.skipIf(torch is None or find_spec("transformers") is None, "torch and transformers are optional")
class QwenArchitectureSteeringTests(unittest.TestCase):
    def test_hybrid_conditional_generation_steers_suffix_and_cached_positions(self):
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

        config = Qwen3_5Config(
            text_config={
                "vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
                "num_hidden_layers": 2, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 8,
                "linear_num_key_heads": 1, "linear_num_value_heads": 2,
                "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                "linear_conv_kernel_dim": 2, "max_position_embeddings": 64,
                "layer_types": ["linear_attention", "full_attention"],
                "rope_parameters": {
                    "rope_type": "default", "rope_theta": 10000.0,
                    "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2],
                },
                "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": None,
            },
            vision_config={
                "depth": 1, "hidden_size": 16, "intermediate_size": 32,
                "num_heads": 2, "out_hidden_size": 16, "num_position_embeddings": 16,
                "patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 2,
            },
        )
        with torch.random.fork_rng():
            torch.manual_seed(123)
            model = Qwen3_5ForConditionalGeneration(config).eval()
        model.generation_config.eos_token_id = None
        model.generation_config.pad_token_id = 0
        model.generation_config.bos_token_id = 1
        layers = model.model.language_model.layers
        bank = torch.zeros(2, len(AXES), 16)
        bank[:, :, 0] = 0.25
        steering = MoodSteering(model, bank)
        before = [[] for _ in layers]
        after = [[] for _ in layers]
        handles = []

        def capture(target, index):
            def hook(module, args, kwargs):
                target[index].append({
                    "hidden": args[0].detach().clone(),
                    "positions": kwargs["position_ids"].detach().clone(),
                    "cache": kwargs.get("past_key_values"),
                })
            return hook

        initial_weights = [parameter.detach().clone() for parameter in model.parameters()]
        try:
            for index, layer in enumerate(layers):
                handles.append(layer.register_forward_pre_hook(capture(before, index), with_kwargs=True))
            with steering.apply([2, 0, 0, 0, 0, 0], 4), torch.inference_mode():
                for index, layer in enumerate(layers):
                    handles.append(layer.register_forward_pre_hook(capture(after, index), with_kwargs=True))
                output = model.generate(
                    input_ids=torch.tensor([[1, 5, 6, 7, 8, 9]]),
                    attention_mask=torch.ones(1, 6, dtype=torch.long),
                    do_sample=False, use_cache=True, max_new_tokens=3,
                    pad_token_id=0, eos_token_id=None, logits_to_keep=1,
                )
            self.assertEqual(tuple(output.shape), (1, 9))
            for index in range(len(layers)):
                self.assertEqual(len(before[index]), 3)
                self.assertEqual(len(after[index]), 3)
                expected_positions = [torch.arange(6).reshape(1, -1), torch.tensor([[6]]), torch.tensor([[7]])]
                for original, modified, positions in zip(before[index], after[index], expected_positions):
                    self.assertEqual(original["positions"].ndim, 2)
                    self.assertTrue(torch.equal(original["positions"], positions))
                    self.assertIsNotNone(original["cache"])
                    self.assertIs(original["cache"], before[index][0]["cache"])
                    offset = 2 * bank[index, 0]
                    expected = torch.where(
                        (positions >= 4).unsqueeze(-1), original["hidden"] + offset, original["hidden"],
                    )
                    torch.testing.assert_close(modified["hidden"], expected, rtol=0, atol=0)
                self.assertTrue(torch.equal(before[index][0]["hidden"][:, :4], after[index][0]["hidden"][:, :4]))
            for original, parameter in zip(initial_weights, model.parameters()):
                self.assertTrue(torch.equal(original, parameter))
        finally:
            for handle in handles:
                handle.remove()
        for layer in layers:
            self.assertEqual(len(layer._forward_pre_hooks), 0)


if __name__ == "__main__":
    unittest.main()
