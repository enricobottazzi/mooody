"""Tensor and pinned-Qwen checks; the lightweight backend suite needs no torch."""

from importlib.util import find_spec
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

from deployment.core import AXES, MAX_MOOD_COEFFICIENT, MOOD_COEFFICIENTS
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
            self.assertEqual(len(layer._forward_hooks), 0)

    def test_signed_sum_masks_prefill_and_all_cached_tokens(self):
        coefficients = [MAX_MOOD_COEFFICIENT * level for level in [1, -0.5, 0.5, 0, -1, 0.5]]
        expected_offsets = torch.tensor([
            [-0.5, -1.0, 1.5, -0.125], [-0.5, -1.0, 1.5, -0.125],
        ]) * MAX_MOOD_COEFFICIENT
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

    def test_fractional_coefficients_mix_actual_increments_without_rescaling(self):
        coefficients = [MAX_MOOD_COEFFICIENT * level for level in [-1, -0.5, 0, 0.25, 0.5, 1.0]]
        expected = sum(level * self.steering.incremental_vectors[:, axis]
                       for axis, level in enumerate(coefficients))
        torch.testing.assert_close(self.steering.offsets(coefficients), expected, rtol=0, atol=0)
        for level in MOOD_COEFFICIENTS:
            with self.subTest(level=level):
                torch.testing.assert_close(
                    self.steering.offsets([level, 0, 0, 0, 0, 0]),
                    level * self.steering.incremental_vectors[:, 0], rtol=0, atol=0,
                )

    def test_invalid_coefficients_fail_before_any_hooks_are_installed(self):
        invalid = (True, False, 2, -2, 1, -1, MAX_MOOD_COEFFICIENT + 0.001,
                   -MAX_MOOD_COEFFICIENT - 0.001, float("nan"),
                   float("inf"), -float("inf"), "0.5", None, 10 ** 1000)
        for level in invalid:
            with self.subTest(level=level), self.assertRaises(ValueError):
                with self.steering.apply([level, 0, 0, 0, 0, 0], 0):
                    self.fail("Invalid coefficients must not start generation")
            self.assert_no_hooks()
        with self.assertRaises(ValueError):
            self.steering.offsets([0] * 5)

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
            with self.steering.apply([MAX_MOOD_COEFFICIENT, 0, 0, 0, 0, 0], 1):
                first = layer(hidden, position_ids=positions)
                torch.testing.assert_close(first[:, 1:], (MAX_MOOD_COEFFICIENT * self.bank[0, 0]).expand(1, 2, -1))
                raise RuntimeError("generation failed")
        self.assert_no_hooks()
        self.assertIs(layer(hidden), hidden)
        with self.steering.apply([-0.5 * MAX_MOOD_COEFFICIENT, 0, 0, 0, 0, 0], 2):
            second = layer(hidden, position_ids=positions)
            self.assertTrue(torch.equal(second[:, :2], hidden[:, :2]))
            torch.testing.assert_close(second[:, 2:], (-0.5 * MAX_MOOD_COEFFICIENT * self.bank[0, 0]).reshape(1, 1, -1))
        self.assert_no_hooks()

    def test_missing_real_bank_is_rejected_without_consuming_rng(self):
        rng = torch.get_rng_state().clone()
        with self.assertRaisesRegex(ValueError, "real extracted persona vector bank"):
            MoodSteering(self.model)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assert_no_hooks()

    def test_increment_is_added_after_block_computation(self):
        class ScalingLayer(torch.nn.Module):
            def forward(self, hidden_states, position_ids=None):
                return hidden_states * 2

        self.model.model.language_model.layers[0] = ScalingLayer()
        steering = MoodSteering(self.model, self.bank)
        hidden = torch.ones(1, 3, 4)
        positions = torch.arange(3).reshape(1, -1)
        with steering.apply([MAX_MOOD_COEFFICIENT, 0, 0, 0, 0, 0], 2):
            first = steering.layers[0](hidden, position_ids=positions)
            expected = hidden * 2
            expected[:, 2:] += MAX_MOOD_COEFFICIENT * self.bank[0, 0]
            torch.testing.assert_close(first, expected)
            second = steering.layers[1](first, position_ids=positions)
            expected[:, 2:] += MAX_MOOD_COEFFICIENT * (self.bank[1, 0] - self.bank[0, 0])
            torch.testing.assert_close(second, expected)
        self.assert_no_hooks()

    def test_generated_control_tokens_are_excluded_but_final_prompt_token_is_included(self):
        class TextModel(torch.nn.Module):
            def __init__(self, fixture):
                super().__init__()
                self.device, self.config, self.model = fixture.device, fixture.config, fixture.model

            def forward(self, input_ids, position_ids):
                hidden = torch.zeros(*input_ids.shape, 4)
                for layer in self.model.language_model.layers:
                    hidden = layer(hidden, position_ids=position_ids)
                return hidden

        model = TextModel(self.model)
        steering = MoodSteering(model, self.bank, excluded_content_token_ids=[10, 11, 12], thinking_token_ids=(11, 12))
        with steering.apply([MAX_MOOD_COEFFICIENT, 0, 0, 0, 0, 0], 2):
            prefill = model(input_ids=torch.tensor([[5, 10, 10]]), position_ids=torch.tensor([[0, 1, 2]]))
            torch.testing.assert_close(prefill[:, :2], torch.zeros(1, 2, 4))
            torch.testing.assert_close(prefill[:, 2:], (MAX_MOOD_COEFFICIENT * self.bank[1, 0]).reshape(1, 1, 4))
            control = model(input_ids=torch.tensor([[10]]), position_ids=torch.tensor([[3]]))
            torch.testing.assert_close(control, torch.zeros(1, 1, 4))
            content = model(input_ids=torch.tensor([[5]]), position_ids=torch.tensor([[4]]))
            torch.testing.assert_close(content, prefill[:, 2:])
            for position, token in ((5, 11), (6, 5), (7, 12)):
                thinking = model(input_ids=torch.tensor([[token]]), position_ids=torch.tensor([[position]]))
                torch.testing.assert_close(thinking, torch.zeros(1, 1, 4))
            resumed = model(input_ids=torch.tensor([[5]]), position_ids=torch.tensor([[8]]))
            torch.testing.assert_close(resumed, content)
        self.assertEqual(len(model._forward_pre_hooks), 0)
        self.assert_no_hooks()

    def test_supplied_magnitudes_are_preserved_and_bank_is_copied(self):
        self.assertTrue(torch.equal(self.steering.vectors, self.bank))
        self.assertFalse(torch.equal(
            self.steering.vectors.norm(dim=-1), torch.ones((2, len(AXES))),
        ))
        self.bank.mul_(100)
        torch.testing.assert_close(self.steering.vectors, provided_bank())
        self.assertEqual(self.steering.metadata()["mood_vectors_source"], "provided")

    def test_incremental_offsets_telescope_to_each_raw_layer_for_signed_axis_mixtures(self):
        model = tensor_model(layer_count=7)
        bank = torch.arange(7 * len(AXES) * 4, dtype=torch.float32).reshape(7, len(AXES), 4) / 4
        bank[1::2] *= -1
        original = bank.clone()
        steering = MoodSteering(model, bank)
        coefficients = [MAX_MOOD_COEFFICIENT * level for level in [1, -0.5, 0, 0.5, -1, 0.5]]
        # Independent reference combines raw directions first, then takes their
        # layer differences. Prefix sums must recover each raw combined vector.
        raw_mix = sum(level * bank[:, axis] for axis, level in enumerate(coefficients))
        offsets = steering.offsets(coefficients)
        torch.testing.assert_close(offsets[0], raw_mix[0], rtol=0, atol=0)
        torch.testing.assert_close(offsets[1:], raw_mix[1:] - raw_mix[:-1], rtol=0, atol=0)
        torch.testing.assert_close(offsets.cumsum(dim=0), raw_mix, rtol=0, atol=0)
        hidden = torch.zeros(1, 2, 4)
        with steering.apply(coefficients, 1):
            for layer, raw_direction in zip(steering.layers, raw_mix):
                hidden = layer(hidden, position_ids=torch.tensor([[0, 1]]))
                torch.testing.assert_close(hidden[:, 0], torch.zeros(1, 4), rtol=0, atol=0)
                torch.testing.assert_close(hidden[:, 1], raw_direction.reshape(1, -1), rtol=0, atol=0)
        self.assertTrue(torch.equal(steering.vectors, original))
        self.assertTrue(torch.equal(bank, original))
        self.assertEqual(len(model.model.language_model.layers[0]._forward_hooks), 0)

    def test_constant_raw_bank_injects_only_at_first_layer_with_zero_predecessor(self):
        model = tensor_model(layer_count=5)
        bank = self.bank[0].expand(5, -1, -1).clone()
        steering = MoodSteering(model, bank)
        offsets = steering.offsets([0, 0, 0, 0.5 * MAX_MOOD_COEFFICIENT, 0, 0])
        torch.testing.assert_close(offsets[0], 0.5 * MAX_MOOD_COEFFICIENT * bank[0, 3], rtol=0, atol=0)
        self.assertTrue(torch.equal(offsets[1:], torch.zeros(4, 4)))
        hidden = torch.zeros(1, 1, 4)
        with steering.apply([0, 0, 0, 0.5 * MAX_MOOD_COEFFICIENT, 0, 0], 0):
            for layer in steering.layers:
                hidden = layer(hidden, position_ids=torch.tensor([[0]]))
                torch.testing.assert_close(hidden, (0.5 * MAX_MOOD_COEFFICIENT * bank[0, 3]).reshape(1, 1, 4), rtol=0, atol=0)
        self.assertTrue(torch.equal(steering.vectors, bank))

    def test_metadata_separates_current_runtime_from_immutable_publication_method(self):
        steering = MoodSteering(self.model, self.bank, {
            "mood_vectors_source": "persona_vectors",
            "mood_vectors_published_inference_method": "direct_raw_all_layers",
        })
        metadata = steering.metadata()
        self.assertEqual(metadata["steering_method"], "paper_incremental_all_layers")
        self.assertEqual(metadata["steering_incremental_definition"], "raw_layer_vector_minus_previous_layer_vector")
        self.assertEqual(metadata["steering_first_layer_previous_vector"], "zero")
        self.assertEqual(metadata["mood_coefficients"], [-0.25, -0.125, 0, 0.125, 0.25])
        self.assertEqual(metadata["mood_vectors_published_inference_method"], "direct_raw_all_layers")
        self.assertFalse(metadata["mood_vectors_validated"])

    def test_invalid_bank_is_rejected_before_hooks_can_be_registered(self):
        invalid = [
            self.bank[:, :-1], self.bank.to(torch.int32), self.bank.to(torch.bfloat16), self.bank * float("nan"),
            self.bank * float("inf"),
        ]
        for bank in invalid:
            with self.subTest(shape=tuple(bank.shape), dtype=bank.dtype):
                with self.assertRaises(ValueError):
                    MoodSteering(self.model, bank)
        self.assert_no_hooks()

    def test_zero_layer_directions_and_entire_zero_traits_are_retained_and_flagged(self):
        self.bank[0, 0] = 0
        self.bank[:, 1] = 0
        steering = MoodSteering(self.model, self.bank)
        self.assertTrue(torch.equal(steering.vectors, self.bank))
        self.assertEqual(tuple(steering.vectors.shape), (2, len(AXES), 4))
        metadata = steering.metadata()
        self.assertEqual(metadata["mood_vectors_zero_layer_trait_count"], 3)
        self.assertEqual(metadata["mood_vectors_zero_layers"][AXES[0]], [1])
        self.assertEqual(metadata["mood_vectors_entirely_zero_traits"], [AXES[1]])
        self.assertFalse(metadata["mood_vectors_entirely_zero_bank"])
        torch.testing.assert_close(steering.offsets([0, MAX_MOOD_COEFFICIENT, 0, 0, 0, 0]), torch.zeros(2, 4))

        all_zero = MoodSteering(self.model, torch.zeros_like(self.bank))
        self.assertTrue(all_zero.metadata()["mood_vectors_entirely_zero_bank"])
        self.assertEqual(all_zero.metadata()["mood_vectors_zero_layer_trait_count"], 2 * len(AXES))
        self.assertFalse(all_zero.metadata()["mood_vectors_validated"])

    def test_unexpected_decoder_positions_fail_and_hooks_are_removed(self):
        layer = self.model.model.language_model.layers[0]
        hidden = torch.zeros(1, 3, 4)
        for positions in [None, torch.zeros(4, 1, 3), torch.zeros(1, 2)]:
            with self.subTest(position_shape=None if positions is None else tuple(positions.shape)):
                with self.assertRaisesRegex(ValueError, "Decoder text positions"):
                    with self.steering.apply([MAX_MOOD_COEFFICIENT, 0, 0, 0, 0, 0], 2):
                        layer(hidden, position_ids=positions)
                self.assert_no_hooks()


@unittest.skipIf(torch is None or find_spec("transformers") is None, "torch and transformers are optional")
class QwenArchitectureSteeringTests(unittest.TestCase):
    def test_hybrid_conditional_generation_steers_block_outputs_and_cached_positions(self):
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
        bank[0, :, 0] = 0.25
        bank[1, :, 0] = 0.375
        excluded = [0, 1, 2]
        steering = MoodSteering(model, bank, excluded_content_token_ids=excluded)
        before = [[] for _ in layers]
        after = [[] for _ in layers]
        forward_input_ids = []
        handles = []

        def capture_tokens(module, args, kwargs):
            forward_input_ids.append(kwargs["input_ids"].detach().clone())

        def force_tokens(input_ids, logits):
            # Exercise one excluded generated control ID followed by content,
            # independently of which IDs the randomly initialized head prefers.
            token = (0, 5, 6)[input_ids.shape[-1] - 6]
            logits.fill_(float("-inf"))
            logits[:, token] = 0
            return logits

        def capture(target, index):
            def hook(module, args, kwargs, output):
                target[index].append({
                    "hidden": output.detach().clone(),
                    "positions": kwargs["position_ids"].detach().clone(),
                    "cache": kwargs.get("past_key_values"),
                })
            return hook

        initial_weights = [parameter.detach().clone() for parameter in model.parameters()]
        try:
            handles.append(model.register_forward_pre_hook(capture_tokens, with_kwargs=True))
            for index, layer in enumerate(layers):
                handles.append(layer.register_forward_hook(capture(before, index), with_kwargs=True))
            with steering.apply([0.5 * MAX_MOOD_COEFFICIENT, 0, 0, 0, 0, 0], 5), torch.inference_mode():
                for index, layer in enumerate(layers):
                    handles.append(layer.register_forward_hook(capture(after, index), with_kwargs=True))
                output = model.generate(
                    input_ids=torch.tensor([[1, 5, 6, 7, 8, 9]]),
                    attention_mask=torch.ones(1, 6, dtype=torch.long),
                    do_sample=False, use_cache=True, max_new_tokens=3,
                    pad_token_id=0, eos_token_id=None, logits_to_keep=1,
                    logits_processor=[force_tokens],
                )
            self.assertEqual(tuple(output.shape), (1, 9))
            for index in range(len(layers)):
                self.assertEqual(len(before[index]), 3)
                self.assertEqual(len(after[index]), 3)
                expected_positions = [torch.arange(6).reshape(1, -1), torch.tensor([[6]]), torch.tensor([[7]])]
                for original, modified, positions, current_ids in zip(before[index], after[index], expected_positions, forward_input_ids):
                    self.assertEqual(original["positions"].ndim, 2)
                    self.assertTrue(torch.equal(original["positions"], positions))
                    self.assertIsNotNone(original["cache"])
                    self.assertIs(original["cache"], before[index][0]["cache"])
                    offset = 0.5 * MAX_MOOD_COEFFICIENT * (bank[index, 0] - (bank[index - 1, 0] if index else 0))
                    eligible = (positions == 5) | (
                        (positions > 5) & ~torch.isin(current_ids, torch.tensor(excluded))
                    )
                    expected = torch.where(
                        eligible.unsqueeze(-1), original["hidden"] + offset, original["hidden"],
                    )
                    torch.testing.assert_close(modified["hidden"], expected, rtol=0, atol=0)
                self.assertTrue(torch.equal(before[index][0]["hidden"][:, :5], after[index][0]["hidden"][:, :5]))
                self.assertTrue(torch.equal(before[index][1]["hidden"], after[index][1]["hidden"]))
            for original, parameter in zip(initial_weights, model.parameters()):
                self.assertTrue(torch.equal(original, parameter))
        finally:
            for handle in handles:
                handle.remove()
        for layer in layers:
            self.assertEqual(len(layer._forward_pre_hooks), 0)
            self.assertEqual(len(layer._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
