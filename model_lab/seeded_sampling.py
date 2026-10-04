"""Independent native warped sampling for the audited batched continuation."""
import torch
import copy
from transformers import LogitsProcessor, LogitsProcessorList
from transformers.generation.logits_process import (
    TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, LogitNormalization,
)


def validate_supported_config(config):
    # 5.18 stores unspecified options as None and resolves global defaults only
    # inside generate. Resolve identically before choosing the sampler warpers.
    config = copy.deepcopy(config)
    config.update(**config._get_default_generation_params(), defaults_only=True)
    # Native processors still run before our draw. Fail closed on transforms
    # after custom processors that this narrowly scoped experiment cannot mirror.
    inactive = {"top_h": None, "min_p": None, "typical_p": 1.0,
                "epsilon_cutoff": 0.0, "eta_cutoff": 0.0,
                "watermarking_config": None,
                "num_beams": 1, "num_return_sequences": 1,
                "repetition_penalty": 1.0, "no_repeat_ngram_size": 0,
                "encoder_repetition_penalty": 1.0, "guidance_scale": None}
    for name, expected in inactive.items():
        if getattr(config, name, expected) != expected:
            raise ValueError(f"Batched continuation does not support active {name}")
    return config


class IndependentRowSampler(LogitsProcessor):
    """Draw from native warped scores with a dedicated generator per row.

    Transformers 5.18 merges custom processors before sampling warpers. We
    reproduce the active native warpers for the draw, then return a single
    finite score. Reapplication by HF cannot change that one-token support.
    HF retains its normal cache, EOS, stopping, and padding behavior.
    """
    def __init__(self, seeds, config, device):
        config = validate_supported_config(config)
        self.generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]
        self.warpers = LogitsProcessorList()
        if config.temperature is not None and config.temperature != 1.0:
            self.warpers.append(TemperatureLogitsWarper(config.temperature))
        if config.top_k is not None and config.top_k != 0:
            self.warpers.append(TopKLogitsWarper(config.top_k, min_tokens_to_keep=1))
        if config.top_p is not None and config.top_p < 1.0:
            self.warpers.append(TopPLogitsWarper(config.top_p, min_tokens_to_keep=1))
        if config.renormalize_logits is True:
            self.warpers.append(LogitNormalization())
        self.draws = 0

    def __call__(self, input_ids, scores):
        if scores.shape[0] != len(self.generators):
            raise ValueError("Sampling row order/width changed")
        warped = self.warpers(input_ids, scores)
        probabilities = torch.softmax(warped, dim=-1)
        chosen = torch.cat([torch.multinomial(probabilities[index:index + 1], 1, generator=generator)
                            for index, generator in enumerate(self.generators)], dim=0)
        self.draws += 1
        return torch.full_like(scores, -torch.inf).scatter_(1, chosen, 0.0)


def left_padded_inputs(prompt_ids, pad_token_id, device):
    width = max(map(len, prompt_ids))
    inputs = torch.full((len(prompt_ids), width), pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(inputs)
    for index, ids in enumerate(prompt_ids):
        inputs[index, -len(ids):] = torch.tensor(ids, dtype=torch.long, device=device)
        mask[index, -len(ids):] = 1
    return inputs, mask
