"""Generation helpers for the conditional VAE."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import torch

from ..config import CONDITION_COLUMNS, GENERATION_CONDITION_COLUMNS, VALID_AMINO_ACIDS
from ..data.tokenizer import VocabBundle


def condition_vector(**kwargs) -> list[int]:
    return [int(kwargs.get(name, 0)) for name in CONDITION_COLUMNS]


def generation_condition_vector(**kwargs) -> list[int]:
    """Build a condition vector for generation using the active 6-label schema."""
    return [int(kwargs.get(name, 0)) for name in GENERATION_CONDITION_COLUMNS]


def align_condition_to_model(condition: Iterable[int | float], model) -> list[float]:
    values = [float(x) for x in condition]
    model_cond_dim = int(getattr(model, "cond_dim", len(values)))
    if len(values) == model_cond_dim:
        return values
    raise ValueError(
        f"Condition vector has length {len(values)}, but the loaded cVAE expects {model_cond_dim}. "
        "Use a checkpoint trained with the current 6-label schema or supply a matching condition vector."
    )


def normalize_condition_name(name: str) -> str:
    name = str(name).lower().replace("+", "plus")
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def is_valid_sequence(seq: str, min_len: int = 5, max_len: int = 50, valid_aa: Iterable[str] = VALID_AMINO_ACIDS) -> bool:
    # Match normalization in evaluation/generation_metrics.validity_rate.
    if not seq:
        return False
    seq = str(seq).strip().upper()
    if not seq:
        return False
    if not (min_len <= len(seq) <= max_len):
        return False
    return set(seq).issubset(set(valid_aa))


def _apply_top_k_top_p(logits: torch.Tensor, top_k: int | None, top_p: float | None) -> torch.Tensor:
    """Mask logits for top-k / top-p (nucleus) sampling. Operates on last dim."""
    if top_k is not None and top_k > 0:
        k = min(int(top_k), logits.size(-1))
        kth_vals = torch.topk(logits, k, dim=-1).values[..., -1:]
        logits = torch.where(logits < kth_vals, torch.full_like(logits, float("-inf")), logits)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        # Keep tokens where cumulative prob < top_p, plus the one that crosses threshold.
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)
    return logits


def decode_from_z_cond(
    model,
    z: torch.Tensor,
    cond_t: torch.Tensor,
    vocab: VocabBundle,
    max_gen_len: int | None = None,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    model.eval()
    max_gen_len = int(max_gen_len or vocab.max_len)
    z_cond = torch.cat([z, cond_t], dim=1)
    h = torch.tanh(model.decoder.init_h(z_cond)).unsqueeze(0)
    c = torch.tanh(model.decoder.init_c(z_cond)).unsqueeze(0)

    token = torch.full((z.size(0), 1), vocab.sos_idx, dtype=torch.long, device=z.device)
    finished = torch.zeros(z.size(0), dtype=torch.bool, device=z.device)
    sequences = [[] for _ in range(z.size(0))]

    # Never sample PAD, SOS, UNK. EOS is allowed (signals stop).
    bad_idx = torch.tensor(
        [vocab.pad_idx, vocab.sos_idx, vocab.unk_idx],
        device=z.device,
        dtype=torch.long,
    )

    for _ in range(max_gen_len):
        emb = model.decoder.embedding(token)
        dec_inp = torch.cat([emb, z_cond.unsqueeze(1)], dim=-1)
        out, (h, c) = model.decoder.lstm(dec_inp, (h, c))
        logits = model.decoder.fc_out(out[:, -1, :])
        logits[..., bad_idx] = float("-inf")
        logits = logits / temperature
        logits = _apply_top_k_top_p(logits, top_k, top_p)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1, generator=generator)
        token = next_token

        for i, idx in enumerate(next_token.squeeze(1).tolist()):
            if finished[i]:
                continue
            if idx == vocab.eos_idx:
                finished[i] = True
            elif idx not in (vocab.pad_idx, vocab.sos_idx, vocab.unk_idx):
                sequences[i].append(vocab.idx2char.get(idx, ""))

        if bool(finished.all()):
            break

    return ["".join(seq) for seq in sequences]


@torch.no_grad()
def generate_batch(
    model,
    vocab: VocabBundle,
    condition: Iterable[int],
    n_samples: int = 128,
    temperature: float = 0.8,
    alpha: float = 1.0,
    device=None,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
):
    device = device or next(model.parameters()).device
    aligned_condition = align_condition_to_model(condition, model)
    cond_arr = np.asarray(aligned_condition, dtype=np.float32) * float(alpha)
    cond_t = torch.tensor(cond_arr, dtype=torch.float32, device=device).unsqueeze(0).repeat(n_samples, 1)
    z = torch.randn(n_samples, model.latent_dim, device=device, generator=generator)
    return decode_from_z_cond(
        model,
        z,
        cond_t,
        vocab=vocab,
        max_gen_len=vocab.max_len,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        generator=generator,
    )


# Alias expected by spec ("generate"); keep generate_batch as existing name too.
generate = generate_batch


@torch.no_grad()
def interpolate_latent(
    model,
    vocab: VocabBundle,
    z_start: torch.Tensor,
    z_end: torch.Tensor,
    condition: Iterable[int],
    n_steps: int = 9,
    temperature: float = 0.8,
    samples_per_step: int = 1,
    device=None,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> list[tuple[float, list[str]]]:
    device = device or next(model.parameters()).device
    aligned_condition = align_condition_to_model(condition, model)
    cond_arr = np.asarray(aligned_condition, dtype=np.float32)
    cond_base = torch.tensor(cond_arr, dtype=torch.float32, device=device)

    z_start = z_start.to(device).view(-1)
    z_end = z_end.to(device).view(-1)

    alphas = torch.linspace(0.0, 1.0, n_steps)
    results: list[tuple[float, list[str]]] = []
    for a in alphas.tolist():
        z_mix = (1.0 - a) * z_start + a * z_end
        z_batch = z_mix.unsqueeze(0).repeat(samples_per_step, 1)
        cond_batch = cond_base.unsqueeze(0).repeat(samples_per_step, 1)
        seqs = decode_from_z_cond(
            model,
            z_batch,
            cond_batch,
            vocab=vocab,
            max_gen_len=vocab.max_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )
        results.append((float(a), seqs))
    return results


@torch.no_grad()
def interpolate_conditions(
    model,
    vocab: VocabBundle,
    cond_start: Iterable[int],
    cond_end: Iterable[int],
    z: torch.Tensor | None = None,
    n_steps: int = 9,
    temperature: float = 0.8,
    samples_per_step: int = 1,
    device=None,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> list[tuple[float, list[str]]]:
    device = device or next(model.parameters()).device
    aligned_start = align_condition_to_model(cond_start, model)
    aligned_end = align_condition_to_model(cond_end, model)
    c_start = torch.tensor(aligned_start, dtype=torch.float32, device=device)
    c_end = torch.tensor(aligned_end, dtype=torch.float32, device=device)

    if z is None:
        z = torch.randn(model.latent_dim, device=device, generator=generator)
    z = z.to(device).view(-1)

    alphas = torch.linspace(0.0, 1.0, n_steps)
    results: list[tuple[float, list[str]]] = []
    for a in alphas.tolist():
        c_mix = (1.0 - a) * c_start + a * c_end
        z_batch = z.unsqueeze(0).repeat(samples_per_step, 1)
        cond_batch = c_mix.unsqueeze(0).repeat(samples_per_step, 1)
        seqs = decode_from_z_cond(
            model,
            z_batch,
            cond_batch,
            vocab=vocab,
            max_gen_len=vocab.max_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )
        results.append((float(a), seqs))
    return results
