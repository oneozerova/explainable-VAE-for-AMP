"""Conditional VAE architecture used by the project."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ..config import CONDITION_COLUMNS
from ..data.tokenizer import VocabBundle


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, latent_dim: int, cond_dim: int, pad_idx: int, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim + cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim + cond_dim, latent_dim)

    def forward(self, x, lengths, cond):
        embedded = self.drop(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        hidden = h_n[-1]
        hidden_cond = torch.cat([hidden, cond], dim=1)
        return self.fc_mu(hidden_cond), self.fc_logvar(hidden_cond)


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, latent_dim: int, cond_dim: int, pad_idx: int, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.drop = nn.Dropout(dropout)
        self.init_h = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.init_c = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.lstm = nn.LSTM(embed_dim + latent_dim + cond_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x,
        z,
        cond,
        pad_idx: int,
        unk_idx: int,
        sos_idx: int | None = None,
        eos_idx: int | None = None,
        word_dropout_rate: float = 0.0,
    ):
        batch_size, seq_len = x.shape
        z_cond = torch.cat([z, cond], dim=1)
        h0 = torch.tanh(self.init_h(z_cond)).unsqueeze(0)
        c0 = torch.tanh(self.init_c(z_cond)).unsqueeze(0)

        if self.training and word_dropout_rate > 0:
            # Bowman 2016 word dropout: never replace PAD, SOS, or EOS. SOS at pos 0
            # is the generation start signal; replacing it breaks teacher forcing.
            keep = x != pad_idx
            if sos_idx is not None:
                keep = keep & (x != sos_idx)
            if eos_idx is not None:
                keep = keep & (x != eos_idx)
            mask = (torch.rand(batch_size, seq_len, device=x.device) < word_dropout_rate) & keep
            x = x.clone()
            x[mask] = unk_idx

        embedded = self.drop(self.embedding(x))
        z_expand = z.unsqueeze(1).expand(-1, seq_len, -1)
        cond_expand = cond.unsqueeze(1).expand(-1, seq_len, -1)
        lstm_input = torch.cat([embedded, z_expand, cond_expand], dim=2)
        outputs, _ = self.lstm(lstm_input, (h0, c0))
        return self.fc_out(outputs)


class CVAE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        enc_hidden: int,
        dec_hidden: int,
        latent_dim: int,
        cond_dim: int,
        pad_idx: int,
        unk_idx: int,
        sos_idx: int | None = None,
        eos_idx: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_dim, enc_hidden, latent_dim, cond_dim, pad_idx, num_layers=num_layers, dropout=dropout)
        self.decoder = Decoder(vocab_size, embed_dim, dec_hidden, latent_dim, cond_dim, pad_idx, num_layers=num_layers, dropout=dropout)
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, src, lengths, cond, word_dropout_rate: float = 0.0):
        mu, logvar = self.encoder(src, lengths, cond)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(
            src[:, :-1],
            z,
            cond,
            self.pad_idx,
            self.unk_idx,
            sos_idx=self.sos_idx,
            eos_idx=self.eos_idx,
            word_dropout_rate=word_dropout_rate,
        )
        return logits, src[:, 1:], mu, logvar


def infer_cvae_hparams_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    latent_dim = int(state_dict["encoder.fc_mu.weight"].shape[0])
    cond_dim = int(state_dict["decoder.init_h.weight"].shape[1] - latent_dim)
    enc_hidden = int(state_dict["encoder.fc_mu.weight"].shape[1] - cond_dim)
    dec_hidden = int(state_dict["decoder.init_h.weight"].shape[0])
    embed_dim = int(state_dict["encoder.embedding.weight"].shape[1])
    encoder_lstm_layers = len([k for k in state_dict if k.startswith("encoder.lstm.weight_ih_l")])
    decoder_lstm_layers = len([k for k in state_dict if k.startswith("decoder.lstm.weight_ih_l")])
    num_layers = max(encoder_lstm_layers, decoder_lstm_layers, 1)
    return {
        "embed_dim": embed_dim,
        "enc_hidden": enc_hidden,
        "dec_hidden": dec_hidden,
        "latent_dim": latent_dim,
        "cond_dim": cond_dim,
        "num_layers": num_layers,
    }


def build_cvae(vocab: VocabBundle, embed_dim: int = 128, enc_hidden: int = 1024, dec_hidden: int = 512, latent_dim: int = 32, cond_dim: int | None = None, num_layers: int = 1, dropout: float = 0.2) -> CVAE:
    cond_dim = int(cond_dim or len(CONDITION_COLUMNS))
    return CVAE(
        vocab_size=len(vocab.char2idx),
        embed_dim=embed_dim,
        enc_hidden=enc_hidden,
        dec_hidden=dec_hidden,
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        pad_idx=vocab.pad_idx,
        unk_idx=vocab.unk_idx,
        sos_idx=vocab.sos_idx,
        eos_idx=vocab.eos_idx,
        num_layers=num_layers,
        dropout=dropout,
    )


def load_cvae_checkpoint(checkpoint_path: str | Path, vocab: VocabBundle, device=None, **kwargs) -> CVAE:
    checkpoint_path = Path(checkpoint_path)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    inferred_hparams = infer_cvae_hparams_from_state_dict(state_dict)
    model_kwargs = {**inferred_hparams, **kwargs}
    model = build_cvae(vocab, **model_kwargs)
    if device is not None:
        model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model
