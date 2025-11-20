# vit_llama_latent_model.py

import math
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def _init_weights(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
        if getattr(m, "weight", None) is not None:
            nn.init.xavier_uniform_(m.weight)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.zeros_(m.bias)


# ------------------------------------------------------------
# Token Embeddings
# ------------------------------------------------------------
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


# ------------------------------------------------------------
# SINGLE LATENT TOKEN PROJECTOR
# ------------------------------------------------------------
class SingleLatentToken(nn.Module):
    """
    Takes the whole latent vector (B, latent_dim)
    -> produces ONE token (B, 1, d_model)
    """

    def __init__(self, latent_dim, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Optional modality embedding
        self.latent_type = nn.Parameter(torch.zeros(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.apply(_init_weights)

    def forward(self, latent_vec):
        """
        latent_vec: (B, latent_dim)
        """
        if latent_vec.dim() == 3:   # safety
            latent_vec = latent_vec.squeeze(-1)  # (B, 50)

        x = self.proj(latent_vec)     # (B, d_model)
        x = x.unsqueeze(1)            # (B, 1, d_model)
        x = x + self.latent_type
        return self.norm(x)


# ------------------------------------------------------------
# LATENT → LLaMA DECODER
# ------------------------------------------------------------
class LatentLLaMA_SingleToken(nn.Module):
    """
    Sequence is:
        [ LATENT_TOKEN , MECH_TOKEN , TARGET_TOKENS ]
    """

    def __init__(
        self,
        latent_dim: int,               # should be 50
        tgt_seq_len: int = 25,
        vocab_size: int = 204,
        d_model: int = 512,
        h: int = 8,
        N: int = 6,
        num_labels: int = 17,
        dropout: float = 0.1,
        pad_token_id: int = 2,
        debug: bool = False,
    ):
        super().__init__()
        self.tgt_seq_len = tgt_seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id
        self.debug = debug

        # ---- Embeddings ----
        self.tgt_embed = InputEmbeddings(d_model, vocab_size)
        self.mech_embed = nn.Embedding(num_labels, d_model)
        self.tgt_pos = nn.Embedding(tgt_seq_len + 1, d_model)

        # ---- Latent token ----
        self.latent_token = SingleLatentToken(latent_dim, d_model, dropout)

        # ---- LLaMA config ----
        llama_cfg = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=d_model,
            intermediate_size=4 * d_model,
            num_attention_heads=h,
            num_hidden_layers=N,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            use_cache=False,
        )

        self.llama = LlamaModel(llama_cfg)

        # freeze unused embedding
        for p in self.llama.embed_tokens.parameters():
            p.requires_grad = False

        # output projection
        self.proj = nn.Linear(d_model, vocab_size)

        print("🔥 Initialized LatentLLaMA_SingleToken (one-token latent conditioning)")

    # --------------------------------------------------------
    def forward(self, decoder_input_ids, _, latent_vec, mech_labels):
        """
        decoder_input_ids: (B, T)
        latent_vec:       (B, latent_dim = 50)
        mech_labels:      (B,)
        """
        B, T = decoder_input_ids.shape
        device = decoder_input_ids.device

        # ---- latent ----
        lat_tok = self.latent_token(latent_vec)     # (B, 1, D)
        if self.debug: print("latent:", lat_tok.shape)

        # ---- mech ----
        mech_tok = self.mech_embed(mech_labels).unsqueeze(1)  # (B, 1, D)
        if self.debug: print("mech:", mech_tok.shape)

        # ---- targets ----
        tgt_emb = self.tgt_embed(decoder_input_ids)  # (B, T, D)
        if self.debug: print("targets:", tgt_emb.shape)

        pos_ids = torch.arange(T + 1, device=device).unsqueeze(0)
        mech_tok = mech_tok + self.tgt_pos(pos_ids[:, :1])
        tgt_emb = tgt_emb + self.tgt_pos(pos_ids[:, 1:])

        # ---- combine ----
        full_seq = torch.cat([lat_tok, mech_tok, tgt_emb], dim=1)
        attn_mask = torch.ones(B, full_seq.size(1), device=device, dtype=torch.long)

        # ---- LLaMA ----
        out = self.llama(inputs_embeds=full_seq, attention_mask=attn_mask)
        hidden = out.last_hidden_state

        # skip latent + mech → get target token outputs
        dec_out = hidden[:, 2:, :]     # (B, T, D)
        logits = self.proj(dec_out)    # (B, T, V)

        return logits
