import math
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel


class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class Stage1LLaMA(nn.Module):
    def __init__(
        self,
        tgt_seq_len: int,
        vocab_size: int,
        d_model: int,
        h: int,
        N: int,
        num_labels: int,
        dropout: float = 0.1,
        pad_token_id: int = 2,
        debug: bool = False,
    ):
        super().__init__()

        self.debug = debug
        self.pad_token_id = pad_token_id
        self.tgt_seq_len = tgt_seq_len

        # embeddings
        self.tgt_embed = InputEmbeddings(d_model, vocab_size)
        self.mech_embed = nn.Embedding(num_labels, d_model)
        self.pos_embed = nn.Embedding(tgt_seq_len + 1, d_model)   # <-- NEW

        # LLaMA
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

        for p in self.llama.embed_tokens.parameters():
            p.requires_grad = False

        self.proj = nn.Linear(d_model, vocab_size)

        print("🔥 Initialized Stage1LLaMA (mech_type + joints + positional embeddings)")

    def forward(self, decoder_input_ids, _, mech_labels):
        B, T = decoder_input_ids.shape
        device = decoder_input_ids.device

        # mech token (B, 1, D)
        mech_tok = self.mech_embed(mech_labels).unsqueeze(1)

        # target tokens (B, T, D)
        tgt_emb = self.tgt_embed(decoder_input_ids)

        # concat (B, T+1, D)
        full_seq = torch.cat([mech_tok, tgt_emb], dim=1)

        # --- ADD POSITIONAL EMBEDDINGS HERE ---
        pos_ids = torch.arange(T + 1, device=device).unsqueeze(0)  # (1, T+1)
        full_seq = full_seq + self.pos_embed(pos_ids)              # (B, T+1, D)

        # llama
        attn_mask = torch.ones(B, T+1, device=device, dtype=torch.long)
        out = self.llama(inputs_embeds=full_seq, attention_mask=attn_mask)
        hidden = out.last_hidden_state

        # skip mech token
        hidden = hidden[:, 1:, :]

        logits = self.proj(hidden)
        return logits


# import math
# import torch
# import torch.nn as nn
# from transformers import LlamaConfig, LlamaModel


# class InputEmbeddings(nn.Module):
#     def __init__(self, d_model, vocab_size):
#         super().__init__()
#         self.d_model = d_model
#         self.embedding = nn.Embedding(vocab_size, d_model)

#     def forward(self, x):
#         return self.embedding(x) * math.sqrt(self.d_model)


# class Stage1LLaMA(nn.Module):
#     """
#     Stage-1 model: only mech_type + joint tokens.
#     Sequence:
#         [ MECH_TOKEN , TARGET_TOKENS ]
#     No latent token.
#     """

#     def __init__(
#         self,
#         tgt_seq_len: int,
#         vocab_size: int,
#         d_model: int,
#         h: int,
#         N: int,
#         num_labels: int,
#         dropout: float = 0.1,
#         pad_token_id: int = 2,
#         debug: bool = False,
#     ):
#         super().__init__()

#         self.debug = debug
#         self.pad_token_id = pad_token_id
#         self.tgt_seq_len = tgt_seq_len

#         # token embeddings
#         self.tgt_embed = InputEmbeddings(d_model, vocab_size)
#         self.mech_embed = nn.Embedding(num_labels, d_model)

#         # LLaMA
#         llama_cfg = LlamaConfig(
#             vocab_size=vocab_size,
#             hidden_size=d_model,
#             intermediate_size=4 * d_model,
#             num_attention_heads=h,
#             num_hidden_layers=N,
#             rms_norm_eps=1e-6,
#             hidden_act="silu",
#             use_cache=False,
#         )
#         self.llama = LlamaModel(llama_cfg)

#         # Freeze default token embeddings
#         for p in self.llama.embed_tokens.parameters():
#             p.requires_grad = False

#         # output projection
#         self.proj = nn.Linear(d_model, vocab_size)

#         print("🔥 Initialized Stage1LLaMA (mech_type + joints only)")

#     def forward(self, decoder_input_ids, _, mech_labels):
#         """
#         decoder_input_ids: (B, T)
#         mech_labels: (B,)
#         """
#         B, T = decoder_input_ids.shape
#         device = decoder_input_ids.device

#         # mech token (B, 1, D)
#         mech_tok = self.mech_embed(mech_labels).unsqueeze(1)

#         # target token embeddings (B, T, D)
#         tgt_emb = self.tgt_embed(decoder_input_ids)
        
#         # final sequence: [ MECH_TOKEN , target tokens ]
#         full_seq = torch.cat([mech_tok, tgt_emb], dim=1)

#         attn_mask = torch.ones(B, full_seq.size(1), device=device, dtype=torch.long)

#         out = self.llama(inputs_embeds=full_seq, attention_mask=attn_mask)
#         hidden = out.last_hidden_state

#         # skip mech token → return only predictions for target positions
#         hidden = hidden[:, 1:, :]   # (B, T, D)

#         logits = self.proj(hidden)  # (B, T, V)

#         return logits
