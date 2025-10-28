import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Utils / inits
# ------------------------------------------------------------
def _init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Embedding)):
        if getattr(m, "weight", None) is not None:
            nn.init.xavier_uniform_(m.weight)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.zeros_(m.bias)


# ------------------------------------------------------------
# Norm / MLP blocks (kept for parity; FFN not used directly here)
# ------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = 4 * d_model
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return self.dropout(x)


# ------------------------------------------------------------
# Token & Positional embeddings
# ------------------------------------------------------------
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x) -> torch.Tensor:
        # (B, T) -> (B, T, D)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


# ------------------------------------------------------------
# ViT-style Image Encoder (sequence output)
# For 64x64 grayscale images, patch_size=8 -> N=64 tokens
# ------------------------------------------------------------
class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder that outputs patch embeddings (B, N, D).
    Designed for 64x64 grayscale images by default.
    """
    def __init__(
        self,
        emb_size: int = 512,
        in_channels: int = 1,
        image_size: int = 64,
        patch_size: int = 8,
        num_layers: int = 6,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.num_patches = (image_size // patch_size) ** 2
        self.emb_size = emb_size

        # Patch embedding via strided conv
        self.patch_embed = nn.Conv2d(
            in_channels,
            emb_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=False,
        )

        # Learnable positional embeddings for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, emb_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder over patch tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=nhead,
            dim_feedforward=4 * emb_size,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(emb_size)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        returns: (B, N, D)
        """
        patches = self.patch_embed(x)                 # (B, D, H/ps, W/ps)
        tokens = patches.flatten(2).transpose(1, 2)   # (B, N, D)
        tokens = tokens + self.pos_embed
        encoded = self.encoder(tokens)                # (B, N, D)
        return self.norm(encoded)


# ------------------------------------------------------------
# SingleImageTransformer with ViT encoder and safe attention masking
# ------------------------------------------------------------
class SingleImageTransformer(nn.Module):
    """
    - Image encoder: ViT-style, outputs (B, N, D)
    - Decoder: PyTorch TransformerDecoder
    - Target tokens: embedded + sinusoidal positions
    - Safe attention mask handling:
        * Accepts tgt_mask as (B, T, T) or (T, T) with True==allowed
        * Converts to float mask with -inf where masked
        * Uses tgt_key_padding_mask derived from PAD tokens in forward()
    """
    def __init__(
        self,
        tgt_seq_len: int = 25,
        vocab_size: int = 204,
        d_model: int = 512,
        h: int = 8,
        N: int = 6,
        num_labels: int = 105,  # kept for API parity; unused internally
        dropout: float = 0.1,
        img_in_channels: int = 1,
        img_size: int = 64,
        img_patch: int = 8,
        pad_token_id: int = 2,
    ):
        super().__init__()
        self.tgt_seq_len = tgt_seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_labels = num_labels
        self.n_heads = h
        self.pad_token_id = pad_token_id

        # Target token embeddings + positions
        self.tgt_embed = InputEmbeddings(d_model, vocab_size)
        self.decoder_positional_encoding = PositionalEncoding(d_model, seq_len=tgt_seq_len, dropout=dropout)

        # Image encoder (ViT-style -> sequence output)
        self.image_encoder = ViTEncoder(
            emb_size=d_model,
            in_channels=img_in_channels,
            image_size=img_size,
            patch_size=img_patch,
            num_layers=N,
            nhead=h,
            dropout=dropout,
        )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=h,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=N)

        # LM head
        self.projection = nn.Linear(d_model, vocab_size)

        self.apply(_init_weights)

    # ------------------------
    # Mask utilities
    # ------------------------
    @staticmethod
    def _to_attn_mask(tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Convert incoming boolean mask to float attention mask with -inf
        for disallowed positions. Accepts:
          - (T, T)  with True==allowed
          - (B, T, T) with True==allowed -> uses the first batch (assumes causal identical across batch)
        Returns:
          - (T, T) float with 0 for allowed, -inf for masked
        """
        if tgt_mask.dim() == 3:
            tgt_mask = tgt_mask[0]  # (T, T)
        elif tgt_mask.dim() != 2:
            raise ValueError(f"Unexpected tgt_mask shape {tgt_mask.shape}. Expected (T,T) or (B,T,T).")

        # Incoming True==allowed, False==masked
        disallowed = ~tgt_mask.bool()
        attn = torch.zeros_like(tgt_mask, dtype=torch.float, device=tgt_mask.device)
        attn[disallowed] = float('-inf')
        return attn  # (T, T) float

    # ------------------------
    # Encode / Decode
    # ------------------------
    def encode(self, image_data: torch.Tensor) -> torch.Tensor:
        """
        image_data: (B, C, H, W)
        returns memory: (B, N, D)
        """
        return self.image_encoder(image_data)

    def decode(
        self,
        memory: torch.Tensor,        # (B, N, D)
        tgt_ids: torch.Tensor,       # (B, T)
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Returns: decoder hidden states (B, T, D)
        """
        tgt = self.tgt_embed(tgt_ids)                 # (B, T, D)
        tgt = self.decoder_positional_encoding(tgt)   # (B, T, D)

        attn_mask = None
        if tgt_mask is not None:
            attn_mask = self._to_attn_mask(tgt_mask)  # (T, T) float

        output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=attn_mask,                            # (T, T) float with -inf on masked
            memory_mask=memory_mask,                       # (T, S) if used (optional)
            tgt_key_padding_mask=tgt_key_padding_mask,     # (B, T) bool where True==PAD
            memory_key_padding_mask=memory_key_padding_mask  # (B, N) if you ever add it
        )
        return output  # (B, T, D)

    # ------------------------
    # Forward
    # ------------------------
    def forward(
        self,
        decoder_input_ids: torch.Tensor,   # (B, T) token ids
        decoder_mask: torch.Tensor,        # (B, T, T) or (T, T) with True==allowed
        image_data: torch.Tensor,          # (B, C, 64, 64)
        labels: torch.Tensor = None,       # kept for API parity; unused
    ) -> torch.Tensor:
        """
        Returns logits: (B, T, vocab_size)
        """
        # Encode image to (B, N, D)
        memory = self.encode(image_data)

        # Build key padding mask for decoder (True == PAD)
        tgt_key_padding_mask = (decoder_input_ids == self.pad_token_id)

        # Decode
        dec_out = self.decode(
            memory=memory,
            tgt_ids=decoder_input_ids,
            tgt_mask=decoder_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # LM head
        logits = self.projection(dec_out)  # (B, T, V)
        return logits
