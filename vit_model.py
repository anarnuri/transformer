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

        # print(f"decoder_input_ids: {decoder_input_ids.shape}")
        # print(f"decoder_mask: {decoder_mask.shape}")
        # print(f"image_data: {image_data.shape}")
        # print(f"memory: {memory.shape}")
        # print(f"tgt_key_padding_mask: {tgt_key_padding_mask.shape}")
        # print(f"dec_out: {dec_out.shape}")
        # print(f"logits: {logits.shape}")

        return logits


class SingleImageTransformerCLIP(nn.Module):
    def __init__(
        self,
        tgt_seq_len: int = 25,
        vocab_size: int = 204,
        d_model: int = 512,
        h: int = 8,
        N: int = 6,
        num_labels: int = 17,      # mech types
        dropout: float = 0.1,
        img_in_channels: int = 1,
        img_size: int = 64,
        img_patch: int = 8,
        pad_token_id: int = 2,
        debug: bool = False,       # <--- added global debug flag
    ):
        super().__init__()
        self.tgt_seq_len = tgt_seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id
        self.debug = debug          # store debug flag globally

        # --- target + mech embeddings ---
        self.tgt_embed = InputEmbeddings(d_model, vocab_size)
        self.mech_embedding = nn.Embedding(num_labels, d_model)
        self.decoder_positional_encoding = PositionalEncoding(
            d_model, seq_len=tgt_seq_len + 1, dropout=dropout
        )

        # --- image encoder ---
        self.image_encoder = ViTEncoder(
            emb_size=d_model,
            in_channels=img_in_channels,
            image_size=img_size,
            patch_size=img_patch,
            num_layers=N,
            nhead=h,
            dropout=dropout,
        )

        # --- decoder ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=h,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=N)

        # --- head ---
        self.projection = nn.Linear(d_model, vocab_size)

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def _dprint(self, *args, **kwargs):
        """Conditional print helper based on self.debug flag."""
        if self.debug:
            print(*args, **kwargs)

    def enable_debug(self, mode: bool = True):
        """Toggle debugging globally."""
        self.debug = mode
        print(f"[DEBUG MODE] set to {self.debug}")

    # ------------------------------------------------------------------
    # Mask makers
    # ------------------------------------------------------------------
    def _make_causal_attn_mask(self, L: int, device):
        """Return a standard 2D causal mask with -inf for disallowed positions."""
        mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
        float_mask = torch.zeros(L, L, device=device, dtype=torch.float32)
        float_mask.masked_fill_(~mask, float("-inf"))
        return float_mask

    # ------------------------------------------------------------------
    # encode / decode / forward
    # ------------------------------------------------------------------
    def encode(self, image_data: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, N, D)
        return self.image_encoder(image_data)

    def decode(
        self,
        memory: torch.Tensor,       # (B, N, D)
        tgt_ids: torch.Tensor,      # (B, T)
        mech_labels: torch.Tensor,  # (B,)
        tgt_key_padding_mask: torch.Tensor = None,  # (B, T)
    ) -> torch.Tensor:
        self._dprint("\n[DEBUG] ===== DECODE START =====")
        self._dprint(f"memory: {memory.shape} {memory.dtype} {memory.device}")
        self._dprint(f"tgt_ids: {tgt_ids.shape} {tgt_ids.dtype} {tgt_ids.device}")
        self._dprint(f"mech_labels: {mech_labels.shape} {mech_labels.dtype} {mech_labels.device}")
        if tgt_key_padding_mask is not None:
            self._dprint(f"tgt_key_padding_mask: {tgt_key_padding_mask.shape}, sum(pad)={tgt_key_padding_mask.sum().item()}")

        B, T = tgt_ids.shape
        device = tgt_ids.device

        # 1) mech token (B, 1, D)
        mech_emb = self.mech_embedding(mech_labels).unsqueeze(1)
        self._dprint(f"[DEBUG] mech_emb: {mech_emb.shape}")

        # 2) normal target tokens (B, T, D)
        tgt = self.tgt_embed(tgt_ids)
        self._dprint(f"[DEBUG] tgt (before concat): {tgt.shape}")

        # 3) prepend mech -> (B, T+1, D)
        tgt = torch.cat([mech_emb, tgt], dim=1)
        self._dprint(f"[DEBUG] tgt (after concat): {tgt.shape}")

        # 4) add positions
        tgt = self.decoder_positional_encoding(tgt)
        self._dprint(f"[DEBUG] tgt (after positional): {tgt.shape}")

        # 5) causal mask (L, L)
        L = T + 1
        causal_mask = self._make_causal_attn_mask(L, device)
        self._dprint(f"[DEBUG] causal_mask: {causal_mask.shape}, "
                     f"min={causal_mask.min().item():.3f}, max={causal_mask.max().item():.3f}")

        # 6) extend padding mask
        if tgt_key_padding_mask is not None:
            mech_pad = torch.zeros((B, 1), dtype=torch.bool, device=device)
            tgt_key_padding_mask = torch.cat([mech_pad, tgt_key_padding_mask], dim=1)
            self._dprint(f"[DEBUG] tgt_key_padding_mask (after mech): "
                         f"{tgt_key_padding_mask.shape}, sum(pad)={tgt_key_padding_mask.sum().item()}")
        else:
            self._dprint("[DEBUG] tgt_key_padding_mask: None")

        # Optional prints
        if self.debug and L <= 20:
            self._dprint("[DEBUG] causal_mask (upper-left corner):")
            self._dprint(causal_mask.cpu())

        if self.debug and tgt_key_padding_mask is not None and T <= 20:
            self._dprint("[DEBUG] tgt_key_padding_mask (first few rows):")
            self._dprint(tgt_key_padding_mask.cpu())

        self._dprint("[DEBUG] calling TransformerDecoder...")
        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        self._dprint(f"[DEBUG] decoder output: {out.shape}")
        self._dprint("[DEBUG] ===== DECODE END =====\n")

        return out

    def forward(
        self,
        decoder_input_ids: torch.Tensor,  # (B, T)
        _decoder_mask_unused: torch.Tensor,  # ignored for compatibility
        image_data: torch.Tensor,         # (B, 1, 64, 64)
        mech_labels: torch.Tensor,        # (B,)
    ) -> torch.Tensor:
        self._dprint("\n[DEBUG] ===== FORWARD START =====")
        self._dprint(f"decoder_input_ids: {decoder_input_ids.shape} {decoder_input_ids.dtype} {decoder_input_ids.device}")
        self._dprint(f"image_data:        {image_data.shape} {image_data.dtype} {image_data.device}")
        self._dprint(f"mech_labels:       {mech_labels.shape} {mech_labels.dtype} {mech_labels.device}")

        # --- Encode ---
        memory = self.encode(image_data)
        self._dprint(f"[DEBUG] memory: {memory.shape} {memory.dtype}")

        # --- Key padding mask ---
        tgt_key_padding_mask = (decoder_input_ids == self.pad_token_id)
        self._dprint(f"[DEBUG] tgt_key_padding_mask: {tgt_key_padding_mask.shape}, "
                     f"sum={tgt_key_padding_mask.sum().item()}")

        # --- Decode ---
        dec_out = self.decode(
            memory=memory,
            tgt_ids=decoder_input_ids,
            mech_labels=mech_labels,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        self._dprint(f"[DEBUG] dec_out (raw): {dec_out.shape} {dec_out.dtype}")

        # --- Drop mech token ---
        dec_out = dec_out[:, 1:, :]
        self._dprint(f"[DEBUG] dec_out (after drop): {dec_out.shape}")

        # --- Project to logits ---
        logits = self.projection(dec_out)
        self._dprint(f"[DEBUG] logits: {logits.shape} {logits.dtype}, "
                     f"min={logits.min().item():.4f}, max={logits.max().item():.4f}")
        self._dprint("[DEBUG] ===== FORWARD END =====\n")

        return logits

class SingleLatentTransformer(nn.Module):
    """
    Transformer decoder that conditions on VAE latent sequences (length 50).
    The latent sequence is processed by a Transformer encoder to produce memory of shape (B, 50, D).
    Includes mech-type embeddings and detailed debug printouts.
    """

    def __init__(
        self,
        tgt_seq_len: int = 25,
        vocab_size: int = 204,
        latent_dim: int = 50,
        d_model: int = 512,
        h: int = 8,
        N: int = 6,
        num_labels: int = 17,       # mechanism types
        dropout: float = 0.1,
        pad_token_id: int = 2,
        noise_std: float = 0.05,    # amount of Gaussian noise added to latents
        debug: bool = True,         # enable detailed shape printouts
    ):
        super().__init__()

        # ----------------------------
        # Basic parameters
        # ----------------------------
        self.tgt_seq_len = tgt_seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id
        self.noise_std = noise_std
        self.debug = debug
        self.latent_dim = latent_dim

        # ----------------------------
        # Embeddings
        # ----------------------------
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.mech_embedding = nn.Embedding(num_labels, d_model)

        # positional encoding for decoder tokens (+1 for mech token)
        self.decoder_positional_encoding = self._build_positional_encoding(
            d_model, seq_len=tgt_seq_len + 1
        )

        # ----------------------------
        # Latent Transformer Encoder
        # ----------------------------
        # Treat latents (B, 50) as sequence of 50 scalar tokens
        self.latent_proj = nn.Linear(1, d_model)  # project each scalar to D
        self.latent_pos_encoding = self._build_positional_encoding(d_model, seq_len=latent_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=h,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.latent_encoder = nn.TransformerEncoder(encoder_layer, num_layers=N)
        self.latent_norm = nn.LayerNorm(d_model)

        # ----------------------------
        # Transformer Decoder
        # ----------------------------
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=h,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=N)

        # ----------------------------
        # Output projection
        # ----------------------------
        self.projection = nn.Linear(d_model, vocab_size)

        self.apply(self._init_weights)
        print("✅ Initialized SingleLatentTransformer (latent-sequence-based)")

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------
    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def _dprint(self, *args, **kwargs):
        """Conditional print helper based on debug flag."""
        if self.debug:
            print(*args, **kwargs)

    def _build_positional_encoding(self, d_model, seq_len):
        """Create sinusoidal positional encodings."""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)  # (1, seq_len, d_model)

    def _make_causal_attn_mask(self, L: int, device):
        mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
        float_mask = torch.zeros(L, L, device=device, dtype=torch.float32)
        float_mask.masked_fill_(~mask, float("-inf"))
        return float_mask

    # ------------------------------------------------------------
    # Encode latent sequence
    # ------------------------------------------------------------
    def encode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        latents: (B, latent_dim) or (B, latent_dim, 1)
        Returns: memory (B, latent_dim, D)
        """
        if latents.dim() == 2:
            latents = latents.unsqueeze(-1)  # (B, L, 1)
        B, L, _ = latents.shape

        # Add Gaussian noise during training
        if self.training and self.noise_std > 0:
            latents = latents + self.noise_std * torch.randn_like(latents)

        # Project each latent dimension into d_model
        x = self.latent_proj(latents)  # (B, L, D)
        x = x + self.latent_pos_encoding[:, :L, :].to(latents.device)

        self._dprint(f"[ENCODE] input latents: {latents.shape} -> projected: {x.shape}")

        # Apply transformer encoder
        memory = self.latent_encoder(x)
        memory = self.latent_norm(memory)
        self._dprint(f"[ENCODE] memory (after transformer): {memory.shape}")
        return memory  # (B, L, D)

    # ------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------
    def forward(
        self,
        decoder_input_ids: torch.Tensor,  # (B, T)
        _mask_unused: torch.Tensor,       # ignored
        latents: torch.Tensor,            # (B, latent_dim) or (B, latent_dim, 1)
        mech_labels: torch.Tensor,        # (B,)
    ) -> torch.Tensor:
        """
        Returns logits: (B, T, vocab_size)
        """
        self._dprint("\n========== FORWARD START ==========")
        self._dprint(f"decoder_input_ids: {decoder_input_ids.shape}")
        self._dprint(f"latents: {latents.shape}")
        self._dprint(f"mech_labels: {mech_labels.shape}")

        B, T = decoder_input_ids.shape
        device = latents.device

        # --- Encode latent sequence ---
        memory = self.encode(latents)  # (B, L, D)
        self._dprint(f"[FORWARD] memory: {memory.shape}")

        # --- Mech embedding ---
        mech_emb = self.mech_embedding(mech_labels).unsqueeze(1)
        self._dprint(f"[FORWARD] mech_emb: {mech_emb.shape}")

        # --- Decoder tokens ---
        tgt = self.tgt_embed(decoder_input_ids)
        self._dprint(f"[FORWARD] tgt_embed: {tgt.shape}")

        # prepend mech token
        tgt = torch.cat([mech_emb, tgt], dim=1)
        tgt = tgt + self.decoder_positional_encoding[:, :tgt.size(1), :].to(device)
        self._dprint(f"[FORWARD] tgt (after concat + pos): {tgt.shape}")

        # --- Causal mask ---
        causal_mask = self._make_causal_attn_mask(T + 1, device)
        tgt_key_padding_mask = (decoder_input_ids == self.pad_token_id)
        mech_pad = torch.zeros((B, 1), dtype=torch.bool, device=device)
        tgt_key_padding_mask = torch.cat([mech_pad, tgt_key_padding_mask], dim=1)
        self._dprint(
            f"[FORWARD] tgt_key_padding_mask: {tgt_key_padding_mask.shape}, pad_sum={tgt_key_padding_mask.sum().item()}"
        )

        # --- Decode ---
        dec_out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        self._dprint(f"[FORWARD] decoder output: {dec_out.shape}")

        # --- Drop mech token ---
        dec_out = dec_out[:, 1:, :]
        self._dprint(f"[FORWARD] dec_out (after drop): {dec_out.shape}")

        # --- Project to logits ---
        logits = self.projection(dec_out)
        self._dprint(
            f"[FORWARD] logits: {logits.shape}, min={logits.min().item():.3f}, max={logits.max().item():.3f}"
        )
        self._dprint("========== FORWARD END ==========\n")

        return logits
