"""
M4 — Encoder-free Görsel Kodlayıcı (Tuna-2 tarzı)

Mekanizma:
  - Görüntüyü 48×48 patch'lere böl
  - Her patch → lineer projeksiyon ile token uzayına
  - 2D-RoPE pozisyonel kodlama (faktörize X/Y)
  - Dinamik token bütçesi (~70-1120)
  - Encoder yok, VAE yok — doğrudan LLM token uzayına

Referans: Tuna-2 (arXiv 2604.24763)

Kullanım:
  .venv/bin/python experiments/m4_visual.py
"""
import torch
import torch.nn as nn
import argparse
from pathlib import Path


class PatchEmbed2D(nn.Module):
    """Görüntüyü patch'lere böl ve lineer projeksiyonla embed et."""
    
    def __init__(self, patch_size=48, embed_dim=1024, in_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] görüntü tensörü
        Returns:
            patches: [B, N, D] patch embedding'leri
            grid: (h, w) patch grid boyutu
        """
        B, C, H, W = x.shape
        p = self.patch_size
        
        # Padding (tam bölünebilir yap)
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))
        
        H_pad, W_pad = x.shape[2], x.shape[3]
        
        # Patch'lere böl: [B, C, H, W] → [B, N, C*p*p]
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        h_grid, w_grid = patches.shape[1], patches.shape[2]
        patches = patches.view(B, h_grid * w_grid, -1)
        
        # Lineer projeksiyon
        patches = self.proj(patches)
        
        return patches, (h_grid, w_grid)


class RoPE2D(nn.Module):
    """2D Rotary Position Embedding — faktörize X/Y koordinat."""
    
    def __init__(self, dim, max_grid=64):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        
        # X ve Y için ayrı frekanslar
        freqs_x = 1.0 / (10000 ** (torch.arange(0, half_dim // 2, 2).float() / (half_dim // 2)))
        freqs_y = 1.0 / (10000 ** (torch.arange(0, half_dim // 2, 2).float() / (half_dim // 2)))
        
        self.register_buffer('freqs_x', freqs_x)
        self.register_buffer('freqs_y', freqs_y)
        
    def forward(self, x, grid_h, grid_w):
        """
        Args:
            x: [B, N, D]
            grid_h, grid_w: patch grid boyutu
        Returns:
            x: [B, N, D] RoPE uygulanmış
        """
        B, N, D = x.shape
        device = x.device
        
        # Koordinat matrisleri
        coords_h = torch.arange(grid_h, device=device).float()
        coords_w = torch.arange(grid_w, device=device).float()
        
        # Faktörize: her h,w için sin/cos hesapla
        grid_y, grid_x = torch.meshgrid(coords_h, coords_w, indexing='ij')
        positions = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)  # [N, 2]
        
        # X RoPE
        theta_x = positions[:, 1:2] * self.freqs_x  # [N, half_dim//4]
        theta_x = theta_x.repeat_interleave(2, dim=-1)
        cos_x, sin_x = theta_x.cos(), theta_x.sin()
        
        # Y RoPE
        theta_y = positions[:, 0:1] * self.freqs_y
        theta_y = theta_y.repeat_interleave(2, dim=-1)
        cos_y, sin_y = theta_y.cos(), theta_y.sin()
        
        # Birleştir: [cos_x, cos_y, sin_x, sin_y]
        quarter = D // 4
        cos = torch.cat([cos_x[:, :quarter], cos_y[:, :quarter]], dim=-1)  # [N, D/2]
        sin = torch.cat([sin_x[:, :quarter], sin_y[:, :quarter]], dim=-1)
        
        # Pad (eğer boyut eşleşmezse)
        if cos.shape[-1] < D // 2:
            cos = nn.functional.pad(cos, (0, D // 2 - cos.shape[-1]))
            sin = nn.functional.pad(sin, (0, D // 2 - sin.shape[-1]))
        
        # RoPE uygula
        x1, x2 = x[..., :D//2], x[..., D//2:]
        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return x_rotated


class VisualTokenizer(nn.Module):
    """
    Encoder-free görsel tokenleştirici.
    
    Görüntü → Patch Embed + 2D-RoPE → Token uzayı (Gemma 4 embedding boyutuna)
    """
    
    def __init__(self, patch_size=48, embed_dim=1024, output_dim=2560, max_grid=64):
        """
        Args:
            patch_size: Patch boyutu
            embed_dim: Ara embedding boyutu
            output_dim: LLM embedding boyutu (Gemma 4 E4B: 2560)
            max_grid: Maksimum grid boyutu
        """
        super().__init__()
        self.patch_embed = PatchEmbed2D(patch_size, embed_dim)
        self.rope2d = RoPE2D(embed_dim, max_grid)
        self.output_proj = nn.Linear(embed_dim, output_dim)
        self.patch_size = patch_size
        
    def forward(self, images):
        """
        Args:
            images: [B, C, H, W] veya PIL Image listesi
        Returns:
            tokens: [B, N, output_dim] LLM embedding uzayında token'lar
            info: dict — token bütçesi ve grid bilgisi
        """
        if isinstance(images, list):
            # PIL → tensor (basit ön işleme)
            images = self._preprocess(images)
        
        B = images.shape[0]
        
        # Patch embed
        patches, (grid_h, grid_w) = self.patch_embed(images)
        
        # 2D-RoPE
        patches = self.rope2d(patches, grid_h, grid_w)
        
        # LLM embedding boyutuna projekte et
        tokens = self.output_proj(patches)
        
        info = {
            'num_tokens': grid_h * grid_w,
            'grid': (grid_h, grid_w),
            'patch_size': self.patch_size,
        }
        
        return tokens, info
    
    def _preprocess(self, pil_images):
        """PIL görüntülerini tensöre çevir."""
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        tensors = [transform(img) for img in pil_images]
        return torch.stack(tensors)


# ============================================================
# Deney
# ============================================================

def test_visual_tokenizer():
    """Visual tokenizer testi."""
    print("Visual Tokenizer Test (Tuna-2 style)")
    print("=" * 50)
    
    # Dummy görüntü
    img = torch.randn(1, 3, 224, 224)
    
    # Tokenizer
    tokenizer = VisualTokenizer(
        patch_size=48,
        embed_dim=1024,
        output_dim=2560  # Gemma 4 E4B embedding
    )
    
    tokens, info = tokenizer(img)
    
    print(f"  Giriş: {img.shape} → Patch: {info['grid']} → Token: {tokens.shape}")
    print(f"  Token bütçesi: {info['num_tokens']} token")
    print(f"  Token/görüntü: {tokens.shape[1]} token")
    
    # Dinamik bütçe: farklı çözünürlükler
    for size in [112, 224, 448, 672]:
        img2 = torch.randn(1, 3, size, size)
        tokens2, info2 = tokenizer(img2)
        print(f"  {size}×{size} → {info2['num_tokens']} token (grid: {info2['grid']})")
    
    print("\n✅ Encoder-free visual tokenizer hazır")


if __name__ == "__main__":
    test_visual_tokenizer()
