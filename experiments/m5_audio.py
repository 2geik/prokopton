"""
M5 — Encoder-free Doğrudan Ses Kodlayıcı (Mel-LLM tarzı)

Mekanizma:
  - Ham ses → Mel-spektrogram yamaları (hafif ön işleme)
  - Yamalar → lineer projeksiyon → LLM token uzayına
  - STT yok, encoder yok → ton/duygu/mikro-ifadeler korunur
  - Hizalama LLM'in kendi parametrelerinde

Referans: Mel-LLM (arXiv 2606.10231)

Kullanım:
  .venv/bin/python experiments/m5_audio.py [--audio test.wav]
"""
import torch
import torch.nn as nn
import math


class MelPatchExtractor(nn.Module):
    """Ham ses → Mel-spektrogram yamaları (hafif ön işleme, encoder yok)."""
    
    def __init__(self, 
                 sample_rate=16000,
                 n_mels=80,
                 n_fft=400,      # 25ms @ 16kHz
                 hop_length=160,  # 10ms @ 16kHz
                 patch_frames=4,  # 40ms patch (4 × 10ms)
                 patch_mels=80):
        """
        Args:
            sample_rate: Örnekleme hızı
            n_mels: Mel filtresi sayısı
            n_fft: FFT pencere boyutu
            hop_length: Atlama boyutu
            patch_frames: Her yamadaki zaman çerçevesi sayısı (40ms = 4 × 10ms)
            patch_mels: Her yamadaki mel bandı sayısı
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.patch_frames = patch_frames
        self.patch_mels = patch_mels
        
        super().__init__()
        
        # Mel filtre bankası (statik)
        self.register_mel_filters()
        
    def register_mel_filters(self):
        """Mel filtre bankasını oluştur."""
        # Basitleştirilmiş mel scale: lineer frekans → mel
        mel_freqs = torch.linspace(0, self._hz_to_mel(self.sample_rate // 2), self.n_mels + 2)
        hz_freqs = self._mel_to_hz(mel_freqs)
        
        # FFT bin indeksleri
        fft_bins = torch.floor((self.n_fft + 1) * hz_freqs / self.sample_rate).long()
        
        # Filtre bankası
        self.register_buffer('fft_bins', fft_bins)
        self.n_fft_bins = self.n_fft // 2 + 1
        
    @staticmethod
    def _hz_to_mel(hz):
        if not isinstance(hz, torch.Tensor):
            hz = torch.tensor(hz)
        return 2595 * torch.log10(1 + hz.float() / 700)
    
    @staticmethod  
    def _mel_to_hz(mel):
        if not isinstance(mel, torch.Tensor):
            mel = torch.tensor(mel)
        return 700 * (10 ** (mel.float() / 2595) - 1)
    
    def extract_patches(self, waveform):
        """
        Ham dalga formu → Mel-spektrogram yamaları.
        
        Args:
            waveform: [T] veya [1, T] float tensor (-1 to 1)
        Returns:
            patches: [N, patch_frames * patch_mels] yassı yamalar
            n_frames: Toplam zaman çerçevesi sayısı
        """
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        
        device = waveform.device
        
        # STFT (basit: Hann pencere + FFT)
        window = torch.hann_window(self.n_fft, device=device)
        
        # Frame'lere böl
        n_frames = 1 + (len(waveform) - self.n_fft) // self.hop_length
        if n_frames <= 0:
            return torch.zeros(0, self.patch_frames * self.patch_mels, device=device), 0
        
        frames = torch.zeros(n_frames, self.n_fft, device=device)
        for i in range(n_frames):
            start = i * self.hop_length
            frames[i] = waveform[start:start + self.n_fft] * window
        
        # FFT → güç spektrumu
        spec = torch.abs(torch.fft.rfft(frames, dim=-1))[:, :self.n_fft_bins]
        spec = spec ** 2
        
        # Mel filtre bankası uygula
        mel_spec = torch.zeros(n_frames, self.n_mels, device=device)
        fft_bins = self.fft_bins.to(device)
        
        for m in range(self.n_mels):
            start_bin = fft_bins[m].item()
            end_bin = fft_bins[m + 2].item()
            if end_bin > start_bin and end_bin <= self.n_fft_bins:
                mel_spec[:, m] = spec[:, start_bin:end_bin].mean(dim=-1)
        
        # Log-scale
        mel_spec = torch.log(mel_spec + 1e-10)
        
        # Normalleştir
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        # Yamalara böl (overlapping olmadan)
        n_patches = n_frames // self.patch_frames
        patches = []
        for i in range(n_patches):
            start = i * self.patch_frames
            end = start + self.patch_frames
            patch = mel_spec[start:end].flatten()
            patches.append(patch)
        
        if patches:
            patches = torch.stack(patches)
        else:
            patches = torch.zeros(0, self.patch_frames * self.patch_mels, device=device)
        
        return patches, n_frames


class AudioTokenizer(nn.Module):
    """
    Encoder-free ses tokenleştirici.
    
    Ham dalga → Mel patch → Lineer projeksiyon → LLM token uzayı
    """
    
    def __init__(self, 
                 sample_rate=16000,
                 n_mels=80,
                 patch_frames=4,
                 embed_dim=1024,
                 output_dim=2560):
        """
        Args:
            sample_rate: Örnekleme hızı (Hz)
            n_mels: Mel filtresi sayısı
            patch_frames: Yama başına zaman çerçevesi (40ms @ 10ms hop)
            embed_dim: Ara embedding boyutu
            output_dim: LLM embedding boyutu (Gemma 4: 2560)
        """
        super().__init__()
        self.extractor = MelPatchExtractor(
            sample_rate=sample_rate,
            n_mels=n_mels,
            patch_frames=patch_frames,
            patch_mels=n_mels,
        )
        
        patch_flat_dim = patch_frames * n_mels
        self.input_proj = nn.Linear(patch_flat_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, output_dim)
        
        # Pozisyonel encoding (öğrenilebilir sinüzoidal)
        max_patches = 4096
        position = torch.arange(max_patches).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                       (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_patches, embed_dim)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer('positional_encoding', pe)
        
    def forward(self, waveform):
        """
        Args:
            waveform: [T] veya [1, T] veya [B, T] float tensor
        Returns:
            tokens: [B, N, output_dim] LLM embedding uzayında token'lar
            info: dict
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        B = waveform.shape[0]
        all_tokens = []
        all_infos = []
        
        for b in range(B):
            patches, n_frames = self.extractor.extract_patches(waveform[b])
            
            if patches.shape[0] == 0:
                # Boş ses → tek token
                tokens = torch.zeros(1, self.output_proj.out_features, device=waveform.device)
                info = {'num_tokens': 1, 'duration_ms': 0}
            else:
                # Lineer projeksiyon + pozisyonel encoding
                x = self.input_proj(patches)
                x = x + self.positional_encoding[:x.shape[0]]
                tokens = self.output_proj(x)
                
                duration_ms = (n_frames * self.extractor.hop_length / self.extractor.sample_rate) * 1000
                info = {
                    'num_tokens': tokens.shape[0],
                    'duration_ms': duration_ms,
                    'tokens_per_second': tokens.shape[0] / (duration_ms / 1000) if duration_ms > 0 else 0,
                }
            
            all_tokens.append(tokens)
            all_infos.append(info)
        
        return all_tokens, all_infos


# ============================================================
# Deney
# ============================================================

def test_audio_tokenizer():
    """Ses tokenizer testi."""
    print("Audio Tokenizer Test (Mel-LLM style)")
    print("=" * 50)
    
    # Dummy ses: 2 saniye, 16kHz
    duration = 2.0
    sample_rate = 16000
    t = torch.linspace(0, duration, int(sample_rate * duration))
    
    # Basit sinüs dalgası
    waveform = torch.sin(2 * math.pi * 440 * t) * 0.5  # 440 Hz A notası
    
    print(f"  Giriş ses: {duration}s @ {sample_rate}Hz → {len(waveform)} örnek")
    
    tokenizer = AudioTokenizer(
        sample_rate=sample_rate,
        n_mels=80,
        patch_frames=4,    # 40ms patch
        embed_dim=1024,
        output_dim=2560,
    )
    
    tokens, infos = tokenizer(waveform)
    info = infos[0]
    
    print(f"  Çıkış: {tokens[0].shape} token")
    print(f"  Token bütçesi: {info['num_tokens']} token")
    print(f"  Süre: {info['duration_ms']:.0f}ms")
    print(f"  Token/saniye: {info['tokens_per_second']:.1f}")
    print(f"  Ortalama: {info['duration_ms']/info['num_tokens']:.1f}ms/token")
    
    # Farklı süreler
    for dur in [0.5, 1.0, 5.0, 10.0]:
        t2 = torch.linspace(0, dur, int(sample_rate * dur))
        wf2 = torch.sin(2 * math.pi * 220 * t2)
        tokens2, infos2 = tokenizer(wf2)
        print(f"  {dur:.1f}s → {infos2[0]['num_tokens']} token ({infos2[0]['duration_ms']:.0f}ms)")
    
    print("\n  Not: STT yok → ton, duygu, mikro-ifadeler korunur")
    print("✅ Encoder-free audio tokenizer hazır")


if __name__ == "__main__":
    test_audio_tokenizer()
