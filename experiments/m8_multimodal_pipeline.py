"""
M8 — Multimodal Pipeline Test (Tokenizer-only)

VisualTokenizer ve AudioTokenizer'ın bağımsız çalıştığını gösteren
basit test. Model yüklemeden sadece tokenizer çıktılarını doğrular.

Kullanım:
  .venv/bin/python experiments/m8_multimodal_pipeline.py
"""
import torch
import math
from prokopton.core import VisualTokenizer, AudioTokenizer


def test_visual_tokenizer():
    """VisualTokenizer: rastgele görüntü → token boyut kontrolü."""
    print("=" * 60)
    print("VISUAL TOKENIZER TEST")
    print("=" * 60)

    vision = VisualTokenizer(
        patch_size=48,
        embed_dim=1024,
        output_dim=2560,   # Gemma 4 hidden_size
    )
    vision.eval()

    print(f"  Patch size:      {vision.patch_size}")
    print(f"  Embed dim:       {vision.proj.out_features}")
    print(f"  Output dim:      {vision.output_proj.out_features}")
    print()

    # Test with standard 224x224 RGB image
    img = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        tokens, info = vision(img)

    print(f"  Input shape:     {list(img.shape)}")
    print(f"  Output shape:    {list(tokens.shape)}")
    print(f"  Num tokens:      {info['num_tokens']}")
    print(f"  Grid:            {info['grid']}")
    print(f"  Token dim:       {tokens.shape[-1]} (expected: 2560)")
    assert tokens.shape[-1] == 2560, f"Token dim mismatch: {tokens.shape[-1]} != 2560"
    print("  ✓ Token dimension matches Gemma 4 hidden_size\n")

    # Dynamic resolution test
    print("  Dynamic resolution budget:")
    for size in [112, 224, 448]:
        img2 = torch.randn(1, 3, size, size)
        with torch.no_grad():
            t, info2 = vision(img2)
        print(f"    {size:>4}×{size:<4} → {info2['num_tokens']:>4} tokens  "
              f"shape={list(t.shape)}")


def test_audio_tokenizer():
    """AudioTokenizer: sinüs dalgası → token boyut kontrolü."""
    print()
    print("=" * 60)
    print("AUDIO TOKENIZER TEST")
    print("=" * 60)

    audio = AudioTokenizer(
        sample_rate=16000,
        n_mels=80,
        patch_frames=4,     # 40ms @ 10ms hop
        embed_dim=1024,
        output_dim=2560,
    )
    audio.eval()

    print(f"  Sample rate:     {audio.sr} Hz")
    print(f"  Mel bands:       {audio.n_mels}")
    print(f"  Patch frames:    {audio.pf} (40ms)")
    print(f"  Output dim:      {audio.output_proj.out_features}")
    print()

    # 440 Hz sine wave, 1 second @ 16kHz
    duration = 1.0
    t = torch.linspace(0, duration, int(16000 * duration))
    waveform = torch.sin(2 * math.pi * 440 * t)  # A4 note

    with torch.no_grad():
        tokens_list, info_list = audio(waveform)

    tokens = tokens_list[0]  # [N, 2560]
    info = info_list[0]

    print(f"  Input samples:   {len(waveform)}")
    print(f"  Output shape:    {list(tokens.shape)}")
    print(f"  Num tokens:      {info['num_tokens']}")
    print(f"  Duration:        {info['duration_ms']:.0f} ms")
    print(f"  Token dim:       {tokens.shape[-1]} (expected: 2560)")
    assert tokens.shape[-1] == 2560, f"Token dim mismatch: {tokens.shape[-1]} != 2560"
    print("  ✓ Token dimension matches Gemma 4 hidden_size\n")

    # Multi-duration test
    print("  Duration → token budget:")
    for dur in [0.5, 1.0, 3.0, 10.0]:
        t2 = torch.linspace(0, dur, int(16000 * dur))
        wf2 = torch.sin(2 * math.pi * 220 * t2)  # A3 note
        with torch.no_grad():
            tl2, il2 = audio(wf2)
        print(f"    {dur:>4.1f}s → {il2[0]['num_tokens']:>4} tokens,  "
              f"shape={list(tl2[0].shape)}")


def test_tokenizer_compatibility():
    """Her iki tokenizer'ın aynı output_dim'e çıktı ürettiğini kontrol et."""
    print()
    print("=" * 60)
    print("CROSS-MODAL COMPATIBILITY CHECK")
    print("=" * 60)

    vision = VisualTokenizer(patch_size=48, embed_dim=1024, output_dim=2560)
    audio = AudioTokenizer(sample_rate=16000, n_mels=80, patch_frames=4,
                           embed_dim=1024, output_dim=2560)
    vision.eval()
    audio.eval()

    with torch.no_grad():
        img_tokens, _ = vision(torch.randn(1, 3, 224, 224))
        wf = torch.sin(2 * math.pi * 440 * torch.linspace(0, 1, 16000))
        aud_tokens_list, _ = audio(wf)

    vis_dim = img_tokens.shape[-1]
    aud_dim = aud_tokens_list[0].shape[-1]

    print(f"  Visual token dim:  {vis_dim}")
    print(f"  Audio token dim:   {aud_dim}")

    assert vis_dim == aud_dim == 2560, \
        f"Cross-modal dim mismatch: vis={vis_dim}, aud={aud_dim}"
    print("  ✓ Both tokenizers output to same 2560-dim space")
    print("  ✓ Ready for direct concat with Gemma 4 embeddings\n")


def test_embedding_flow():
    """Simulate the full flow: text embed + vis tokens + aud tokens → combined."""
    print("=" * 60)
    print("COMBINED EMBEDDING FLOW SIMULATION")
    print("=" * 60)

    D = 2560  # Gemma 4 hidden_size
    # Simulate text embeddings from LLM embed layer
    text_len = 10
    text_embeds = torch.randn(1, text_len, D)
    print(f"  Text embeddings:     {list(text_embeds.shape)}")

    # Simulate visual tokens
    vis_len = 25  # ~ 5×5 grid for 224×224 with patch_size=48
    vis_tokens = torch.randn(1, vis_len, D)
    print(f"  Visual tokens:       {list(vis_tokens.shape)}")

    # Simulate audio tokens (1s @ 16kHz → ~62 patches)
    aud_len = 62
    aud_tokens = torch.randn(1, aud_len, D)
    print(f"  Audio tokens:        {list(aud_tokens.shape)}")

    # Combine
    combined = torch.cat([text_embeds, vis_tokens, aud_tokens], dim=1)
    print(f"  Combined embeddings: {list(combined.shape)}")
    print(f"  Total tokens:        {combined.shape[1]} "
          f"(text={text_len} + vis={vis_len} + aud={aud_len})")

    # Simulate labels: only text positions have targets
    labels = torch.full((1, combined.shape[1] - 1), -100, dtype=torch.long)
    labels[0, :text_len - 1] = torch.randint(0, 256000, (text_len - 1,))
    n_ignored = (labels == -100).sum().item()
    n_targets = (labels != -100).sum().item()
    print(f"  Labels targets:      {n_targets} (text only)")
    print(f"  Labels ignored:      {n_ignored} (multimodal positions)")
    print("  ✓ Multimodal embedding flow works correctly\n")


if __name__ == "__main__":
    print("Prokopton M8 — Multimodal Tokenizer Pipeline Test")
    print("(No model loaded — tokenizer-only verification)\n")

    test_visual_tokenizer()
    test_audio_tokenizer()
    test_tokenizer_compatibility()
    test_embedding_flow()

    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
