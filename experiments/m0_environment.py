"""
M0 — Ortam Doğrulama ve İskelet Testi
======================================
Prokopton'un çalışması için gereken tüm altyapıyı doğrular:
- ROCm / CUDA / MPS / CPU backend algılama
- PyTorch versiyonu ve GPU uyumluluğu
- Temel tensor operasyonları
- transformers kütüphanesi
- titans-pytorch (isteğe bağlı)
- Prokopton paket kurulumu
- VRAM durumu

Kullanım:
  .venv/bin/python experiments/m0_environment.py
  .venv/bin/python experiments/m0_environment.py --verbose
"""

import sys
import argparse
import time
from pathlib import Path


def section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def check(name: str, passed: bool, detail: str = "", warn: bool = False):
    icon = "✓" if passed else ("⚠" if warn else "✗")
    status = f"{icon} {name}"
    if detail:
        status += f"  → {detail}"
    print(f"  {status}")
    return passed


def main(verbose: bool = False):
    results = []
    start = time.time()

    print("=" * 60)
    print("  M0 — Prokopton Ortam Doğrulama")
    print("=" * 60)

    # === 1. Python ===
    section("1. Python Ortamı")
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ok = sys.version_info >= (3, 10)
    results.append(check("Python ≥ 3.10", ok, py_ver))

    # === 2. PyTorch ===
    section("2. PyTorch")
    try:
        import torch
        results.append(check("PyTorch kurulu", True, torch.__version__))

        # ROCm / CUDA kontrolü
        if hasattr(torch.version, 'hip') and torch.version.hip:
            hip_ver = torch.version.hip
            results.append(check("ROCm (HIP)", True, f"HIP {hip_ver}"))
        elif torch.cuda.is_available():
            cuda_ver = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
            results.append(check("CUDA", True, f"CUDA {cuda_ver}, {gpu_name}"))
        else:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                results.append(check("GPU", True, "Apple MPS", warn=True))
            else:
                results.append(check("GPU", False, "CPU only", warn=True))

        # Cihaz sayısı
        if torch.cuda.is_available():
            n_gpu = torch.cuda.device_count()
            for i in range(n_gpu):
                name = torch.cuda.get_device_name(i)
                mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                results.append(check(f"GPU[{i}]", True, f"{name} — {mem_total:.1f} GB"))

        # Temel tensor testi
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            a = torch.randn(100, 100, device=device)
            b = torch.randn(100, 100, device=device)
            c = a @ b
            results.append(check("Tensor matmul", True, f"{c.shape}, device={device}"))
        except Exception as e:
            results.append(check("Tensor matmul", False, str(e)))

        # VRAM kullanımı
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            if verbose:
                results.append(check("VRAM allocated", True, f"{allocated:.2f} GB"))
                results.append(check("VRAM reserved", True, f"{reserved:.2f} GB"))

    except ImportError as e:
        results.append(check("PyTorch kurulu", False, str(e)))
        print("\n  ⚠ PyTorch bulunamadı! Sonraki testler atlanıyor.")
        print(f"     Hata: {e}")

    # === 3. Transformers ===
    section("3. HuggingFace Transformers")
    try:
        import transformers
        tf_ver = transformers.__version__
        results.append(check("transformers", True, tf_ver))

        # Gemma uyumluluğu
        try:
            from transformers import AutoConfig
            # Gemma 4 config check — varmış gibi test etmiyoruz,
            # sadece AutoConfig'in çalıştığını doğrula
            results.append(check("AutoConfig API", True, "hazır"))
        except Exception as e:
            results.append(check("AutoConfig API", False, str(e)))

    except ImportError:
        results.append(check("transformers", False, "pip install transformers"))
        print("  ⚠ transformers bulunamadı!")

    # === 4. Prokopton Paketi ===
    section("4. Prokopton Paketi")
    try:
        import prokopton
        ok = hasattr(prokopton, '__version__')
        ver = getattr(prokopton, '__version__', 'bilinmiyor')
        results.append(check("prokopton import", ok, f"v{ver}"))

        # Core modülleri
        try:
            from prokopton.core import Prokopton, ProkoptonConfig, FastWeight, CMSAdapter
            results.append(check("prokopton.core", True, "Prokopton, Config, FastWeight, CMS"))
        except ImportError as e:
            results.append(check("prokopton.core", False, str(e)))

        # Backend
        try:
            from prokopton.backends import detect_backend, load_model, apply_backend_patches
            backend = detect_backend()
            results.append(check("prokopton.backends", True, f"Algılandı: {backend.name}"))
            if verbose:
                print(f"         Detay: {backend.description}")
                print(f"         Device: {backend.device}")
                print(f"         VRAM: {backend.vram_gb:.1f} GB" if backend.vram_gb > 0 else "")
        except ImportError as e:
            results.append(check("prokopton.backends", False, str(e)))

        # REPL
        try:
            from prokopton import repl
            results.append(check("prokopton.repl", True, "hazır"))
        except ImportError:
            results.append(check("prokopton.repl", False, "modül yok", warn=True))

        # TUI
        try:
            from prokopton import tui
            results.append(check("prokopton.tui", True, "hazır"))
        except ImportError:
            results.append(check("prokopton.tui", False, "modül yok (textual kurulu değil?)", warn=True))

        # Eval
        try:
            from prokopton.eval import CLBenchmark
            results.append(check("prokopton.eval", True, "CLBenchmark hazır"))
        except ImportError:
            results.append(check("prokopton.eval", False, "modül yok", warn=True))

    except ImportError as e:
        results.append(check("prokopton import", False, str(e)))

    # === 5. titans-pytorch (opsiyonel) ===
    section("5. Bağımlılıklar (opsiyonel)")
    try:
        import titans_pytorch
        results.append(check("titans-pytorch", True, f"v{getattr(titans_pytorch, '__version__', '?')}"))
    except ImportError:
        results.append(check("titans-pytorch", False, "opsiyonel — sadece M1(a) için", warn=True))

    # bitsandbytes
    try:
        import bitsandbytes
        results.append(check("bitsandbytes", True, f"v{bitsandbytes.__version__}"))
        if verbose:
            print("         UYARI: ROCm'da 4/8-bit segfault yapabilir — bf16 kullan.")
    except ImportError:
        results.append(check("bitsandbytes", False, "opsiyonel"))

    # textual (TUI için)
    try:
        import textual
        results.append(check("textual", True, f"v{textual.__version__}"))
    except ImportError:
        results.append(check("textual", False, "opsiyonel — TUI için gerekli", warn=True))

    # === 6. ROCm Sistemi ===
    section("6. ROCm Sistem Kontrolü")
    import subprocess
    try:
        out = subprocess.check_output(["rocm-smi", "--showproductname"], text=True, timeout=5)
        results.append(check("rocm-smi", True, "çalışıyor"))
        if verbose:
            for line in out.strip().split("\n")[:5]:
                print(f"         {line}")
    except FileNotFoundError:
        results.append(check("rocm-smi", False, "komut yok", warn=True))
    except Exception as e:
        results.append(check("rocm-smi", False, str(e), warn=True))

    try:
        out = subprocess.check_output(["hipconfig", "--version"], text=True, timeout=5).strip()
        results.append(check("hipconfig", True, out))
    except FileNotFoundError:
        results.append(check("hipconfig", False, "komut yok", warn=True))
    except Exception as e:
        results.append(check("hipconfig", False, str(e), warn=True))

    # === 7. Proje Yapısı ===
    section("7. Proje İskeleti")
    project_root = Path(__file__).parent.parent
    required_dirs = [
        "prokopton/core",
        "prokopton/models",
        "prokopton/backends.py",
        "experiments",
        "tests",
    ]
    for d in required_dirs:
        p = project_root / d
        exists = p.exists()
        results.append(check(f"  {d}", exists))

    # pyproject.toml
    ppt = project_root / "pyproject.toml"
    results.append(check("pyproject.toml", ppt.exists()))

    # === ÖZET ===
    elapsed = time.time() - start
    passed = sum(1 for r in results if r)
    failed = sum(1 for r in results if not r)
    total = len(results)
    rate = passed / total * 100 if total > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"  M0 SONUÇ")
    print(f"{'=' * 60}")
    print(f"  Kontrol: {passed}/{total} başarılı ({rate:.0f}%)")
    print(f"  Süre:    {elapsed:.2f}s")

    if failed == 0:
        print(f"\n  ✓ TÜM KONTROLLER BAŞARILI — Ortam Prokopton için hazır.")
    else:
        print(f"\n  ✗ {failed} kontrol başarısız. Yukarıdaki uyarıları kontrol et.")

    # JSON çıktı
    out_dir = project_root / "experiments" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    result_dict = {
        "experiment": "M0",
        "passed": passed,
        "failed": failed,
        "total": total,
        "success_rate": round(rate, 1),
        "elapsed_s": round(elapsed, 2),
    }
    with open(out_dir / "m0_environment.json", "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"\n  💾 Sonuç: experiments/runs/m0_environment.json")

    return passed == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M0 — Prokopton Ortam Doğrulama")
    parser.add_argument("--verbose", "-v", action="store_true", help="Detaylı çıktı")
    args = parser.parse_args()

    success = main(verbose=args.verbose)
    sys.exit(0 if success else 1)
