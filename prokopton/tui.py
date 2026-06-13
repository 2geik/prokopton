"""
Prokopton TUI — Terminal User Interface
========================================
Textual tabanlı, kullanıcı dostu arayüz.
Konuştukça öğrenen LLM ile sohbet, model yönetimi, bellek kontrolü.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Optional

import torch
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    ProgressBar,
    RichLog,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
    TextArea,
)
from textual.widget import Widget
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.console import RenderableType

# Prokopton imports
from prokopton.core import Prokopton, ProkoptonConfig
from prokopton.models import AVAILABLE_MODELS
from prokopton.backends import (
    detect_backend,
    load_model,
    apply_backend_patches,
    get_vram_usage,
    backend_summary,
    BackendInfo,
)


# ============================================================
# MODEL DOWNLOADER
# ============================================================

def download_model_from_hf(
    url_or_id: str, target_dir: str = "models", callback=None
) -> bool:
    """
    HuggingFace URL'sinden veya model ID'sinden model indir.

    Desteklenen formatlar:
        - huggingface.co/google/gemma-4-E2B
        - hf.co/google/gemma-4-E2B
        - google/gemma-4-E2B (doğrudan ID)
    """
    from huggingface_hub import snapshot_download, hf_hub_url

    # URL'den model ID çıkar
    model_id = url_or_id.strip()
    for prefix in ["https://huggingface.co/", "https://hf.co/", "http://huggingface.co/", "http://hf.co/"]:
        if model_id.startswith(prefix):
            model_id = model_id[len(prefix):]
            break

    # /tree/main veya /blob/... kısımlarını temizle
    model_id = model_id.split("/tree/")[0].split("/blob/")[0].rstrip("/")

    if not model_id or "/" not in model_id:
        if callback:
            callback(f"❌ Geçersiz model: {model_id}")
        return False

    target_path = Path(target_dir) / model_id.replace("/", "_")
    target_path.mkdir(parents=True, exist_ok=True)

    if callback:
        callback(f"📥 İndiriliyor: {model_id}")

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target_path),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        if callback:
            callback(f"✅ İndirme tamam: {target_path}")
        return True
    except Exception as e:
        if callback:
            callback(f"❌ Hata: {e}")
        return False


def find_local_models(models_dir: str = "models") -> list[dict]:
    """models/ klasöründeki yerel modelleri tara."""
    models = []
    p = Path(models_dir)
    if not p.exists():
        return models

    for d in sorted(p.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            config_file = d / "config.json"
            is_valid = config_file.exists()
            size_gb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9
            models.append({
                "id": d.name,
                "path": str(d),
                "valid": is_valid,
                "size_gb": size_gb,
            })
    return models


# ============================================================
# WIDGETS
# ============================================================

class ChatMessage(Widget):
    """Sohbet mesajı widget'ı."""

    def __init__(self, sender: str, text: str, is_user: bool = False):
        super().__init__()
        self.sender = sender
        self.text = text
        self.is_user = is_user

    def render(self) -> RenderableType:
        style = "bold green" if self.is_user else "bold cyan"
        prefix = "👤 Sen" if self.is_user else "🤖 Prokopton"
        return Panel(
            self.text,
            title=f"[{style}]{prefix}[/]",
            border_style="green" if self.is_user else "blue",
        )


class StatsPanel(Widget):
    """İstatistik paneli widget'ı."""

    stats: reactive[dict] = reactive({})

    def render(self) -> RenderableType:
        if not self.stats:
            return Panel("Henüz istatistik yok.", title="📊 İstatistikler")

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="bold cyan")
        table.add_column("Value", style="white")

        labels = {
            "steps": "Adım",
            "updates": "TTT Güncelleme",
            "weight_change": "Ağırlık Δ",
            "total_surprise": "Toplam Sürpriz",
            "buffer_size": "Buffer",
            "history_len": "Geçmiş",
        }

        for key, val in self.stats.items():
            label = labels.get(key, key)
            if isinstance(val, float):
                val = f"{val:.4f}"
            table.add_row(label, str(val))

        return Panel(table, title="📊 İstatistikler", border_style="yellow")


class ModelInfoBar(Widget):
    """Model bilgi çubuğu."""

    model_name: reactive[str] = reactive("")
    vram: reactive[float] = reactive(0.0)
    ttt_layers: reactive[int] = reactive(0)

    def render(self) -> RenderableType:
        text = Text()
        text.append("🧠 ", style="bold")
        text.append(f"{self.model_name or 'Yüklenmedi'}", style="bold cyan")
        if self.vram > 0:
            text.append(f"  |  💾 {self.vram:.1f} GB VRAM", style="dim")
        if self.ttt_layers > 0:
            text.append(f"  |  ⚡ {self.ttt_layers} TTT katmanı", style="dim yellow")
        return text


# ============================================================
# SCREENS
# ============================================================

class ModelDownloadScreen(ModalScreen[bool]):
    """Model indirme ekranı (modal)."""

    def compose(self) -> ComposeResult:
        yield Container(
            Label("📥 HuggingFace Model İndirme", classes="title"),
            Label("HF URL veya Model ID girin (örn: google/gemma-4-E2B):"),
            Input(placeholder="google/gemma-4-E2B", id="model_url"),
            Label("veya hızlı seç:", classes="dim"),
            Select(
                [(f"{v['name']} ({v['params']})", v['name'])
                 for k, v in AVAILABLE_MODELS.items()],
                prompt="Hızlı model seç...",
                id="quick_model",
            ),
            RichLog(id="download_log", max_lines=12, highlight=True),
            Horizontal(
                Button("⬇ İndir", variant="primary", id="btn_download"),
                Button("❌ Kapat", variant="error", id="btn_close"),
            ),
            id="download_dialog",
        )

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "btn_close":
            self.dismiss(False)
        elif event.button.id == "btn_download":
            self._do_download()

    def _do_download(self):
        url_input = self.query_one("#model_url", Input)
        quick = self.query_one("#quick_model", Select)
        log = self.query_one("#download_log", RichLog)

        url = url_input.value.strip()
        if not url and quick.value != Select.BLANK:
            url = str(quick.value)

        if not url:
            log.write("⚠️ Model URL veya ID girin!")
            return

        btn = self.query_one("#btn_download", Button)
        btn.disabled = True
        btn.label = "⏳ İndiriliyor..."

        def log_cb(msg):
            log.write(msg)

        success = download_model_from_hf(url, "models", log_cb)

        if success:
            log.write("\n✅ Başarılı! Kapatıp modeli ana ekrandan seçebilirsiniz.")
            btn.label = "✅ Tamam"
            self.set_timer(1.5, lambda: self.dismiss(True))
        else:
            btn.label = "⬇ Tekrar Dene"
            btn.disabled = False


class ModelSelectScreen(ModalScreen[Optional[str]]):
    """Model seçim ekranı (modal)."""

    def compose(self) -> ComposeResult:
        yield Container(
            Label("🧠 Model Seçimi", classes="title"),
            Label("Yerel modeller (models/ klasörü):"),
            VerticalScroll(
                *self._build_model_buttons(),
                id="local_models",
            ),
            Label(""),
            Label("Veya HuggingFace model ID'si girin:"),
            Input(placeholder="google/gemma-4-E2B", id="hf_model_id"),
            Horizontal(
                Button("✅ Bu Modeli Yükle", variant="primary", id="btn_load"),
                Button("⬇ HF'den İndir", variant="default", id="btn_download_screen"),
                Button("❌ İptal", variant="error", id="btn_cancel"),
            ),
            id="model_select_dialog",
        )

    def _build_model_buttons(self):
        buttons = []
        # Yerel modeller
        local = find_local_models()
        if local:
            for m in local:
                status = "✓" if m["valid"] else "⚠"
                label = f"{status} {m['id']} ({m['size_gb']:.1f} GB)"
                buttons.append(Button(label, id=f"local_{m['id']}", variant="default"))
        else:
            buttons.append(Label("  (models/ klasörü boş — HF'den indirebilirsiniz)", classes="dim"))

        # Bilinen modeller
        buttons.append(Label(""))
        buttons.append(Label("Bilinen modeller (ilk kullanımda HF'den iner):"))
        for key, info in AVAILABLE_MODELS.items():
            label = f"  🌐 {info['name']} ({info['params']}, {info['vram_bf16']})"
            buttons.append(Button(label, id=f"known_{info['name']}", variant="default"))

        return buttons

    def on_button_pressed(self, event: Button.Pressed):
        bid = event.button.id
        if bid == "btn_cancel":
            self.dismiss(None)
        elif bid == "btn_download_screen":
            self.app.push_screen(ModelDownloadScreen(), self._after_download)
        elif bid == "btn_load":
            model_id = self.query_one("#hf_model_id", Input).value.strip()
            if model_id:
                self.dismiss(model_id)
        elif bid.startswith("local_"):
            model_name = bid[6:]  # 'local_' sonrası
            model_path = Path("models") / model_name
            self.dismiss(str(model_path))
        elif bid.startswith("known_"):
            model_id = bid[6:]  # 'known_' sonrası
            self.dismiss(model_id)

    def _after_download(self, success: bool):
        if success:
            # Ekranı yenile
            self.app.pop_screen()
            self.app.push_screen(ModelSelectScreen(), self._return_result)

    def _return_result(self, result):
        pass


# ============================================================
# ANA TUI UYGULAMASI
# ============================================================

class ProkoptonTUI(App):
    """Prokopton Terminal Arayüzü."""

    CSS = """
    #model_bar {
        height: auto;
        padding: 1;
        border: solid $accent;
    }
    #chat_log {
        height: 1fr;
        border: solid $surface;
    }
    #stats_panel {
        height: 1fr;
        border: solid $surface;
    }
    #input_area {
        height: auto;
        padding: 1;
    }
    .title {
        text-style: bold;
        color: $accent;
        padding: 1;
    }
    .dim {
        color: $text-disabled;
        padding: 1;
    }
    #download_dialog, #model_select_dialog {
        width: 70%;
        height: auto;
        max-height: 90%;
        background: $surface;
        border: thick $accent;
        padding: 1;
        margin: 2 4;
    }
    Screen {
        align: center middle;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Çıkış", show=True),
        Binding("ctrl+s", "save_memory", "Kaydet", show=True),
        Binding("ctrl+l", "load_memory", "Yükle", show=True),
        Binding("ctrl+r", "reset_memory", "Sıfırla", show=True),
        Binding("ctrl+m", "change_model", "Model Değiştir", show=True),
        Binding("ctrl+d", "download_model", "Model İndir", show=True),
        Binding("ctrl+p", "show_stats", "İstatistik", show=True),
    ]

    def __init__(self, model_arg=None, lr=1e-3, n_layers=5, force_cpu=False, 
                 force_backend=None, save_dir="prokopton_memory"):
        super().__init__()
        self.prokopton: Optional[Prokopton] = None
        self.model_name: str = ""
        self.model_path: str = ""
        self._model_arg = model_arg
        self._force_cpu = force_cpu
        self._force_backend = force_backend
        self.backend: Optional[BackendInfo] = None
        self.config = ProkoptonConfig(
            save_dir=save_dir,
            ttt_lr=lr,
            ttt_n_layers=n_layers,
        )

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield ModelInfoBar(id="model_bar")
        yield TabbedContent(
            TabPane("💬 Sohbet", Container(
                RichLog(id="chat_log", markup=True, wrap=True, max_lines=500),
                Horizontal(
                    Input(placeholder="Mesajınızı yazın... (Enter: gönder)", id="msg_input"),
                    Button("Gönder", variant="primary", id="btn_send"),
                    id="input_area",
                ),
            )),
            TabPane("📊 İstatistikler", StatsPanel(id="stats_panel")),
            TabPane("⚙️ Ayarlar", VerticalScroll(
                Label("TTT Ayarları", classes="title"),
                Label("Öğrenme Hızı (lr):"),
                Input(value="0.001", id="cfg_lr"),
                Label("TTT Katman Sayısı:"),
                Input(value="5", id="cfg_layers"),
                Label("CMS Rank:"),
                Input(value="16", id="cfg_rank"),
                Label("Otomatik Kaydet (adım):"),
                Input(value="100", id="cfg_autosave"),
                Button("💾 Ayarları Uygula", variant="primary", id="btn_apply_cfg"),
                Label(""),
                Label("Bellek Yönetimi", classes="title"),
                Button("💾 Kaydet", variant="default", id="btn_save"),
                Button("📂 Yükle", variant="default", id="btn_load"),
                Button("♻ Sıfırla", variant="warning", id="btn_reset"),
                Button("💾 Değişmiş Modeli Kaydet", variant="primary", id="btn_save_model"),
            )),
        )
        yield Footer()

    def on_mount(self):
        """Detect backend and show model selection or load directly."""
        if self._force_cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # Detect optimal backend
        self.backend = detect_backend(force=self._force_backend or ("cpu" if self._force_cpu else None))
        info = backend_summary(self.backend)
        self._log_chat(
            f"🖥️  [bold]{info['description']}[/] | {info['gpu']}"
            + (f" | {info['vram_gb']} GB" if info['vram_gb'] > 0 else ""),
            "cyan"
        )

        if self._model_arg:
            self._log_chat(f"⏳ Model yükleniyor: {self._model_arg}...", "yellow")
            self._load_model(self._model_arg)
        else:
            self.set_timer(0.5, self._prompt_model_select)

    def _prompt_model_select(self):
        self.push_screen(ModelSelectScreen(), self._on_model_selected)

    def _on_model_selected(self, model_choice: Optional[str]):
        if model_choice is None:
            self._log_chat("⚠️ Model seçilmedi. Çıkmak için Ctrl+Q.", "yellow")
            return

        self._log_chat(f"⏳ Model yükleniyor: {model_choice}...", "yellow")
        self._load_model(model_choice)

    def _load_model(self, model_id_or_path: str):
        """Modeli yükle — platform-agnostic."""
        try:
            # Apply platform patches
            apply_backend_patches(self.backend)

            self._log_chat(f"📥 Yükleniyor: {model_id_or_path}...", "yellow")

            # Use unified loader
            model, tokenizer = load_model(model_id_or_path, self.backend)

            vram = get_vram_usage(self.backend)
            self.model_name = model_id_or_path

            self.prokopton = Prokopton(model, tokenizer, self.config)

            # UI güncelle
            bar = self.query_one("#model_bar", ModelInfoBar)
            bar.model_name = self.model_name
            bar.vram = vram or self.backend.vram_gb
            bar.ttt_layers = len(self.prokopton.fast_weights)

            bsummary = backend_summary(self.backend)
            self._log_chat(
                f"✅ Model hazır: {self.model_name} | "
                f"{bsummary['description']} | "
                f"{bsummary['vram_gb']} GB VRAM | "
                f"{len(self.prokopton.fast_weights)} TTT katmanı",
                "green",
            )

            # Önceki bellek varsa yükle
            if self.prokopton.load(silent_on_missing=True):
                self._log_chat("📂 Önceki bellek yüklendi.", "cyan")

            self._update_stats()
            self.query_one("#msg_input", Input).focus()

        except Exception as e:
            self._log_chat(f"❌ Model yüklenemedi: {e}", "red")

    def action_quit(self):
        if self.prokopton:
            self.prokopton.save(silent=True)
        self.exit()

    def action_save_memory(self):
        if self.prokopton:
            self.prokopton.save()
            self._log_chat("💾 Bellek kaydedildi.", "green")

    def action_load_memory(self):
        if self.prokopton:
            if self.prokopton.load():
                self._log_chat("📂 Bellek yüklendi.", "cyan")
                self._update_stats()

    def action_reset_memory(self):
        if self.prokopton:
            self.prokopton.reset()
            self._log_chat("♻ Bellek sıfırlandı.", "yellow")
            self._update_stats()

    def action_change_model(self):
        self.push_screen(ModelSelectScreen(), self._on_model_selected)

    def action_download_model(self):
        self.push_screen(ModelDownloadScreen(), self._after_download)

    def _after_download(self, success: bool):
        if success:
            self._log_chat("✅ Model indirildi. Ctrl+M ile seçebilirsiniz.", "green")

    def action_show_stats(self):
        self._update_stats()
        self.query_one(TabbedContent).active = "tabpane-2"  # İstatistikler

    @on(Button.Pressed, "#btn_send")
    @on(Input.Submitted, "#msg_input")
    def on_send_message(self, event):
        msg_input = self.query_one("#msg_input", Input)
        user_text = msg_input.value.strip()
        if not user_text:
            return

        msg_input.value = ""
        self._log_chat(f"👤 [bold green]Sen:[/] {user_text}")

        if not self.prokopton:
            self._log_chat("⚠️ Önce model yükleyin! Ctrl+M", "yellow")
            return

        try:
            response = self.prokopton.chat(user_text, max_new=256)
            self._log_chat(f"🤖 [bold cyan]Prokopton:[/] {response}")
            self._update_stats()
        except Exception as e:
            self._log_chat(f"❌ Hata: {e}", "red")

    @on(Button.Pressed, "#btn_save")
    def on_save(self):
        self.action_save_memory()

    @on(Button.Pressed, "#btn_load")
    def on_load(self):
        self.action_load_memory()

    @on(Button.Pressed, "#btn_reset")
    def on_reset(self):
        self.action_reset_memory()

    @on(Button.Pressed, "#btn_save_model")
    def on_save_model(self):
        """Değişmiş modeli diske kaydet."""
        if not self.prokopton:
            self._log_chat("⚠️ Önce model yükleyin!", "yellow")
            return

        try:
            save_path = Path("prokopton_model")
            save_path.mkdir(exist_ok=True)

            # CMS adaptörlerini base modele göm
            for cms in self.prokopton.cms_adapters:
                cms.consolidate()
                cms.apply_to_model()

            self.prokopton.model.save_pretrained(str(save_path))
            self.prokopton.tokenizer.save_pretrained(str(save_path))

            self._log_chat(f"💾 Değişmiş model kaydedildi: {save_path}/", "green")
        except Exception as e:
            self._log_chat(f"❌ Kaydetme hatası: {e}", "red")

    @on(Button.Pressed, "#btn_apply_cfg")
    def on_apply_config(self):
        try:
            lr = float(self.query_one("#cfg_lr", Input).value)
            layers = int(self.query_one("#cfg_layers", Input).value)
            rank = int(self.query_one("#cfg_rank", Input).value)
            autosave = int(self.query_one("#cfg_autosave", Input).value)

            self.config.ttt_lr = lr
            self.config.ttt_n_layers = layers
            self.config.cms_rank = rank
            self.config.auto_save_every = autosave

            self._log_chat(
                f"⚙️ Ayarlar güncellendi: lr={lr}, layers={layers}, rank={rank}",
                "green",
            )

            # Mevcut model varsa yeniden TTT kur
            if self.prokopton:
                self.prokopton.config = self.config
                self.prokopton._setup_ttt()
                bar = self.query_one("#model_bar", ModelInfoBar)
                bar.ttt_layers = len(self.prokopton.fast_weights)
        except ValueError as e:
            self._log_chat(f"⚠️ Geçersiz değer: {e}", "yellow")

    def _log_chat(self, text: str, color: str = None):
        log = self.query_one("#chat_log", RichLog)
        if color:
            log.write(f"[{color}]{text}[/]")
        else:
            log.write(text)

    def _update_stats(self):
        if self.prokopton:
            stats = self.prokopton.stats
            panel = self.query_one("#stats_panel", StatsPanel)
            panel.stats = stats


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    """Prokopton TUI başlat."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prokopton — Konuştukça öğrenen, unutmayan LLM (TUI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  prokopton                              # Interactive TUI
  prokopton --model google/gemma-4-E2B   # Skip model selection
  prokopton --lr 0.001 --n-layers 5      # Custom TTT settings
  prokopton --cpu                        # Force CPU mode

Keybindings (inside TUI):
  Ctrl+Q  Quit          Ctrl+S  Save memory
  Ctrl+L  Load memory   Ctrl+R  Reset memory
  Ctrl+M  Switch model  Ctrl+D  Download from HF
  Ctrl+P  Show stats    Enter   Send message
        """,
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model name or path (e.g. google/gemma-4-E2B, or local models/... path)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="TTT learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=5,
        help="Number of TTT layers (default: 5)",
    )
    parser.add_argument(
        "--no-ttt",
        action="store_true",
        help="Disable TTT (frozen model mode)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode (no GPU)",
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["rocm", "cuda", "mps", "mlx", "cpu"],
        default=None,
        help="Force specific backend (auto-detected by default)",
    )
    parser.add_argument(
        "--save-dir",
        default="prokopton_memory",
        help="Memory save directory (default: prokopton_memory)",
    )

    args = parser.parse_args()

    if args.no_ttt:
        args.lr = 0.0

    app = ProkoptonTUI(
        model_arg=args.model,
        lr=args.lr,
        n_layers=args.n_layers,
        force_cpu=args.cpu,
        force_backend=args.backend,
        save_dir=args.save_dir,
    )
    app.run()


if __name__ == "__main__":
    main()
