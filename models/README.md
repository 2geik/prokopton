# Model Klasörü

Bu klasöre HuggingFace modellerini yerleştirebilirsiniz.

## Otomatik İndirme (Önerilen)

TUI içinden `Ctrl+D` ile istediğiniz modeli indirebilirsiniz.

## Manuel Yükleme

```bash
# HuggingFace CLI ile
huggingface-cli download google/gemma-4-E2B --local-dir models/gemma-4-E2B

# Veya manuel olarak model dosyalarını kopyalayın
# Bir model klasöründe şunlar olmalı:
#   config.json
#   model.safetensors (veya pytorch_model.bin)
#   tokenizer.json
#   tokenizer_config.json
```

## Desteklenen Formatlar

- HuggingFace transformers formatı (config.json + safetensors)
- GGUF dosyaları (ileride desteklenecek)

TUI otomatik olarak bu klasördeki geçerli modelleri tarar ve listeler.
