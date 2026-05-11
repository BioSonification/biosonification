# Полный запуск с нуля: structured `v2` пайплайн

Этот документ описывает актуальный путь запуска проекта после перехода на иерархическую схему `Bio -> Harmony -> Melody`.

## 1. Что нужно на входе

- FASTA-файл или каталог FASTA-файлов
- полифонический MIDI-корпус

Если своего корпуса пока нет, пайплайн умеет автоматически поднять fallback-корпус `music21` в `data/midi/polyphonic_music21/`.

## 2. Требования

- Python `3.12`
- NVIDIA GPU с CUDA
- локально проверено на `RTX 2060 6 GB`

## 3. Подготовка окружения

Из корня проекта:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 4. Проверка CUDA

```powershell
@'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("memory:", torch.cuda.get_device_properties(0).total_memory)
'@ | .\.venv\Scripts\python.exe -
```

Ожидаемое поведение:

- `cuda: True`
- устройство `NVIDIA GeForce RTX 2060`

## 5. Проверка кода до обучения

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_v2_pipeline.py
```

Ожидаемое поведение:

- `4 passed`

## 6. Базовый конфиг

Основной конфиг:

- `configs/pipeline_v2_small.json`

Он уже настроен под текущую машину:

- `output_dir = results/v2_music21_rtx2060`
- `mixed_precision = true`
- compact Transformer под `RTX 2060 6 GB`
- `use_esm_embedding = false` по умолчанию

Если хотите обучаться на своём корпусе, обычно меняются:

- `fasta_path`
- `music.midi_dirs`
- `music.max_music21_files`
- `training.batch_size`
- `training.harmony_num_epochs`
- `training.melody_num_epochs`

## 7. Запуск обучения

```powershell
.\.venv\Scripts\python.exe train_bio_music_v2.py --config configs\pipeline_v2_small.json
```

Что делает команда:

1. кодирует FASTA через `BiologicalSequenceEncoder`
2. извлекает `bio vector` и `control profile`
3. строит structured музыкальный корпус `harmony + melody`
4. делает pairing bio-фрагментов и музыкальных сегментов
5. обучает отдельную `harmony model`
6. обучает отдельную `melody model`
7. сохраняет checkpoints, метрики и smoke MIDI

## 8. Что должно появиться после обучения

- `results/v2_music21_rtx2060/resolved_config.json`
- `results/v2_music21_rtx2060/pairing/pair_manifest.json`
- `results/v2_music21_rtx2060/pairing/pair_calibration.npz`
- `results/v2_music21_rtx2060/checkpoints/harmony_best.pt`
- `results/v2_music21_rtx2060/checkpoints/melody_best.pt`
- `results/v2_music21_rtx2060/checkpoints/structured_pipeline.pt`
- `results/v2_music21_rtx2060/metrics.json`
- `results/v2_music21_rtx2060/smoke/structured_sample.mid`

## 9. Как проверять корректность обучения

Откройте:

```powershell
Get-Content results\v2_music21_rtx2060\metrics.json
```

Нормальные признаки:

- `device` = `cuda`
- `harmony_history[*].val_loss` в среднем уменьшается
- `melody_history[*].val_loss` в среднем уменьшается
- `harmony_test_loss` и `melody_test_loss` конечные и не `NaN`

Дополнительная проверка smoke MIDI:

```powershell
@'
from music21 import converter, chord, note
score = converter.parse("results/v2_music21_rtx2060/smoke/structured_sample.mid")
print("highestTime:", float(score.highestTime))
for i, part in enumerate(score.parts):
    flat = list(part.flatten().notes)
    print("part", i, "events", len(flat), "notes", sum(isinstance(x, note.Note) for x in flat), "chords", sum(isinstance(x, chord.Chord) for x in flat))
'@ | .\.venv\Scripts\python.exe -
```

Для текущего structured-пайплайна ожидается:

- ровно 2 партии
- гармония состоит из аккордов
- мелодия состоит из одиночных нот
- общая длина не выходит за границы гармонической сетки

## 10. Запуск генерации из FASTA

```powershell
.\.venv\Scripts\python.exe generate_from_fasta_v2.py --config configs\pipeline_v2_small.json --checkpoint results\v2_music21_rtx2060\checkpoints\structured_pipeline.pt --fasta data\fasta\quick_sample.fa --output results\v2_generation\structured_from_fasta.mid --metadata-output results\v2_generation\structured_from_fasta.json
```

## 11. Как проверять корректность генератора

Проверьте, что появились:

- `results/v2_generation/structured_from_fasta.mid`
- `results/v2_generation/structured_from_fasta.json`

Быстрая техническая проверка:

```powershell
@'
from music21 import converter, chord, note
score = converter.parse("results/v2_generation/structured_from_fasta.mid")
print("highestTime:", float(score.highestTime))
for i, part in enumerate(score.parts):
    flat = list(part.flatten().notes)
    print("part", i, "events", len(flat), "notes", sum(isinstance(x, note.Note) for x in flat), "chords", sum(isinstance(x, chord.Chord) for x in flat))
'@ | .\.venv\Scripts\python.exe -
```

Ожидаемое поведение:

- `highestTime` совпадает с длиной `num_bars`
- первая партия аккордовая
- вторая партия мелодическая и остаётся монофонической

Содержимое `structured_from_fasta.json` полезно для контроля:

- `sequence_id`
- `sequence_type`
- `tempo_bpm`
- `tonic_pc_hint`
- `generated_harmony_bars`
- `generated_melody_note_count`

## 12. Что означает `use_esm_embedding`

В `bio`-конфиге есть:

- `use_esm_embedding`
- `esm_model_name`
- `esm_feature_dim`
- `esm_max_length`

Это опциональный слой белковых embedding’ов через `transformers`. В small-конфиге он выключен по умолчанию, потому что на `RTX 2060 6 GB` основное обучение стабильнее держать на заранее вычисляемых более лёгких признаках. Включать его лучше после того, как базовый structured-контур уже работает на ваших данных.

## 13. Что пока не реализовано

Текущий основной результат:

- `Bio -> Harmony`
- `Bio + Harmony -> Melody`

Следующий, пока не реализованный этап:

- `Bio + Harmony + Melody -> Accompaniment`

Именно поэтому текущий MIDI состоит из двух дорожек: аккордовая гармония и монофоническая мелодия.
