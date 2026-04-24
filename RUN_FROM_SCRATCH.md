# Полный запуск с нуля: актуальный `v2` пайплайн

Этот документ описывает **текущий рабочий способ** запуска проекта после последних изменений. Основной контур теперь находится в `bio_music_pipeline/v2` и рассчитан на:

- полифоническую музыку без сведения к монофонии;
- biologically informed encoding из FASTA;
- обучение компактной conditional Transformer-модели на `RTX 2060 6 GB`;
- отдельный проверяемый путь `train -> checkpoint -> generate`.

Старый стек `run_pipeline.py` и связанные модули сохранены в репозитории как legacy-контур, но **не являются рекомендуемым способом запуска**.

## 1. Что нужно на входе

- FASTA-файл или каталог FASTA-файлов
- MIDI-корпус

Если своего полифонического MIDI-корпуса пока нет, `v2` умеет автоматически поднять fallback-корпус через `music21` и экспортировать локальные полифонические MIDI в `data/midi/polyphonic_music21/`.

## 2. Требования

- Windows 11 или другой современный desktop environment
- Python `3.12`
- NVIDIA GPU, желательно CUDA-совместимая
- для текущего локального сценария проверено на `RTX 2060 6 GB`

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
- корректное имя GPU

## 5. Подготовка данных

Минимальная структура:

```text
data/
  fasta/
    quick_sample.fa
  midi/
    polyphonic_music21/      # может создаться автоматически
```

Если у вас есть свой полифонический корпус, укажите путь к нему в `configs/pipeline_v2_small.json`:

- поле `music.midi_dirs`

Если у вас свой FASTA-корпус:

- замените `fasta_path`
- при необходимости настройте `bio.fragment_length`, `bio.fragment_stride`, `bio.max_fragments_per_record`

## 6. Проверка кода до запуска обучения

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_v2_pipeline.py
```

Ожидаемое поведение:

- `3 passed`

## 7. Запуск обучения

Базовый запуск:

```powershell
.\.venv\Scripts\python.exe train_bio_music_v2.py --config configs\pipeline_v2_small.json
```

Что делает этот запуск:

1. читает FASTA и режет последовательности на фрагменты;
2. строит `bio embedding` и `control profile`;
3. поднимает полифонический музыкальный корпус;
4. сегментирует MIDI на полифонические окна;
5. делает pairing по многомерным музыкальным дескрипторам;
6. обучает conditional Transformer;
7. сохраняет checkpoint, метрики и smoke MIDI.

## 8. Что должно появиться после обучения

Основные артефакты:

- `results/v2_music21_rtx2060/resolved_config.json`
- `results/v2_music21_rtx2060/pairing/pair_manifest.json`
- `results/v2_music21_rtx2060/pairing/pair_calibration.npz`
- `results/v2_music21_rtx2060/checkpoints/best_model.pt`
- `results/v2_music21_rtx2060/metrics.json`
- `results/v2_music21_rtx2060/smoke/sample_from_training_pipeline.mid`

## 9. Как проверять корректность обучения

### Во время запуска

Проверьте, что процесс не падает и создаются промежуточные каталоги:

- `results/v2_music21_rtx2060/pairing/`
- `results/v2_music21_rtx2060/checkpoints/`

### После запуска

Откройте:

```powershell
Get-Content results\v2_music21_rtx2060\metrics.json
```

Что считать нормальным:

- `device` = `cuda`
- `history[*].val.loss` в среднем уменьшается по эпохам
- checkpoint существует
- smoke MIDI существует

## 10. Запуск генерации из FASTA

После обучения:

```powershell
.\.venv\Scripts\python.exe generate_from_fasta_v2.py --config configs\pipeline_v2_small.json --checkpoint results\v2_music21_rtx2060\checkpoints\best_model.pt --fasta data\fasta\quick_sample.fa --output results\v2_generation\generated_from_fasta.mid --metadata-output results\v2_generation\generated_from_fasta.json
```

## 11. Как проверять корректность генератора

Проверьте, что появились:

- `results/v2_generation/generated_from_fasta.mid`
- `results/v2_generation/generated_from_fasta.json`

Быстрая техническая проверка:

```powershell
@'
from music21 import converter
score = converter.parse("results/v2_generation/generated_from_fasta.mid")
notes = list(score.flatten().notes)
print("notes:", len(notes))
print("duration:", float(score.highestTime))
'@ | .\.venv\Scripts\python.exe -
```

Нормальный результат:

- `notes > 0`
- ненулевая длительность

Если нужна ещё одна sanity-проверка на полифонию:

```powershell
@'
from music21 import converter
from collections import Counter
score = converter.parse("results/v2_generation/generated_from_fasta.mid")
notes = list(score.flatten().notes)
onsets = Counter(round(float(n.offset), 3) for n in notes)
print("polyphonic_onsets:", sum(v > 1 for v in onsets.values()))
print("max_simultaneous:", max(onsets.values()) if onsets else 0)
'@ | .\.venv\Scripts\python.exe -
```

## 12. Где менять параметры под свой датасет

Основной файл:

- `configs/pipeline_v2_small.json`

Чаще всего меняются:

- `output_dir`
- `fasta_path`
- `music.midi_dirs`
- `music.max_music21_files`
- `training.num_epochs`
- `training.batch_size`
- `training.grad_accum_steps`
- `generation.temperature`
- `generation.top_k`
- `generation.top_p`

## 13. Что уже проверено локально

На текущем устройстве и в текущем состоянии репозитория проверено:

- окружение установлено;
- `torch` видит `RTX 2060`;
- `pytest` проходит;
- обучение `v2` завершается успешно;
- сохраняется `best_model.pt`;
- отдельный inference-скрипт генерирует MIDI из FASTA;
- smoke и final generation содержат полифонические onset’ы.

## 14. Legacy-контур

Старые команды вида:

- `python run_pipeline.py ...`
- `python -m bio_music_pipeline.data.paired_dataset_creator ...`

относятся к прежнему стеку. Они оставлены для исторической совместимости и анализа старой архитектуры, но не описывают текущий рекомендованный production-like сценарий запуска.
