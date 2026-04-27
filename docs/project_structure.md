# Подробный Каталог Файлов И Каталогов

Этот документ отражает текущее состояние репозитория после перехода на structured `v2` пайплайн.

## 1. Корень репозитория

| Путь | Назначение |
|---|---|
| `README.md` | Краткая обзорная документация и быстрый старт |
| `RUN_FROM_SCRATCH.md` | Полный запуск structured-пайплайна с нуля |
| `requirements.txt` | Актуальные зависимости, включая `torch`, `Biopython`, `music21`, `ViennaRNA`, `transformers` |
| `train_bio_music_v2.py` | Основной CLI обучения |
| `generate_from_fasta_v2.py` | Основной CLI генерации |
| `run_pipeline.py` | Legacy orchestration старого проекта |
| `generate_from_fasta.py` | Legacy inference-скрипт |

## 2. Актуальный пакет `bio_music_pipeline/v2/`

| Путь | Назначение |
|---|---|
| `bio_music_pipeline/v2/__init__.py` | Единая точка экспорта актуальных stable API structured `v2` |
| `bio_music_pipeline/v2/config.py` | Dataclass-конфиги и загрузка JSON |
| `bio_music_pipeline/v2/bio.py` | Biological sequence encoder |
| `bio_music_pipeline/v2/structured_music.py` | Извлечение и токенизация `harmony + melody`, MIDI renderer |
| `bio_music_pipeline/v2/structured_pairing.py` | Structured pairing и calibration |
| `bio_music_pipeline/v2/structured_model.py` | `BioConditionedSequenceModel` |
| `bio_music_pipeline/v2/structured_train.py` | Обучение двухступенчатого пайплайна |
| `bio_music_pipeline/v2/structured_generate.py` | Inference `FASTA -> harmony -> melody -> MIDI` |

## 3. Переходные и legacy модули внутри `v2`

Эти файлы всё ещё есть и импортируются напрямую при необходимости воспроизвести старые эксперименты, но не экспортируются из `bio_music_pipeline.v2` как stable API:

| Путь | Роль |
|---|---|
| `bio_music_pipeline/v2/dataset.py` | Более ранний single-stream symbolic слой |
| `bio_music_pipeline/v2/pairing.py` | Более ранний pairing для single-stream `v2` |
| `bio_music_pipeline/v2/model.py` | Более ранняя single-stream модель |
| `bio_music_pipeline/v2/train.py` | Старый train orchestration `v2` |
| `bio_music_pipeline/v2/generate.py` | Старый inference orchestration `v2` |

## 4. Legacy-стек `bio_music_pipeline/*`

| Путь | Назначение |
|---|---|
| `bio_music_pipeline/extractors/` | Старый extraction layer |
| `bio_music_pipeline/sonification/` | Старый deterministic sonification |
| `bio_music_pipeline/data/` | Старые dataset и paired creator |
| `bio_music_pipeline/models/` | Старый conditioned transformer |
| `bio_music_pipeline/baselines/` | Старые baseline-генераторы |
| `bio_music_pipeline/evaluation/` | Старый evaluation framework |
| `bio_music_pipeline/utils/` | Общие legacy-утилиты |

## 5. Конфиги

| Путь | Назначение |
|---|---|
| `configs/pipeline_v2_small.json` | Основной рабочий конфиг под `RTX 2060 6 GB` |
| `configs/pipeline_config.json` | Legacy конфиг |
| `configs/pipeline_full_paired.json` | Legacy paired-конфиг |
| `configs/pipeline_quick_paired.json` | Legacy quick paired |
| `configs/pipeline_quick_paired_v2.json` | Legacy quick paired v2 |

## 6. Данные

### `data/fasta/`

| Путь | Назначение |
|---|---|
| `data/fasta/quick_sample.fa` | Demo FASTA для smoke- и quick-run |
| `data/fasta/README.txt` | Краткая инструкция по FASTA |

### `data/midi/`

| Путь | Назначение |
|---|---|
| `data/midi/polyphonic_music21/` | Fallback полифонический корпус, экспортируемый через `music21` |
| `data/midi/monophonic_extracted/` | Старый монофонический набор |
| `data/midi/README.txt` | Краткая инструкция по MIDI |

## 7. Тесты

| Путь | Назначение |
|---|---|
| `tests/test_v2_pipeline.py` | Smoke tests structured `v2` пайплайна |

## 8. Результаты

Runtime-артефакты из `results/`, `outputs/`, `tmp/` и `web/output/` игнорируются git. Ниже перечислена ожидаемая структура локальных результатов, а не файлы, которые нужно коммитить.

### После обучения

Основной каталог:

- `results/v2_music21_rtx2060/`

Ключевые артефакты:

| Путь | Назначение |
|---|---|
| `results/v2_music21_rtx2060/resolved_config.json` | Зафиксированный конфиг запуска |
| `results/v2_music21_rtx2060/metrics.json` | История `harmony` и `melody` обучения |
| `results/v2_music21_rtx2060/checkpoints/harmony_best.pt` | Лучший checkpoint гармонической модели |
| `results/v2_music21_rtx2060/checkpoints/melody_best.pt` | Лучший checkpoint мелодической модели |
| `results/v2_music21_rtx2060/checkpoints/structured_pipeline.pt` | Совмещённый checkpoint для inference |
| `results/v2_music21_rtx2060/pairing/pair_manifest.json` | Structured pairing manifest |
| `results/v2_music21_rtx2060/pairing/pair_calibration.npz` | Калибровка bio profile к music distribution |
| `results/v2_music21_rtx2060/smoke/structured_sample.mid` | Smoke-generation после обучения |

### После генерации

| Путь | Назначение |
|---|---|
| `results/v2_generation/structured_from_fasta.mid` | Итоговый MIDI из FASTA |
| `results/v2_generation/structured_from_fasta.json` | Метаданные генерации |

## 9. Web-часть

| Путь | Назначение |
|---|---|
| `web/app.py` | Flask entrypoint для structured `v2` |
| `web/generator.py` | Web wrapper вокруг `generate_structured_music_from_fasta()` |
| `web/midi_to_audio.py` | Опциональная конвертация MIDI в WAV |
| `web/templates/` | HTML-шаблоны |
| `web/static/` | CSS и JS |

Web-слой использует structured `v2` checkpoint `structured_pipeline.pt` и пишет runtime-файлы в `web/output/`.

## 10. Legacy policy

Подробно см. `docs/legacy.md`. Коротко: `run_pipeline.py`, `generate_from_fasta.py`, старые single-stream модули внутри `bio_music_pipeline/v2/` и старый top-level stack оставлены для воспроизводимости и сравнения, но новые функции должны строиться вокруг structured `v2`.
