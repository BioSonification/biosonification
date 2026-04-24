# Подробный Каталог Файлов И Каталогов

Этот документ описывает **актуальную файловую карту проекта после внедрения `v2` пайплайна**.

## 1. Корень репозитория

| Путь | Тип | Назначение |
|---|---|---|
| `.gitignore` | служебный | Правила игнорирования Git |
| `README.md` | документация | Главная обзорная документация проекта |
| `RUN_FROM_SCRATCH.md` | документация | Актуальный пошаговый запуск `v2` с нуля |
| `DATASETS_GUIDE.md` | документация | Руководство по пользовательским FASTA/MIDI данным |
| `SCIENTIFIC_ARTICLE_AND_TALK_BASE.md` | документация | Текстовая база для статьи и выступления |
| `requirements.txt` | зависимости | Python-зависимости, включая `torch`, `Biopython`, `music21`, `ViennaRNA` |
| `train_bio_music_v2.py` | исполняемый код | Основной CLI запуска обучения нового пайплайна |
| `generate_from_fasta_v2.py` | исполняемый код | Основной CLI запуска генерации нового пайплайна |
| `run_pipeline.py` | legacy-код | Старый оркестратор прежнего пайплайна |
| `generate_from_fasta.py` | legacy-код | Старый одиночный inference-скрипт |
| `scan_datasets.py` | сервисный код | Сканирование каталогов данных |

## 2. Пакет `bio_music_pipeline/`

### 2.1 Актуальный стек `bio_music_pipeline/v2/`

| Путь | Назначение |
|---|---|
| `bio_music_pipeline/v2/__init__.py` | Единая точка экспорта `v2` |
| `bio_music_pipeline/v2/config.py` | Dataclass-конфиги и загрузка JSON-конфига |
| `bio_music_pipeline/v2/bio.py` | Биологический encoder с sequence/ORF/protein/RNA features |
| `bio_music_pipeline/v2/dataset.py` | Полифонический tokenizer, сегментация, dataset и fallback corpus |
| `bio_music_pipeline/v2/pairing.py` | Pairing по многомерным дескрипторам и calibration |
| `bio_music_pipeline/v2/model.py` | `ControlConditionedTransformer` |
| `bio_music_pipeline/v2/train.py` | Основной train orchestration |
| `bio_music_pipeline/v2/generate.py` | Независимый inference orchestration |

### 2.2 Legacy-стек `bio_music_pipeline/*`

Ниже перечисленные директории всё ещё присутствуют и важны для истории проекта, но больше не являются основным рекомендованным способом запуска:

| Путь | Назначение |
|---|---|
| `bio_music_pipeline/extractors/` | Legacy extraction layer |
| `bio_music_pipeline/sonification/` | Legacy deterministic sonification |
| `bio_music_pipeline/data/` | Legacy MIDI dataset и old paired creator |
| `bio_music_pipeline/models/` | Legacy conditioned transformer |
| `bio_music_pipeline/baselines/` | Baseline-генераторы старого контура |
| `bio_music_pipeline/evaluation/` | Legacy evaluation framework |
| `bio_music_pipeline/utils/` | Общие утилиты старого стека |

## 3. Конфиги `configs/`

| Путь | Назначение |
|---|---|
| `configs/pipeline_v2_small.json` | Основной рабочий конфиг `v2` для локального устройства с `RTX 2060 6 GB` |
| `configs/pipeline_config.json` | Legacy базовый конфиг |
| `configs/pipeline_full_paired.json` | Legacy полный paired-режим |
| `configs/pipeline_quick_paired.json` | Legacy быстрый paired-режим |
| `configs/pipeline_quick_paired_v2.json` | Legacy сверхбыстрый paired-режим |
| `configs/data_paths_config.json` | Дополнительная конфигурация путей |
| `configs/sample_data_paths.json` | Шаблон конфигурации пользовательских директорий |

## 4. Данные `data/`

### 4.1 `data/fasta/`

| Путь | Назначение |
|---|---|
| `data/fasta/README.txt` | Инструкция по FASTA |
| `data/fasta/quick_sample.fa` | Демонстрационный FASTA для smoke- и quick-run |

`v2` режет FASTA на фрагменты автоматически, поэтому даже небольшой demo FASTA может быть использован для тестового запуска.

### 4.2 `data/midi/`

| Путь | Назначение |
|---|---|
| `data/midi/README.txt` | Инструкция по MIDI |
| `data/midi/monophonic_extracted/` | Legacy монофонический набор |
| `data/midi/polyphonic_music21/` | Локально экспортируемый fallback-корпус полифонических MIDI через `music21` |

Каталог `data/midi/polyphonic_music21/`:

- создаётся автоматически при первом запуске `v2`, если внешний полифонический корпус не задан;
- используется как fallback для быстрого локального обучения и smoke-проверок.

## 5. Тесты

| Путь | Назначение |
|---|---|
| `tests/test_v2_pipeline.py` | Smoke tests нового пайплайна |

## 6. Результаты `results/`

### 6.1 Актуальный `v2` запуск

Основной каталог:

- `results/v2_music21_rtx2060/`

Ключевые артефакты:

| Путь | Назначение |
|---|---|
| `results/v2_music21_rtx2060/resolved_config.json` | Зафиксированный конфиг запуска |
| `results/v2_music21_rtx2060/metrics.json` | История train/val и test metrics |
| `results/v2_music21_rtx2060/checkpoints/best_model.pt` | Лучший checkpoint |
| `results/v2_music21_rtx2060/pairing/pair_manifest.json` | Манифест пар bio↔music |
| `results/v2_music21_rtx2060/pairing/pair_calibration.npz` | Калибровка bio profile к music distribution |
| `results/v2_music21_rtx2060/smoke/sample_from_training_pipeline.mid` | Smoke-generation после обучения |

### 6.2 Отдельный inference output

| Путь | Назначение |
|---|---|
| `results/v2_generation/generated_from_fasta.mid` | Результат генерации из FASTA |
| `results/v2_generation/generated_from_fasta.json` | Метаданные генерации |

### 6.3 Legacy-результаты

Любые каталоги вида:

- `results/full_paired_run/`
- `results/paired_data/`
- `results/research_artifacts/`

относятся к прежнему экспериментальному контуру и не являются основным результатом `v2`.

## 7. Web-часть

| Путь | Назначение |
|---|---|
| `web/app.py` | Legacy Flask entrypoint |
| `web/generator.py` | Legacy web generator |
| `web/midi_to_audio.py` | Конвертация MIDI в WAV для web-сценариев |
| `web/templates/` | HTML-шаблоны |
| `web/static/` | CSS/JS |

Web-слой пока не переподключён на `v2` и поэтому должен восприниматься как legacy-интерфейс поверх старого inference-контура.
