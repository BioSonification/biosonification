# Подробный Разбор Кода

Этот документ описывает текущий рабочий код после перехода на structured `v2` пайплайн. Главная идея новой реализации: генерация больше не идёт через один общий поток токенов, а разделена на `harmony` и `melody`.

## 1. Точки входа

### `train_bio_music_v2.py`

Основной CLI для обучения. Скрипт:

1. читает `configs/pipeline_v2_small.json`
2. вызывает `bio_music_pipeline.v2.train_structured_pipeline()`
3. печатает JSON с путями к артефактам

### `generate_from_fasta_v2.py`

Основной CLI для инференса. Скрипт:

1. загружает `structured_pipeline.pt`
2. кодирует FASTA
3. генерирует гармонию
4. генерирует мелодию
5. сохраняет MIDI и JSON-метаданные

### Legacy entrypoints

- `run_pipeline.py`
- `generate_from_fasta.py`

Они оставлены в репозитории, но не являются рекомендуемым путём запуска.

## 2. Пакет `bio_music_pipeline/v2`

### `config.py`

Содержит все dataclass-конфиги:

- `BioEncoderConfig`
- `MusicDataConfig`
- `PairingConfig`
- `TrainingConfig`
- `GenerationConfig`
- `V2PipelineConfig`

Здесь задаются:

- размерности bio-вектора
- параметры сегментации музыки
- число эпох для `harmony` и `melody`
- длина генерации и sampling-параметры

### `bio.py`

Главный biological encoder.

Ключевые сущности:

- `BioEncodingResult`
- `BiologicalSequenceEncoder`

Что делает:

1. читает FASTA
2. определяет тип последовательности
3. очищает и фрагментирует запись
4. считает sequence features
5. при необходимости переводит в белок
6. извлекает protein features через `ProtParam`
7. извлекает RNA folding признаки через `ViennaRNA`
8. опционально добавляет `ESM` embedding block
9. возвращает `vector`, `control_profile`, `tonic_pc_hint`

### `structured_music.py`

Это ключевой модуль новой архитектуры.

Главные сущности:

- `HarmonyBar`
- `MelodyEvent`
- `StructuredMusicSegment`
- `HarmonyTokenizer`
- `MelodyTokenizer`

Что делает модуль:

1. загружает полифонический корпус
2. выбирает мелодическую партию
3. извлекает аккорды по тактам
4. извлекает мелодические события относительно текущего аккорда
5. токенизирует `harmony`
6. токенизирует `melody`
7. рендерит обе дорожки обратно в MIDI

Важный момент: `decode_melody()` нормализует выход так, чтобы мелодия:

- не выходила за границы гармонической сетки
- не наслаивалась сама на себя
- оставалась монофонической

### `structured_pairing.py`

Новый pairing-слой.

Главные сущности:

- `StructuredPairedSample`
- `calibrate_bio_profiles()`
- `build_structured_paired_dataset()`
- `save_structured_pairing_artifacts()`

Модуль сопоставляет bio fragments и structured music segments через многомерные дескрипторы и формирует веса пар.

### `structured_model.py`

Универсальная autoregressive модель:

- `BioConditionedSequenceModel`

Она используется и для `harmony`, и для `melody`.

Что внутри:

- token embedding
- positional embedding
- projection биовектора в conditioning memory
- causal Transformer stack
- LM head
- sampling с `top-k/top-p`

`compute_loss()` поддерживает `loss_mask`, поэтому префиксные control-токены и harmony-prefix можно не штрафовать как целевую часть последовательности.

### `structured_train.py`

Главный orchestration-модуль обучения.

Ключевая функция:

- `train_structured_pipeline()`

Что она делает:

1. загружает конфиг
2. кодирует FASTA
3. строит structured music corpus
4. делает раздельный split bio/music
5. собирает pairing
6. строит dataset и dataloader для `harmony`
7. обучает `harmony_model`
8. строит dataset и dataloader для `melody`
9. обучает `melody_model`
10. сохраняет checkpoints и `metrics.json`
11. делает smoke generation

### `structured_generate.py`

Главный inference-модуль.

Ключевая функция:

- `generate_structured_music_from_fasta()`

Что она делает:

1. загружает checkpoint
2. восстанавливает calibration
3. кодирует FASTA
4. генерирует гармонию
5. подаёт гармонию в мелодическую модель
6. рендерит двухдорожечный MIDI
7. пишет JSON с метаданными генерации

### Старые модули внутри `v2`

В каталоге `bio_music_pipeline/v2` всё ещё лежат более ранние single-stream модули:

- `dataset.py`
- `pairing.py`
- `model.py`
- `train.py`
- `generate.py`

Сейчас они полезны скорее как промежуточная историческая стадия перехода от legacy-архитектуры к structured pipeline.

## 3. Тесты

### `tests/test_v2_pipeline.py`

Содержит четыре smoke-проверки:

1. `bio encoder` на `quick_sample.fa`
2. извлечение structured music и рендер обратно в score
3. forward/generate для `BioConditionedSequenceModel`
4. проверка, что `decode_melody()` удерживает мелодию внутри формы и без overlap

Это не заменяет full training run, но быстро ловит структурные поломки.

## 4. Артефакты после обучения

После `train_bio_music_v2.py` получаем:

- `resolved_config.json`
- `pair_manifest.json`
- `pair_calibration.npz`
- `harmony_best.pt`
- `melody_best.pt`
- `structured_pipeline.pt`
- `metrics.json`
- `structured_sample.mid`

После `generate_from_fasta_v2.py` получаем:

- `structured_from_fasta.mid`
- `structured_from_fasta.json`
