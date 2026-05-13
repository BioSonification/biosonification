# Разбор Кода

Этот документ описывает актуальный код после очистки репозитория. Главная реализация находится в structured `v2`: генерация разделена на `Bio -> Harmony` и `Bio + Harmony -> Melody`.

## Точки Входа

### `train_bio_music_v2.py`

CLI для обучения. Скрипт читает конфиг, вызывает `bio_music_pipeline.v2.train_structured_pipeline()` и печатает JSON с путями к артефактам.

### `generate_from_fasta_v2.py`

CLI для инференса. Скрипт загружает `structured_pipeline.pt`, кодирует FASTA, генерирует гармонию, затем мелодию, сохраняет MIDI и JSON-метаданные.

## Пакет `bio_music_pipeline/v2`

### `config.py`

Содержит dataclass-конфиги:

- `BioEncoderConfig`
- `MusicDataConfig`
- `PairingConfig`
- `TrainingConfig`
- `GenerationConfig`
- `V2PipelineConfig`

Модуль умеет читать JSON-конфиг с диска и восстанавливать конфиг из checkpoint metadata.

### `bio.py`

Biological encoder. Он читает FASTA, определяет тип последовательности, очищает и фрагментирует записи, считает nucleotide/protein/RNA признаки и возвращает:

- `vector`
- `control_profile`
- `tonic_pc_hint`

### `corpus.py`

Небольшой слой работы с музыкальным корпусом:

- рекурсивно находит поддерживаемые score-файлы;
- при необходимости создаёт fallback-корпус из `music21`.

### `structured_music.py`

Ключевой музыкальный модуль. Он извлекает из score:

- `HarmonyBar`
- `MelodyEvent`
- `StructuredMusicSegment`

Здесь же находятся `HarmonyTokenizer`, `MelodyTokenizer` и renderer двухдорожечного MIDI. `decode_melody()` удерживает мелодию в пределах гармонической сетки, схлопывает события с одинаковым onset и не допускает самоналожений.

### `structured_pairing.py`

Сопоставляет bio fragments и structured music segments через многомерные дескрипторы:

- tempo;
- note density;
- harmonic change rate;
- register;
- harmony complexity;
- mode.

Pairing использует train-time calibration и many-to-many top-k matching.

### `structured_model.py`

Содержит `BioConditionedSequenceModel`: autoregressive Transformer с conditioning по bio vector. Один и тот же класс используется для harmony и melody моделей.

### `structured_train.py`

Orchestration обучения:

1. загружает конфиг;
2. кодирует FASTA;
3. строит structured music corpus;
4. делает split bio/music;
5. собирает pairing;
6. обучает harmony model;
7. обучает melody model;
8. сохраняет checkpoints, metrics и smoke MIDI.

### `structured_generate.py`

Inference:

1. загружает checkpoint;
2. восстанавливает конфиг и calibration;
3. кодирует FASTA;
4. генерирует harmony;
5. использует harmony как prefix для melody;
6. рендерит MIDI;
7. пишет metadata.

### `evaluate.py`

Считает технические метрики generated MIDI, строит random harmony+melody baseline и формирует JSON/Markdown evaluation report.

### `dataset_report.py`

Фиксирует фактические входные данные: FASTA records/fragments, MIDI count, source kind, segment count и сводки по tempo/key/density.

## Тесты

`tests/test_v2_pipeline.py` проверяет:

- bio encoder;
- извлечение structured music и render обратно в score;
- forward/generate модели;
- ограничения `decode_melody()`;
- CLI generation;
- evaluation helpers;
- dataset report;
- состав stable exports.

Это не заменяет полноценное обучение, но быстро ловит структурные поломки после рефакторинга.

## Артефакты

После обучения ожидаются:

- `resolved_config.json`
- `pair_manifest.json`
- `pair_calibration.npz`
- `harmony_best.pt`
- `melody_best.pt`
- `structured_pipeline.pt`
- `metrics.json`
- `structured_sample.mid`

После генерации ожидаются:

- `structured_from_fasta.mid`
- `structured_from_fasta.json`
