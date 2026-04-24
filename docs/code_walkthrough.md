# Подробный Разбор Кода

Этот документ описывает **актуальную инженерную структуру проекта после перехода на `v2` пайплайн**. Старый стек по-прежнему присутствует в репозитории, но теперь он рассматривается как legacy-контур, а не как основной путь запуска.

## 1. Карта входных точек

### `train_bio_music_v2.py`

Главная точка запуска обучения.

Что делает:

1. читает `configs/pipeline_v2_small.json`;
2. вызывает `bio_music_pipeline.v2.train.train_pipeline()`;
3. печатает JSON со ссылками на итоговые артефакты.

### `generate_from_fasta_v2.py`

Главная точка инференса.

Что делает:

1. загружает обученный checkpoint;
2. кодирует FASTA;
3. применяет train-time calibration;
4. генерирует токены;
5. сохраняет MIDI и JSON-метаданные.

### `run_pipeline.py`

Старый orchestration entrypoint.

Он всё ещё в репозитории, но описывает прежнюю постановку с:

- legacy tokenizer;
- old paired creator;
- old conditioned transformer;
- старым evaluation loop.

Использовать его как основной путь запуска больше не рекомендуется.

## 2. Новый пакет `bio_music_pipeline.v2`

## 2.1 `bio_music_pipeline/v2/__init__.py`

Экспортирует:

- конфиги;
- bio encoder;
- dataset/tokenizer;
- pairing;
- модель;
- train/generate entrypoints.

Это новый компактный фасад всего актуального пайплайна.

## 2.2 `bio_music_pipeline/v2/config.py`

Главный слой конфигурации.

Содержит dataclass-конфиги:

- `BioEncoderConfig`
- `MusicDataConfig`
- `PairingConfig`
- `TrainingConfig`
- `GenerationConfig`
- `V2PipelineConfig`

Также содержит `load_v2_config()`, который:

- поднимает значения по умолчанию;
- читает JSON;
- рекурсивно мержит пользовательские значения с dataclass-дефолтами.

Ключевая идея: конфиг больше не размазан по старым модулям и не держится на неявных полях.

## 2.3 `bio_music_pipeline/v2/bio.py`

Новый biologically informed encoder.

Главные сущности:

- `BioEncodingResult`
- `BiologicalSequenceEncoder`

### Что делает `BiologicalSequenceEncoder`

1. читает FASTA через `Biopython`;
2. определяет тип последовательности: `dna / rna / protein`;
3. режет длинные записи на фрагменты;
4. очищает последовательность;
5. считает нуклеотидные и/или белковые признаки;
6. при наличии `ViennaRNA` считает secondary structure признаки;
7. переводит DNA в белок через longest ORF;
8. собирает фиксированный `bio vector`;
9. дополнительно собирает `control_profile`, пригодный для музыкального conditioning.

### Биологические блоки признаков

- nucleotide composition
- entropy
- GC/AT skew
- k-mer distribution
- ORF-derived protein features
- `ProtParam` features:
  - aromaticity
  - instability index
  - isoelectric point
  - gravy
  - molecular weight
  - flexibility
  - secondary structure fractions
- RNA folding descriptors:
  - MFE
  - paired fraction
  - loop fraction
  - structural entropy

Это принципиальное отличие от старого `bio_extractor.py`, который ограничивался простыми DNA statistics.

## 2.4 `bio_music_pipeline/v2/dataset.py`

Новый polyphonic symbolic music layer.

Главные сущности:

- `NoteEvent`
- `MusicSegment`
- `PolyphonicMusicTokenizer`
- `BioMusicPairDataset`
- `load_music_corpus()`
- `bootstrap_music21_corpus()`

### `PolyphonicMusicTokenizer`

Токенизатор использует event-based схему:

- `BOS`
- control tokens
- `SEP`
- `TIME_*`
- `NOTE_*`
- `DUR_*`
- `VEL_*`
- `EOS`

Ключевое отличие от старого токенизатора:

- он не пытается делать `NOTE_ON/NOTE_OFF` с грубой time clipping;
- он работает прямо с полифоническими onset-группами;
- он несёт control-prefix в самой последовательности.

### `load_music_corpus()`

Этот метод:

1. ищет MIDI/XML-корпус в `music.midi_dirs`;
2. если корпус отсутствует, поднимает fallback через `music21`;
3. парсит произведения;
4. сегментирует их по тактам;
5. фильтрует слишком бедные сегменты;
6. считает музыкальные дескрипторы;
7. возвращает список `MusicSegment`.

### Дескрипторы сегмента

Для каждого сегмента считаются:

- tempo
- density
- polyphony
- register
- harmony
- mode

Эти величины используются и для pairing, и как control signal для генератора.

## 2.5 `bio_music_pipeline/v2/pairing.py`

Новый pairing-слой.

Главные сущности:

- `PairedSample`
- `calibrate_bio_profiles()`
- `build_paired_dataset()`
- `save_pairing_artifacts()`

### Что изменилось относительно старого pairing

Старый контур делал rank-matching по одному scalar complexity score. Новый:

1. берёт многомерный `control_profile` из био-энкодера;
2. калибрует его под реальное распределение музыкального корпуса;
3. считает weighted distance до music descriptor vectors;
4. выбирает `top-k` ближайших сегментов;
5. создаёт many-to-many pairing с весами.

Это даёт модели существенно более осмысленное conditioning-пространство.

## 2.6 `bio_music_pipeline/v2/model.py`

Новая компактная модель `ControlConditionedTransformer`.

### Архитектура

1. token embedding
2. position embedding
3. `bio_projection` в несколько memory tokens
4. `TransformerDecoder`
5. `lm_head`
6. `descriptor_head`

### Как модель получает conditioning

Условие подаётся двумя путями:

- через discrete control tokens в префиксе последовательности;
- через continuous bio embedding как cross-attention memory.

### `compute_loss()`

Функция потерь состоит из:

- token cross-entropy
- descriptor alignment loss

То есть модель учится не только предсказывать токены, но и сохранять заданный музыкальный профиль.

### `generate()`

Метод генерации:

- стартует с `BOS + control tokens + SEP`;
- использует `top-k/top-p` sampling;
- запрещает `EOS` раньше `min_new_tokens`;
- корректно останавливается по `EOS`.

Это принципиально чище старого генератора, где stopping logic и serialization были источником ошибок.

## 2.7 `bio_music_pipeline/v2/train.py`

Главный orchestration-модуль обучения.

### `train_pipeline()`

Внутри он:

1. загружает конфиг;
2. фиксирует seed;
3. кодирует FASTA;
4. строит музыкальный корпус;
5. делает split отдельно по bio и music;
6. строит pairing для train/val/test;
7. создаёт DataLoader’ы;
8. создаёт `ControlConditionedTransformer`;
9. обучает модель;
10. сохраняет лучший checkpoint;
11. прогоняет test evaluation;
12. делает smoke generation.

### Что сохраняется

- `resolved_config.json`
- `pair_manifest.json`
- `pair_calibration.npz`
- `best_model.pt`
- `metrics.json`
- `sample_from_training_pipeline.mid`

## 2.8 `bio_music_pipeline/v2/generate.py`

Независимый inference-слой.

Главная функция:

- `generate_music_from_fasta()`

Что она делает:

1. загружает config и checkpoint;
2. создаёт encoder и tokenizer;
3. кодирует нужную FASTA-запись;
4. применяет calibration из checkpoint;
5. строит control tokens;
6. вызывает `model.generate()`;
7. пишет MIDI и JSON-метаданные.

Это важно архитектурно: генерация больше не зависит от train pipeline напрямую.

## 3. Тесты

### `tests/test_v2_pipeline.py`

Содержит три smoke-проверки:

1. `bio encoder` на demo FASTA
2. roundtrip tokenization
3. forward/generate для модели

Эти тесты не заменяют полноценный training run, но очень быстро ловят структурные поломки.

## 4. Конфиг по умолчанию

### `configs/pipeline_v2_small.json`

Это основной рабочий конфиг для текущего устройства.

Он:

- пишет в `results/v2_music21_rtx2060`
- использует fallback polyphonic corpus при отсутствии своего MIDI-корпуса
- ограничивает модель и batch-size под `RTX 2060 6 GB`
- включает mixed precision

## 5. Legacy-часть репозитория

Всё ниже сохранено, но больше не является основной архитектурой:

- `bio_music_pipeline/extractors/*`
- `bio_music_pipeline/data/*`
- `bio_music_pipeline/models/transformer.py`
- `run_pipeline.py`
- `generate_from_fasta.py`
- старый web backend

Эти файлы полезны:

- для исторического сравнения;
- для разбора исходных архитектурных решений;
- для объяснения, почему понадобился `v2`.
