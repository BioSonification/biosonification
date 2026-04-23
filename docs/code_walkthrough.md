# Подробный Разбор Кода

Этот документ разбирает проект как программную систему: какие файлы запускают пайплайн, какие классы и функции отвечают за отдельные стадии, как между ними передаются данные и в каком порядке работает весь стек.

## 1. Карта входных точек

### `run_pipeline.py`

Главная исполняемая точка проекта. Именно здесь живёт класс `BioMusicPipeline`, который оркестрирует 5 стадий:

1. extraction,
2. sonification,
3. dataset preparation,
4. training,
5. generation + evaluation.

Внутри `main()` парсятся CLI-аргументы:

- `--config`
- `--sequences`
- `--midi-dir`
- `--paired-data`
- `--allow-synthetic`

После этого создаётся объект `BioMusicPipeline` и вызывается `pipeline.run(...)`.

### `generate_from_fasta.py`

Отдельная утилита для одиночной генерации:

- берёт один FASTA-фрагмент,
- строит по нему биовектор,
- прогоняет через обученную модель,
- сохраняет один MIDI и метаинформацию.

Это полезно для sanity check, демонстраций и отдельных примеров в дипломе.

### `scan_datasets.py`

Сервисный скрипт:

- умеет создать стандартные каталоги `data/fasta` и `data/midi`,
- сканирует репозиторий,
- показывает, какие FASTA и MIDI будут подхвачены пайплайном.

### `web/app.py`

Точка входа для Flask-приложения. Поднимает UI для генерации музыки из FASTA без повторного обучения модели.

### `tools/run_multi_seed_experiments.py`

Оркестрирует серию запусков пайплайна на разных seed и агрегирует метрики.

### `tools/generate_research_artifacts.py`

Собирает `final_report.json` из разных запусков и превращает их в:

- summary CSV/JSON,
- графики,
- markdown-отчёт.

## 2. `run_pipeline.py`: детальный разбор

### 2.1 `BioMusicPipeline.__init__`

Что делает:

1. читает JSON-конфиг;
2. фиксирует seed через `set_seed()`;
3. создаёт `output_dir`;
4. инициализирует контейнер `self.results`.

`self.results` потом становится основой для `final_report.json`.

### 2.2 `stage1_extract_bio_vectors()`

Роль:

- загрузить FASTA-последовательности,
- при необходимости синтезировать демо-данные,
- извлечь `bio_vectors.npy`,
- сохранить `sequence_ids.json`.

Что важно в логике:

- если вход не передан явно, метод сначала ищет данные в `data/fasta`;
- synthetic fallback разрешается только при `allow_synthetic=True`;
- векторизация делается по одной последовательности, но с прогресс-баром `tqdm`.

### 2.3 `stage2_apply_sonification()`

Роль:

- создать `SonificationMapper`,
- при необходимости откалибровать его на всём массиве биовекторов,
- получить список `MusicalParameters`,
- собрать `conditioning_vectors.npy`,
- сохранить пример параметров в `musical_params_sample.json`.

Что метод добавляет в отчёт:

- размерность conditioning-вектора,
- распределение по ключам,
- реальный диапазон темпов,
- summary по калибровке.

### 2.4 `_compute_key_distribution()`

Вспомогательная функция для сводки stage 2. Считает частоты ключей по списку `MusicalParameters`.

### 2.5 `stage3_prepare_dataset()`

Роль:

- найти MIDI-каталог;
- при необходимости построить synthetic MIDI;
- создать `MusicDataset`;
- сохранить `train/val/test` списки;
- проверить отсутствие data leakage.

Особенность:

даже если каталог существует, но в нём нет `.mid/.midi`, код умеет построить synthetic fallback, если это явно разрешено.

### 2.6 `_create_synthetic_midi_data()`

Строит набор искусственных MIDI:

- случайный темп,
- случайные ноты по мажорной гамме,
- случайные длительности и velocity.

Это не научный режим, а аварийно-демонстрационный fallback.

### 2.7 `stage4_train_model()`

Непарный режим обучения.

Логика:

1. синхронизирует `vocab_size` и специальные токены модели с `MIDIPreprocessor`;
2. создаёт модель;
3. проверяет gradient flow;
4. запускает основной training loop;
5. для каждого batch-а случайно подмешивает bio-векторы;
6. валидирует модель;
7. сохраняет лучший checkpoint.

Этот путь присутствует для универсальности, но в исследовательском сценарии уступает paired-обучению.

### 2.8 `stage4_train_model_with_paired_data()`

Главный режим stage 4.

Отличие от непарного режима:

- batch уже содержит собственные `bio_vector`;
- conditioning соответствует конкретному MIDI;
- в checkpoint добавляется флаг `paired_training=True`.

Всё остальное:

- optimizer,
- gradient clipping,
- temperature update,
- early stopping

организовано аналогично базовому training loop.

### 2.9 `_train_unconditional_baseline()`

Мини-обучение baseline-модели `UnconditionalTransformer`:

- использует тот же токенизированный train-набор;
- обучается до `min(10, epochs)` эпох;
- служит baseline для H1.

### 2.10 `stage5_generate_and_evaluate()`

Один из самых насыщенных методов всего проекта.

По шагам:

1. подгружает лучший checkpoint;
2. строит baseline-генераторы;
3. обучает `MarkovBaseline` на train-последовательностях;
4. обучает `UnconditionalTransformer`;
5. балансированно выбирает subset биовекторов по кластерам;
6. генерирует 6 наборов последовательностей;
7. конвертирует их в MIDI;
8. запускает `run_comprehensive_evaluation()`;
9. создаёт визуализации;
10. генерирует HTML-анкету для human evaluation.

Именно этот метод превращает просто обученную модель в полноценный исследовательский прогон с артефактами.

### 2.11 `_sample_bio_indices_balanced()`

Вспомогательная функция для sampling-а conditioned биовекторов. Старается обеспечить приблизительный баланс по cluster labels, чтобы H2 не работала на перекошенной выборке.

### 2.12 `_compare_baselines()`

Очень простой агрегатор для `stage5`, считающий:

- среднее значение токенов,
- стандартное отклонение,
- среднее число уникальных токенов на последовательность.

Это не основной научный метрик-блок, а быстрая табличная сводка.

### 2.13 `save_final_report()`

Собирает:

- `final_report.json`
- `summary.txt`

Также на основе `hypothesis_tests` формирует `hypothesis_status` в человекочитаемом виде.

### 2.14 `run()`

Главный end-to-end сценарий. Последовательно запускает стадии 1–5, выбирая paired-обучение, если путь `paired_data_dir` существует.

### 2.15 `main()`

CLI-вход, который:

- валидирует пути,
- при необходимости сам читает FASTA из `--sequences`,
- вызывает `pipeline.run()`,
- завершает процесс кодом `0` или `1`.

## 3. Пакет `bio_music_pipeline`

## 3.1 `bio_music_pipeline/__init__.py`

Содержит:

- метаданные пакета;
- `set_seed()`.

`set_seed()` фиксирует:

- `random`,
- `numpy`,
- `torch`,
- `torch.cuda`,
- детерминированность `cuDNN`.

Это фундамент воспроизводимости проекта.

## 3.2 Папка `extractors/`

### `fasta_loader.py`

Главные сущности:

- `FastaSequence` — dataclass для одной FASTA-записи;
- `FastaDatasetLoader` — универсальный загрузчик FASTA;
- `load_user_fasta_dataset()` — удобная обёртка.

Ключевая логика:

- чтение FASTA из файла;
- поиск FASTA-файлов по директории;
- загрузка из одной или нескольких директорий;
- очистка последовательностей;
- сводная статистика по длинам и GC content.

### `bio_extractor.py`

Главная сущность: `BioVectorExtractor`.

Методы:

- `read_fasta()`
- `compute_nucleotide_frequencies()`
- `compute_shannon_entropy()`
- `compute_kmer_distribution()`
- `compute_gc_skew()`
- `compute_at_skew()`
- `compute_windowed_statistics()`
- `extract_features()`
- `create_bio_vector()`
- `process_file()`

Отдельно экспортируется `extract_bio_vectors_from_sequences()` как функциональная обёртка.

Этот файл — ядро биологической части проекта.

## 3.3 Папка `sonification/`

### `mapper.py`

Главные сущности:

- `MusicalParameters` — dataclass музыкальных параметров;
- `SonificationMapper` — маппер bio → music;
- `apply_sonification_rules()` — пакетная обёртка.

Ключевые методы класса:

- `fit_calibration()`
- `get_calibration_summary()`
- `_generate_chord_distributions()`
- `map_nucleotide_frequencies_to_key()`
- `map_entropy_to_tempo()`
- `map_skew_to_pitch_range()`
- `map_kmer_diversity_to_rhythm_complexity()`
- `map_gc_content_to_scale_type()`
- `map_windowed_stats_to_articulation()`
- `map_features_to_chord_distribution()`
- `bio_vector_to_musical_params()`
- `create_conditioning_vector()`

Это ключевой модуль интерпретируемого conditioning.

## 3.4 Папка `data/`

### `dataset.py`

Содержит два основных класса:

- `MIDIPreprocessor`
- `MusicDataset`

`MIDIPreprocessor` отвечает за:

- vocabulary,
- чтение MIDI,
- перевод MIDI в события,
- перевод событий в токены,
- перевод токенов в id,
- фильтрацию по длительности.

`MusicDataset` отвечает за:

- вызов `MIDIPreprocessor` по каталогу,
- random split,
- генераторы batch-ей,
- сохранение файловых списков split-ов.

### `paired_dataset_creator.py`

Содержит:

- `MIDIFeatures`
- `PairedSample`
- `PairedDatasetCreator`
- CLI `main()`

Главные методы:

- `extract_midi_features()`
- `_compute_pitch_entropy()`
- `_compute_velocity_entropy()`
- `scan_midi_files()`
- `load_and_extract_bio_vectors()`
- `create_pairs()`
- `save_paired_dataset()`
- `run()`

Это ключ к превращению проекта из «демо with bio-conditioning» в более формальный исследовательский стенд.

### `paired_dataset.py`

Главная сущность: `PairedMusicDataset`, наследник `MusicDataset`.

Его задача:

- прочитать paired-артефакты,
- заново токенизировать MIDI,
- сформировать train/val/test уже по paired-записям,
- отдавать batch-и, где есть `token_ids`, `bio_vector`, `conditioning_vector`, `metadata`.

### `universal_loader.py`

Вспомогательный инфраструктурный модуль для пользовательских датасетов.

Главные сущности:

- `DataLoaderConfig`
- `UniversalDataLoader`
- `setup_user_datasets()`

Он отвечает не за научную логику, а за удобство работы с разными директориями данных.

## 3.5 Папка `models/`

### `transformer.py`

Главные сущности:

- `PositionalEncoding`
- `GumbelSoftmaxSampler`
- `BioConditioningModule`
- `AuxiliaryLanguageModel`
- `BioConditionedTransformerDecoder`
- `create_bio_music_model()`

По сути весь нейросетевой стек проекта находится именно здесь.

Ключевые методы `BioConditionedTransformerDecoder`:

- `_init_weights()`
- `generate_square_subsequent_mask()`
- `encode_tokens()`
- `forward()`
- `compute_loss()`
- `generate()`
- `update_temperature()`

Если читать код как архитектурное описание, это главный файл проекта.

## 3.6 Папка `baselines/`

### `generators.py`

Содержит:

- `_extract_token_groups()`
- `RandomBaseline`
- `MarkovBaseline`
- `UnconditionalTransformer`
- `RuleBasedGenerator`
- `RandomVectorControl`
- `create_baselines()`

Роли baseline-ов:

- `RandomBaseline` — случайная нижняя граница;
- `MarkovBaseline` — статистическая n-gram модель;
- `UnconditionalTransformer` — честный нейросетевой baseline без conditioning;
- `RuleBasedGenerator` — «процедурная музыка» напрямую по musical params;
- `RandomVectorControl` — проверка, несут ли реальные биовекторы что-то сверх случайных векторов.

## 3.7 Папка `evaluation/`

### `validator.py`

Самый важный файл оценочного контура.

Содержит:

- `StatisticalValidator`
- `InformationTransferClassifier`
- `HumanEvaluationSurvey`
- `run_comprehensive_evaluation()`
- `compute_sequence_quality()`
- `compute_musical_quality_scores()`

Логически этот файл отвечает за формальный ответ на вопрос: «работает ли conditioning статистически?»

### `musical_quality.py`

Содержит `MusicalQualityMetrics` и `compare_musical_quality()`.

Это слой предметно-музыкальных метрик.

### `diversity.py`

Содержит `DiversityAnalyzer`.

Это слой анализа разнообразия:

- попарные расстояния,
- novelty,
- vocabulary coverage,
- intra/inter variance.

### `ablation.py`

Содержит:

- схемы индексов биовектора,
- функции абляции,
- `run_ablation_study()`,
- статистику по последовательностям после абляции.

### `visualizations.py`

Содержит:

- `visualize_bio_vectors()`
- `plot_correlation_heatmap()`
- `render_piano_roll()`
- `visualize_attention()`
- `create_all_visualizations()`

Это мост между кодом и визуальным исследовательским материалом.

### `perplexity_metrics.py`

Содержит:

- `ShannonEntropyMetrics`
- `compute_model_perplexity()`
- `compute_complexity_for_maestro()`

Это дополнительный оценочный блок для сложности музыкального материала.

### `idyom_integration.py`

Содержит:

- `IDyOMWrapper`
- `compare_idyom_vs_shannon()`
- `plot_idyom_vs_shannon()`

Это адаптер между основным проектом и встроенным `IDyOMpy`.

### `__init__.py`

Экспортирует наружу ключевые оценочные инструменты. Это удобная точка импорта для `run_pipeline.py`.

## 3.8 Папка `utils/`

### `helpers.py`

Содержит:

- `tokens_to_midi()`
- `batch_tokens_to_midi()`
- `check_gradient_flow()`
- `verify_no_data_leak()`
- `create_sample_bio_sequences()`
- `GradientCheckpoint`

Это вспомогательная, но очень важная часть пайплайна: без неё не было бы ни отладки обучения, ни сохранения генераций в MIDI.

## 4. Веб-слой

## 4.1 `web/app.py`

Flask-приложение.

Маршруты:

- `/`
- `/api/generate`
- `/api/download/<session_id>/<file_type>`
- `/api/status`
- `/api/survey/submit`
- `/api/survey/results`

Что важно:

- backend не обучает модель, а только выполняет inference;
- модель инициализируется один раз и переиспользуется;
- есть обработка ошибок размера файла и внутренних исключений.

## 4.2 `web/generator.py`

Главный backend-класс: `BioMusicGenerator`.

Его обязанности:

- найти конфиг и checkpoint;
- загрузить модель;
- поднять `BioVectorExtractor`;
- поднять `SonificationMapper`;
- поднять `MIDIPreprocessor`;
- провалидировать FASTA;
- выполнить generation;
- сохранить MIDI.

Ключевые методы:

- `_resolve_default_config_path()`
- `_resolve_default_model_path()`
- `initialize()`
- `validate_fasta()`
- `generate()`
- `is_ready()`
- `get_error()`

Также внизу реализован singleton `get_generator()`.

## 4.3 `web/midi_to_audio.py`

Модуль-заглушка:

- `midi_to_wav()` сейчас всегда возвращает `False`;
- `check_audio_synthesizer()` сообщает, что аудио disabled;
- `get_install_instructions()` объясняет, почему.

Это важно описывать честно: текущая web-версия отдаёт MIDI, но не полноценный WAV.

## 4.4 `web/templates/index.html`

Основная HTML-страница:

- вкладки paste/upload,
- поле FASTA,
- drag-and-drop файл,
- блок loading,
- блок error,
- блок results,
- карточки musical parameters,
- кнопка download MIDI.

## 4.5 `web/static/js/app.js`

Frontend-логика:

- переключение вкладок,
- обработка drag-and-drop,
- вызов `/api/generate`,
- показ ошибок,
- показ результатов,
- keyboard shortcut `Ctrl/Cmd + Enter`,
- проверка статуса сервера.

## 4.6 `web/static/css/style.css`

Оформление интерфейса.

По коду видно, что это законченный визуальный слой с:

- CSS-переменными,
- карточками,
- табами,
- блоками загрузки,
- стилями кнопок,
- секцией результатов.

## 5. Исследовательские утилиты и сервисные скрипты

### `tools/run_multi_seed_experiments.py`

Отвечает за:

- парсинг списка seed;
- генерацию seed-specific config;
- вызов `run_pipeline.py` через `subprocess`;
- сбор per-seed summary;
- вычисление aggregate metrics и confidence intervals.

### `tools/generate_research_artifacts.py`

Отвечает за:

- поиск всех `final_report.json`;
- извлечение стандартизированных строк метрик;
- запись summary CSV/JSON;
- генерацию графиков:
  - diversity,
  - p-value overview,
  - conditioning gaps;
- сборку `research_artifacts.md`.

### `tools/clean_pipeline_artifacts.sh`

Безопасная очистка результатов.

По умолчанию это dry-run, реальное удаление происходит только при `--apply`.

## 6. Встроенный внешний пакет `tools/IDyOMpy`

Этот каталог не написан в рамках основного пайплайна с нуля, а включён в репозиторий как отдельный внешний инструмент.

Ключевые части:

- `App.py` — CLI-вход IDyOMpy;
- `idyom/idyom.py` — высокая логика модели;
- `idyom/data.py` — парсинг MIDI и viewpoint representation;
- `idyom/longTermModel.py` — long-term model;
- `idyom/markovChain.py` — марковские цепи порядка `n`;
- `idyom/markovChainOrder0.py` — модель нулевого порядка;
- `idyom/myMidi.py` — чтение монофонического MIDI;
- `unittests/*.py` — unit tests upstream-проекта.

В основном проекте этот пакет не переписывается, а оборачивается через `bio_music_pipeline/evaluation/idyom_integration.py`.

## 7. Как данные протекают через код

Сжатая цепочка:

1. `run_pipeline.py`
2. `FastaDatasetLoader` / `BioVectorExtractor`
3. `SonificationMapper`
4. `MusicDataset` или `PairedMusicDataset`
5. `BioConditionedTransformerDecoder`
6. `batch_tokens_to_midi()`
7. `run_comprehensive_evaluation()`
8. `create_all_visualizations()`
9. `HumanEvaluationSurvey`
10. `save_final_report()`

То есть у проекта очень чёткая структура: каждый модуль отвечает за один слой, и только `run_pipeline.py` знает обо всей цепочке целиком.

## 8. Как читать код, если нужен именно дипломный разбор

Лучший порядок чтения исходников:

1. `run_pipeline.py`
2. `bio_music_pipeline/extractors/bio_extractor.py`
3. `bio_music_pipeline/sonification/mapper.py`
4. `bio_music_pipeline/data/dataset.py`
5. `bio_music_pipeline/data/paired_dataset_creator.py`
6. `bio_music_pipeline/models/transformer.py`
7. `bio_music_pipeline/baselines/generators.py`
8. `bio_music_pipeline/evaluation/validator.py`
9. `bio_music_pipeline/evaluation/musical_quality.py`
10. `web/generator.py`

Такой порядок соответствует и логике пайплайна, и логике академического объяснения.
