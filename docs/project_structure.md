# Подробный Каталог Файлов И Каталогов

Этот документ служит файловой картой проекта. Он описывает, что находится в каждой части репозитория и за что отвечает каждый важный файл.

## Принцип описания

В репозитории присутствуют два принципиально разных типа содержимого:

1. **авторский код, конфиги и документация**, которые имеет смысл описывать пофайлово;
2. **массовые однотипные данные и артефакты** — сотни и тысячи MIDI, generated outputs, upstream datasets и HTML-ассеты.

Для первой группы здесь используется индивидуальное описание файлов. Для второй группы описание дано по именующим шаблонам, количеству файлов и роли в эксперименте, потому что буквальное перечисление тысяч однотипных raw-артефактов не улучшает понимание архитектуры.

## 1. Корень репозитория

| Путь | Тип | Назначение |
|---|---|---|
| `.gitignore` | служебный | Правила игнорирования Git |
| `README.md` | документация | Главная обзорная энциклопедия проекта |
| `RUN_FROM_SCRATCH.md` | документация | Пошаговый запуск проекта с нуля |
| `SCIENTIFIC_ARTICLE_AND_TALK_BASE.md` | документация | Готовая текстовая база для статьи и выступления |
| `DATASETS_GUIDE.md` | документация | Руководство по работе с пользовательскими FASTA/MIDI данными |
| `CHANGES.md` | документация | Журнал ключевых исправлений и расширений проекта |
| `QWEN.md` | документация/контекст | Контекстный обзор проекта, частично дублирующий архитектурные идеи и команды |
| `requirements.txt` | зависимости | Python-зависимости проекта |
| `run_pipeline.py` | исполняемый код | Главный оркестратор всех 5 стадий |
| `generate_from_fasta.py` | исполняемый код | Одиночная генерация одного MIDI из FASTA-фрагмента |
| `scan_datasets.py` | исполняемый код | Сканирование и подготовка каталогов данных |
| `pipeline_output.log` | лог | Локальный лог предыдущих запусков |
| `.DS_Store` | системный файл macOS | Не относится к логике проекта |

## 2. Пакет `bio_music_pipeline/`

### 2.1 Верхний уровень пакета

| Путь | Назначение |
|---|---|
| `bio_music_pipeline/__init__.py` | Инициализация пакета и функция `set_seed()` |

### 2.2 `bio_music_pipeline/extractors/`

| Путь | Назначение |
|---|---|
| `bio_music_pipeline/extractors/__init__.py` | Экспорт `BioVectorExtractor`, `FastaDatasetLoader`, `FastaSequence`, `load_user_fasta_dataset` |
| `bio_music_pipeline/extractors/bio_extractor.py` | Извлечение признаков из последовательности и сборка биовектора |
| `bio_music_pipeline/extractors/fasta_loader.py` | Чтение FASTA, загрузка из директорий, статистика по последовательностям |

### 2.3 `bio_music_pipeline/sonification/`

| Путь | Назначение |
|---|---|
| `bio_music_pipeline/sonification/__init__.py` | Экспорт mapper-а и dataclass музыкальных параметров |
| `bio_music_pipeline/sonification/mapper.py` | Детерминированное отображение биовектора в musical parameters и conditioning vector |

### 2.4 `bio_music_pipeline/data/`

| Путь | Назначение |
|---|---|
| `bio_music_pipeline/data/__init__.py` | Экспорт всех data-компонент |
| `bio_music_pipeline/data/dataset.py` | `MIDIPreprocessor`, `MusicDataset`, сохранение split-ов |
| `bio_music_pipeline/data/paired_dataset_creator.py` | Создание MIDI↔bio paired-набора по complexity matching |
| `bio_music_pipeline/data/paired_dataset.py` | `PairedMusicDataset` для корректного парного обучения |
| `bio_music_pipeline/data/universal_loader.py` | Универсальный поиск пользовательских FASTA/MIDI директорий |

### 2.5 `bio_music_pipeline/models/`

| Путь | Назначение |
|---|---|
| `bio_music_pipeline/models/__init__.py` | Экспорт модели и связанных компонентов |
| `bio_music_pipeline/models/transformer.py` | Conditioned Transformer, auxiliary LM, Gumbel-Softmax, фабрика модели |

### 2.6 `bio_music_pipeline/baselines/`

| Путь | Назначение |
|---|---|
| `bio_music_pipeline/baselines/__init__.py` | Экспорт baseline-генераторов |
| `bio_music_pipeline/baselines/generators.py` | `RandomBaseline`, `MarkovBaseline`, `UnconditionalTransformer`, `RuleBasedGenerator`, `RandomVectorControl` |

### 2.7 `bio_music_pipeline/evaluation/`

| Путь | Назначение |
|---|---|
| `bio_music_pipeline/evaluation/__init__.py` | Экспорт статистики, метрик, визуализаций, IDyOM и абляции |
| `bio_music_pipeline/evaluation/validator.py` | H1/H2, bootstrap, permutation test, survey, evaluation runner |
| `bio_music_pipeline/evaluation/musical_quality.py` | Музыкальные метрики качества |
| `bio_music_pipeline/evaluation/diversity.py` | Анализ разнообразия, novelty и vocabulary coverage |
| `bio_music_pipeline/evaluation/ablation.py` | Абляции по блокам биовектора |
| `bio_music_pipeline/evaluation/visualizations.py` | PCA, t-SNE, correlation heatmap, piano roll, saliency |
| `bio_music_pipeline/evaluation/perplexity_metrics.py` | Энтропийные метрики сложности и model perplexity |
| `bio_music_pipeline/evaluation/idyom_integration.py` | Адаптер к внешнему `IDyOMpy` |

### 2.8 `bio_music_pipeline/utils/`

| Путь | Назначение |
|---|---|
| `bio_music_pipeline/utils/__init__.py` | Экспорт утилит |
| `bio_music_pipeline/utils/helpers.py` | Конвертация токенов в MIDI, gradient check, anti-leak check, synthetic DNA |

## 3. Конфиги `configs/`

| Путь | Назначение |
|---|---|
| `configs/pipeline_config.json` | Базовый конфиг пайплайна |
| `configs/pipeline_full_paired.json` | Полный paired-режим для полноценного научного прогона |
| `configs/pipeline_quick_paired.json` | Укороченный paired-режим для быстрых проверок |
| `configs/pipeline_quick_paired_v2.json` | Ещё более быстрый paired-режим |
| `configs/data_paths_config.json` | Пример явной конфигурации путей к данным |
| `configs/sample_data_paths.json` | Шаблон расширенной конфигурации пользовательских директорий |

### Что реально регулирует каждый конфиг

#### `pipeline_config.json`

Содержит основные разделы:

- `pipeline`
- `extraction`
- `sonification`
- `data`
- `model`
- `training`
- `evaluation`

Это «общий» дефолтный конфиг.

#### `pipeline_full_paired.json`

Отличается тем, что:

- пишет результаты в `results/full_paired_run`;
- рассчитан на CPU;
- использует `max_sequences = 2000`;
- генерирует `60` сэмплов на режим;
- ставит `min_generation_length = 256`.

Именно он соответствует текущему основному полному прогону.

#### `pipeline_quick_paired*.json`

Это облегчённые режимы для smoke-test:

- меньше эпох,
- меньше batch,
- меньше сэмплов,
- меньше перестановок в статистике.

## 4. Данные `data/`

## 4.1 `data/fasta/`

| Путь | Назначение |
|---|---|
| `data/fasta/README.txt` | Инструкция по размещению пользовательских FASTA |
| `data/fasta/quick_sample.fa` | Маленький демонстрационный FASTA |
| `data/fasta/training/Homo_sapiens.GRCh38.cds.all.fa` | Основной FASTA-корпус для серьёзных запусков |
| `data/fasta/.DS_Store` | системный файл macOS |

Количество файлов в `data/fasta`: `4`.

## 4.2 `data/midi/`

| Путь | Назначение |
|---|---|
| `data/midi/README.txt` | Инструкция по размещению пользовательских MIDI |
| `data/midi/maestro-v3.0.0/README` | Краткая ссылка на описание MAESTRO |
| `data/midi/maestro-v3.0.0/LICENSE` | Лицензия внешнего датасета |
| `data/midi/maestro-v3.0.0/maestro-v3.0.0.csv` | Табличные метаданные MAESTRO |
| `data/midi/maestro-v3.0.0/maestro-v3.0.0.json` | JSON-метаданные MAESTRO |
| `data/midi/monophonic_extracted/*.mid*` | Набор монофонических MIDI для исследований и IDyOM-совместимых сценариев |
| `data/midi/maestro-v3.0.0/**/*.mid*` | Основной сырой корпус MAESTRO |
| `data/midi/.DS_Store` | системный файл macOS |
| `data/midi/maestro-v3.0.0/.DS_Store` | системный файл macOS |

Фактические количества:

| Группа | Количество файлов |
|---|---:|
| `data/midi/monophonic_extracted/*.mid*` | 100 |
| `data/midi/maestro-v3.0.0/**/*.mid*` | 1276 |

### Что означает каталог `monophonic_extracted/`

Это рабочий набор монофонических MIDI, который:

- удобен для paired-процедуры,
- потенциально совместим с IDyOM,
- полезен как более «чистый» материал по сравнению с полифоническим MAESTRO.

### Что означает `maestro-v3.0.0/`

Это внешний крупный датасет фортепианной символической музыки. Его MIDI-файлы не являются «кодом проекта», но они составляют значительную часть эмпирической базы обучения.

## 5. Результаты `results/`

Этот каталог содержит одновременно:

1. актуальные результаты полного запуска;
2. paired-артефакты;
3. quick-demo артефакты;
4. legacy-файлы от ранних запусков.

### 5.1 Верхний уровень `results/`

| Путь | Назначение |
|---|---|
| `results/bio_vectors.npy` | Legacy-артефакт раннего запуска |
| `results/conditioning_vectors.npy` | Legacy-артефакт раннего запуска |
| `results/musical_params_sample.json` | Legacy-сводка параметров |
| `results/sequence_ids.json` | Legacy-ID последовательностей |
| `results/fasta_generated_midi.mid` | Legacy-MIDI от ранней генерации |
| `results/idyom_vs_shannon_comparison.csv` | CSV сравнения IDyOM и Shannon |
| `results/.DS_Store` | системный файл macOS |

### 5.2 `results/paired_data/`

| Путь | Назначение |
|---|---|
| `results/paired_data/paired_bio_vectors.npy` | Матрица парных биовекторов |
| `results/paired_data/paired_conditioning_vectors.npy` | Матрица парных conditioning-векторов |
| `results/paired_data/paired_data.json` | Основной JSON-манифест пар |
| `results/paired_data/pairs_manifest.csv` | Компактная человекочитаемая таблица пар |
| `results/paired_data/paired_stats.json` | Сводная статистика по paired-набору |

### 5.3 `results/full_paired_run/`

Это главный завершённый прогон.

| Путь | Назначение |
|---|---|
| `results/full_paired_run/bio_vectors.npy` | Биовекторы stage 1 |
| `results/full_paired_run/conditioning_vectors.npy` | Conditioning-векторы stage 2 |
| `results/full_paired_run/sequence_ids.json` | ID последовательностей |
| `results/full_paired_run/musical_params_sample.json` | Пример музыкальных параметров |
| `results/full_paired_run/models/best_model.pt` | Лучший checkpoint модели |
| `results/full_paired_run/reports/evaluation_results.json` | Детальные результаты оценки |
| `results/full_paired_run/final_report.json` | Полный машинно-читаемый отчёт запуска |
| `results/full_paired_run/summary.txt` | Краткая текстовая сводка |
| `results/full_paired_run/surveys/human_evaluation_survey.html` | HTML-анкета для human evaluation |
| `results/full_paired_run/data_splits/train_files.txt` | Train split |
| `results/full_paired_run/data_splits/val_files.txt` | Validation split |
| `results/full_paired_run/data_splits/test_files.txt` | Test split |
| `results/full_paired_run/data_splits/metadata.json` | Метаданные split-ов |
| `results/full_paired_run/visualizations/visualizations_manifest.json` | Манифест визуализаций |
| `results/full_paired_run/visualizations/bio_vectors_pca.png` | PCA-проекция биовекторов |
| `results/full_paired_run/visualizations/bio_vectors_tsne.png` | t-SNE-проекция биовекторов |
| `results/full_paired_run/visualizations/piano_roll_<condition>_<i>.png` | Piano roll-рендеры для 6 режимов генерации |
| `results/full_paired_run/midi/<condition>/*.mid` | Сгенерированные MIDI по каждому режиму |
| `results/full_paired_run/.DS_Store` | системный файл macOS |
| `results/full_paired_run/midi/.DS_Store` | системный файл macOS |

#### Структура `results/full_paired_run/midi/`

Подкаталоги:

- `conditioned/`
- `unconditional/`
- `random/`
- `markov/`
- `rule_based/`
- `random_vector/`

Всего файлов: `360` MIDI.

#### Визуализации

Фактически на диске лежат:

- 2 глобальных plot-а (`PCA`, `t-SNE`);
- 30 piano roll файлов: по 5 на каждый из 6 режимов.

### 5.4 `results/research_artifacts/`

| Путь | Назначение |
|---|---|
| `results/research_artifacts/artifacts_summary.csv` | Сводная таблица артефактов по найденным отчётам |
| `results/research_artifacts/artifacts_summary.json` | То же в JSON |
| `results/research_artifacts/sonification_diversity.png` | График разнообразия сонификации |
| `results/research_artifacts/hypothesis_pvalues.png` | График p-value по гипотезам |
| `results/research_artifacts/conditioning_gaps.png` | График conditioning gaps |
| `results/research_artifacts/research_artifacts.md` | Markdown-отчёт по агрегированным результатам |

### 5.5 `results/multi_seed*`

В репозитории присутствуют несколько каталогов, связанных с multi-seed и quick demo сценариями:

| Путь | Назначение |
|---|---|
| `results/multi_seed/` | Каталог для полноценных multi-seed прогонов; сейчас содержит только generated config `pipeline_seed_7.json` |
| `results/multi_seed_quick_demo/` | Дополнительный demo-каталог результатов |
| `results/multi_seed_quick_demo_small/` | Малый demo multi-seed набор с aggregate metrics и одним seed-прогоном |

Для `results/multi_seed_quick_demo_small/` видны файлы:

- `aggregate_metrics.json`
- `per_seed_summary.csv`
- `per_seed_summary.json`
- `seed_7/*` — артефакты одного mini-прогона.

## 6. Инструменты `tools/`

### 6.1 Верхний уровень `tools/`

| Путь | Назначение |
|---|---|
| `tools/clean_pipeline_artifacts.sh` | Безопасная очистка результатов |
| `tools/run_multi_seed_experiments.py` | Серийный запуск пайплайна по seed |
| `tools/generate_research_artifacts.py` | Агрегация отчётов и построение научных артефактов |
| `tools/.DS_Store` | системный файл macOS |

### 6.2 `tools/IDyOMpy/`

Этот каталог является встроенной внешней библиотекой.

#### Основные кодовые файлы

| Путь | Назначение |
|---|---|
| `tools/IDyOMpy/README.md` | README внешнего проекта |
| `tools/IDyOMpy/App.py` | Главный CLI-вход IDyOMpy |
| `tools/IDyOMpy/LICENSE` | Лицензия IDyOMpy |
| `tools/IDyOMpy/requirements.txt` | Зависимости IDyOMpy |
| `tools/IDyOMpy/idyom/__init__.py` | Инициализация пакета |
| `tools/IDyOMpy/idyom/data.py` | Data/viewpoint-подсистема |
| `tools/IDyOMpy/idyom/idyom.py` | Высокоуровневая модель IDyOM |
| `tools/IDyOMpy/idyom/longTermModel.py` | Long-term model |
| `tools/IDyOMpy/idyom/markovChain.py` | Markov chain произвольного порядка |
| `tools/IDyOMpy/idyom/markovChainOrder0.py` | Модель нулевого порядка |
| `tools/IDyOMpy/idyom/myMidi.py` | Чтение монофонического MIDI |
| `tools/IDyOMpy/dataset/filter.py` | Вспомогательный upstream-скрипт фильтрации/балансировки датасета |
| `tools/IDyOMpy/unittests/test_longTermModel.py` | Unit test long-term model |
| `tools/IDyOMpy/unittests/test_markovChain.py` | Unit test markovChain |

#### Документация IDyOMpy

| Путь/паттерн | Назначение |
|---|---|
| `tools/IDyOMpy/docs/conf.py` | Sphinx-конфигурация |
| `tools/IDyOMpy/docs/index.rst` | Исходник главной страницы документации |
| `tools/IDyOMpy/docs/doc.rst` | Исходник основного документа |
| `tools/IDyOMpy/docs/Makefile` | Сборка Sphinx |
| `tools/IDyOMpy/docs/index.html` | Собранная точка входа HTML-документации |
| `tools/IDyOMpy/docs/doctrees/*` | Служебные артефакты Sphinx |
| `tools/IDyOMpy/docs/html/index.html` | HTML-версия документации |
| `tools/IDyOMpy/docs/html/doc.html` | Основной HTML-документ |
| `tools/IDyOMpy/docs/html/genindex.html` | Индекс документации |
| `tools/IDyOMpy/docs/html/search.html` | Страница поиска |
| `tools/IDyOMpy/docs/html/_modules/*` | HTML-представления исходников |
| `tools/IDyOMpy/docs/html/_sources/*` | Сгенерированные исходники документации |
| `tools/IDyOMpy/docs/html/_static/*` | CSS/JS/изображения, обслуживающие сайт документации |

#### Встроенные данные IDyOMpy

| Паттерн | Назначение |
|---|---|
| `tools/IDyOMpy/dataset/**/*.mid*` | Внутренние датасеты upstream-проекта для примеров и исследований |
| `tools/IDyOMpy/stimuli/**/*.mid*` | MIDI-стимулы upstream-проекта |
| `tools/IDyOMpy/generations/*.mid` | Примеры генераций upstream-проекта |

Фактические количества:

| Группа | Количество |
|---|---:|
| `tools/IDyOMpy/dataset/**/*.mid*` | 2168 |
| `tools/IDyOMpy/stimuli/**/*.mid*` | 68 |

#### Встроенная `.git/` внутри `tools/IDyOMpy`

Каталог `tools/IDyOMpy/.git/` содержит:

- refs,
- hooks,
- objects,
- logs,
- config,
- index

и служит как внутренние git-метаданные upstream-проекта. Это не часть логики BioSonification, но физически каталог присутствует в репозитории.

## 7. Веб-каталог `web/`

| Путь | Назначение |
|---|---|
| `web/__init__.py` | Пустой marker-файл пакета |
| `web/README.md` | Отдельное описание веб-интерфейса |
| `web/app.py` | Flask-приложение |
| `web/generator.py` | Backend inference-логика |
| `web/midi_to_audio.py` | Заглушка конвертации MIDI→WAV |
| `web/templates/index.html` | Главный HTML-шаблон |
| `web/static/css/style.css` | Стили фронтенда |
| `web/static/js/app.js` | Клиентская логика фронтенда |
| `web/output/midi/*.mid` | Ранее сгенерированные через web MIDI |
| `web/output/.DS_Store` | системный файл macOS |
| `web/.DS_Store` | системный файл macOS |

Фактическое число файлов в `web/output/midi/`: `10`.

## 8. Документы в `docs/`

| Путь | Назначение |
|---|---|
| `docs/architecture_and_science.md` | Научная логика проекта, формулы, гипотезы, результаты |
| `docs/code_walkthrough.md` | Инженерный walkthrough по коду |
| `docs/thesis_chapters.md` | Академически оформленная заготовка глав дипломной работы |
| `docs/project_structure.md` | Этот каталог-справочник |

## 9. Что считать «ядром проекта»

Если выделить минимальный набор файлов, без которых проект перестанет быть тем самым BioSonification-пайплайном, ядро составят:

1. `run_pipeline.py`
2. `bio_music_pipeline/extractors/bio_extractor.py`
3. `bio_music_pipeline/sonification/mapper.py`
4. `bio_music_pipeline/data/dataset.py`
5. `bio_music_pipeline/data/paired_dataset_creator.py`
6. `bio_music_pipeline/models/transformer.py`
7. `bio_music_pipeline/evaluation/validator.py`
8. `bio_music_pipeline/utils/helpers.py`
9. `configs/pipeline_full_paired.json`

Все остальные файлы либо:

- расширяют исследовательский контур,
- улучшают воспроизводимость,
- дают UI,
- поставляют внешние данные,
- или документируют результаты.

## 10. Что считать побочными, но важными файлами

Побочные, но полезные файлы:

- `CHANGES.md` — помогает понять эволюцию проекта;
- `QWEN.md` — даёт контекст и краткую карту;
- `pipeline_output.log` — локальный лог выполнения;
- `.DS_Store` — не несут научной ценности, но присутствуют физически;
- `results/fasta_generated_midi.mid` и верхнеуровневые legacy-артефакты — свидетельства ранних запусков и демонстраций.

## 11. Практический вывод

Структура репозитория устроена достаточно логично:

- в `bio_music_pipeline/` лежит основной исследовательский код,
- в `configs/` живут параметры экспериментов,
- в `data/` — исходные наборы,
- в `results/` — результаты,
- в `tools/` — автоматизация экспериментов,
- в `web/` — пользовательский интерфейс,
- в `tools/IDyOMpy/` — внешний аналитический модуль.

Именно такая организация делает проект удобным и как для разработки, и как для академического описания.
