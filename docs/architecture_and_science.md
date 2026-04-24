# Научная Архитектура И Методология

Этот документ описывает **актуальную научно-инженерную постановку проекта после внедрения `v2` пайплайна**. Старая архитектура сохранена в репозитории, но здесь основным считается новый контур, который лучше соответствует реальной задаче: генерации полифонической музыки, похожей на обучающий корпус, но смещённой биологическим условием.

## 1. Актуальная постановка задачи

### 1.1 Что именно моделируется

Проект больше не трактует задачу как “найти истинное соответствие между FASTA и готовой музыкой”. Вместо этого реализована более корректная постановка:

- есть биологическая последовательность;
- из неё извлекается biologically informed embedding;
- embedding преобразуется в музыкальный control profile;
- control profile сопоставляется с распределением реального музыкального корпуса;
- генератор строит новую музыку в стиле корпуса, но под заданным биологическим смещением.

Это важное методологическое уточнение: модель учится **условной генерации под био-сигналом**, а не “восстановлению истинной музыкальности генома”.

## 2. Основные научные принципы `v2`

1. Музыка должна быть **полифонической сразу**, без промежуточного сведения к монофонии.
2. Биологическое conditioning должно опираться не только на composition statistics, но и на **структурные и физико-химические признаки**.
3. Pairing должен быть не ранговым по одному scalar, а **многомерным и калиброванным**.
4. Генератор должен учиться не просто воспроизводить токены, а **удерживать музыкальный профиль**, заданный био-признаками.

## 3. Биологический слой

Источник реализации:

- `bio_music_pipeline/v2/bio.py`

### 3.1 Поддерживаемые типы последовательностей

Энкодер различает:

- `dna`
- `rna`
- `protein`

### 3.2 Фрагментация

FASTA-записи больше не считаются как одна длинная сущность по умолчанию. Они режутся на фрагменты:

- `fragment_length`
- `fragment_stride`
- `max_fragments_per_record`

Это нужно для:

- увеличения числа train samples;
- более стабильного pairing;
- совместимости с небольшими локальными FASTA-наборами.

### 3.3 Извлекаемые признаки

#### Нуклеотидный блок

- длина последовательности
- nucleotide frequencies
- entropy
- GC content
- GC/AT skew
- codon entropy
- periodicity
- k-mer distributions

#### Белковый блок

Если последовательность DNA/RNA, encoder ищет longest ORF и переводит её в белок. Далее через `Biopython ProtParam` вычисляются:

- aromaticity
- instability index
- isoelectric point
- GRAVY
- molecular weight
- flexibility mean/std
- helix / turn / sheet fractions
- amino acid composition
- charge density

#### RNA structural block

Если доступен пакет `ViennaRNA`, дополнительно считаются:

- minimum free energy
- paired fraction
- loop fraction
- transition rate in dot-bracket structure
- structural entropy

### 3.4 Итог биологического представления

На выходе encoder формирует:

- `vector`: фиксированный `bio embedding`
- `control_profile`: 6-мерный музыкально-интерпретируемый профиль

`control_profile` содержит нормализованные оценки:

- tempo tendency
- density
- polyphony
- register
- harmony
- mode tendency

## 4. Музыкальный слой

Источник:

- `bio_music_pipeline/v2/dataset.py`

### 4.1 Почему отказ от монофонии принципиален

Старый контур обучался на бедном монофоническом представлении и поэтому не мог стабильно порождать:

- насыщенные аккордовые структуры;
- мелодию поверх аккомпанемента;
- реальную вертикаль;
- более человечную фактуру.

В `v2` модель с самого начала видит полифонический символический материал.

### 4.2 Сегментация музыкального корпуса

Произведения режутся по тактам на окна:

- `bars_per_segment`
- `segment_hop_bars`

Каждый сегмент фильтруется по:

- минимальному числу нот;
- доле полифонических onset’ов.

### 4.3 Дескрипторы сегмента

Для каждого сегмента считаются:

- tempo
- density
- polyphony
- register
- harmony
- mode

Эти признаки играют двойную роль:

1. используются в pairing;
2. используются как discrete control tokens при генерации.

### 4.4 Токенизация

Новая схема:

- `BOS`
- control prefix
- `SEP`
- `TIME_*`
- `NOTE_*`
- `DUR_*`
- `VEL_*`
- `EOS`

Она проще для локального генератора и лучше соответствует задаче полифонической event-based музыки, чем старый `NOTE_ON / NOTE_OFF / SHIFT / VEL` контур.

## 5. Pairing

Источник:

- `bio_music_pipeline/v2/pairing.py`

### 5.1 Проблема старого pairing

Старый pipeline сопоставлял bio и MIDI по одному scalar complexity score, что приводило к шумному и методологически слабому matching.

### 5.2 Новый подход

Сначала био-профили калибруются под распределение музыкального корпуса:

- `bio_mean`, `bio_std`
- `music_mean`, `music_std`

Потом для каждого bio-fragment ищутся ближайшие музыкальные сегменты по weighted distance в пространстве 6 дескрипторов.

Далее создаётся `top-k` many-to-many matching с вероятностными весами.

Итог:

- pairing больше не жёсткий 1:1;
- conditioning становится мягким и распределённым;
- модель получает несколько правдоподобных музыкальных соответствий на один биологический паттерн.

## 6. Генеративная модель

Источник:

- `bio_music_pipeline/v2/model.py`

### 6.1 Архитектура

Используется компактный `ControlConditionedTransformer`:

1. token embeddings
2. positional embeddings
3. bio projection в memory tokens
4. Transformer decoder
5. language-model head
6. descriptor prediction head

### 6.2 Два канала conditioning

#### Дискретный канал

Control tokens кодируют:

- tempo bucket
- density bucket
- polyphony bucket
- register bucket
- harmony bucket
- mode token

#### Непрерывный канал

`bio vector` подаётся в cross-attention memory.

Этот двухуровневый conditioning лучше соответствует реальной задаче:

- дискретные токены задают музыкальную сцену;
- непрерывный embedding уточняет её биологическим сигналом.

### 6.3 Функция потерь

Используются две компоненты:

- token cross-entropy
- descriptor alignment loss

Вторая компонента заставляет модель удерживать желаемый музыкальный профиль, а не только копировать локальную статистику токенов.

## 7. Обучение

Источник:

- `bio_music_pipeline/v2/train.py`

### 7.1 Контур обучения

1. кодирование FASTA;
2. загрузка и сегментация музыкального корпуса;
3. независимый split bio/music;
4. pairing для train/val/test;
5. обучение conditional Transformer;
6. сохранение лучшего checkpoint;
7. test evaluation;
8. smoke generation.

### 7.2 Почему это подходит под `RTX 2060 6 GB`

В конфиге `configs/pipeline_v2_small.json` модель и обучение ужаты под локальную GPU:

- `d_model = 256`
- `n_layers = 4`
- `n_heads = 4`
- `batch_size = 4`
- `grad_accum_steps = 4`
- `mixed_precision = true`

То есть фактическая стратегия — небольшая модель плюс accumulation и AMP.

## 8. Проверка корректности

### 8.1 Unit/smoke checks

`tests/test_v2_pipeline.py` проверяет:

- что bio encoder работает;
- что tokenizer умеет roundtrip;
- что модель делает forward и generate без падения.

### 8.2 Training-time checks

После обучения ожидаются:

- `metrics.json`
- `best_model.pt`
- `sample_from_training_pipeline.mid`

Корректный training run должен показывать:

- `device = cuda`
- убывающий `val.loss`
- ненулевой smoke MIDI

### 8.3 Inference-time checks

Отдельный запуск `generate_from_fasta_v2.py` должен создавать:

- `generated_from_fasta.mid`
- `generated_from_fasta.json`

Дополнительно можно проверять:

- число нот;
- длительность;
- число полифонических onset’ов.

## 9. Ограничения текущей версии

1. `v2` пока использует компактный локальный symbolic generator, а не большой pretrained music foundation model.
2. Web-layer ещё не переподключён на `v2`.
3. Fallback-корпус `music21` удобен для smoke-run, но не заменяет полноценный внешний полифонический датасет.
4. Парные соответствия по-прежнему являются псевдо-парами, хотя и гораздо более обоснованными, чем в старом контуре.

## 10. Что изменилось по сравнению со старой архитектурой

Старая архитектура:

- опиралась на слабый DNA-only encoder;
- использовала бедный токенизатор;
- часто теряла conditioning;
- допускала методологически сомнительный pairing.

Новая архитектура:

- вводит richer biological representation;
- работает сразу с полифонией;
- делает control-aware generation;
- сохраняет независимый и проверяемый inference path;
- лучше согласована с реальной исследовательской задачей.
