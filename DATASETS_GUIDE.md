# Работа с пользовательскими датасетами MIDI и FASTA

Этот документ описывает, как использовать ваши собственные датасеты MIDI и FASTA файлов в проекте.

## Структура директорий

Проект автоматически создаёт стандартные директории для данных:

```
/workspace/
├── data/
│   ├── midi/          # Для ваших MIDI файлов
│   │   └── README.txt
│   └── fasta/         # Для ваших FASTA файлов
│       └── README.txt
├── configs/
│   ├── pipeline_config.json
│   └── data_paths_config.json
└── ...
```

## Как добавить свои данные

### 1. MIDI файлы

1. Скачайте MIDI файлы с любого источника:
   - [MuseScore](https://musescore.com/)
   - [MIDI World](http://www.midiworld.com/)
   - [BitMidi](https://bitmidi.com/)
   - Любого другого сайта

2. Поместите файлы в директорию `data/midi/` или любую поддиректорию:
   ```
   data/midi/
   ├── classical/
   │   ├── bach.mid
   │   └── mozart.mid
   ├── jazz/
   │   └── coltrane.mid
   └── custom.mid
   ```

3. Все файлы будут автоматически обнаружены и обработаны конвейером.

### 2. FASTA файлы

1. Скачайте FASTA файлы с любого источника:
   - [NCBI GenBank](https://www.ncbi.nlm.nih.gov/genbank/)
   - [Ensembl](https://www.ensembl.org/)
   - [UCSC Genome Browser](https://genome.ucsc.edu/)
   - [UniProt](https://www.uniprot.org/)

2. Поместите файлы в директорию `data/fasta/` или любую поддиректорию:
   ```
   data/fasta/
   ├── human/
   │   ├── chromosome1.fasta
   │   └── genes.fa
   ├── microbial/
   │   └── bacteria.fna
   └── custom.fasta
   ```

3. Поддерживаемые форматы: `.fasta`, `.fa`, `.fna`, `.ffn`, `.faa`, `.frn`

## Использование в коде

### Базовое использование

```python
from bio_music_pipeline.data import UniversalDataLoader, setup_user_datasets
from bio_music_pipeline.extractors import FastaDatasetLoader, load_user_fasta_dataset

# Настройка стандартных директорий
setup_user_datasets('/path/to/project')

# Сканирование доступных данных
loader = UniversalDataLoader()
loader.print_data_summary()
```

### Загрузка MIDI файлов

```python
from bio_music_pipeline.data import MusicDataset

# Создание датасета из вашей директории с MIDI
dataset = MusicDataset(
    data_dir='data/midi',  # или любой другой путь
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    seed=42
)

# Получение загрузчиков данных
train_loader = dataset.get_train_loader(batch_size=32)
val_loader = dataset.get_val_loader(batch_size=32)
test_loader = dataset.get_test_loader(batch_size=32)
```

### Загрузка FASTA файлов

```python
from bio_music_pipeline.extractors import FastaDatasetLoader, load_user_fasta_dataset

# Загрузка из одной директории
sequences = load_user_fasta_dataset(['data/fasta'])

# Загрузка из нескольких директорий
sequences = load_user_fasta_dataset([
    'data/fasta/human',
    'data/fasta/microbial',
    '/absolute/path/to/custom/fasta'
])

# Извлечение био-векторов
from bio_music_pipeline.extractors import BioVectorExtractor

extractor = BioVectorExtractor()
bio_vectors = []

for seq in sequences:
    features = extractor.extract_features(seq.sequence)
    bio_vector = extractor.create_bio_vector(features, target_dim=128)
    bio_vectors.append(bio_vector)
```

### Конфигурация путей к данным

Создайте файл конфигурации `configs/data_paths_config.json`:

```json
{
  "data_paths": {
    "midi": [
      "data/midi",
      "/absolute/path/to/your/midi/files"
    ],
    "fasta": [
      "data/fasta",
      "/absolute/path/to/your/fasta/files"
    ]
  },
  "loading_options": {
    "min_midi_duration": 30.0,
    "max_midi_duration": 300.0,
    "min_sequence_length": 100,
    "recursive_search": true
  }
}
```

Загрузка конфигурации:

```python
from bio_music_pipeline.data import UniversalDataLoader

loader = UniversalDataLoader()
config = loader.load_from_config('configs/data_paths_config.json')

# Использование путей из конфигурации
midi_dirs = config['midi_dirs']
fasta_dirs = config['fasta_dirs']
```

## Продвинутые возможности

### Универсальный загрузчик данных

```python
from bio_music_pipeline.data import UniversalDataLoader, DataLoaderConfig

# Создание конфигурации
config = DataLoaderConfig(
    min_midi_duration=60.0,
    max_midi_duration=600.0,
    min_sequence_length=200
)

loader = UniversalDataLoader(config)

# Поиск всех директорий с данными
directories = loader.find_data_directories('/workspace')
print(f"MIDI директории: {directories['midi_dirs']}")
print(f"FASTA директории: {directories['fasta_dirs']}")

# Валидация директории
is_valid = loader.validate_directory('data/midi', file_type='midi')

# Получение списка файлов
midi_files = loader.get_midi_files('data/midi', recursive=True)
fasta_files = loader.get_fasta_files('data/fasta', recursive=True)
```

### Загрузчик FASTA датасетов

```python
from bio_music_pipeline.extractors import FastaDatasetLoader, FastaSequence

loader = FastaDatasetLoader(
    min_sequence_length=100,
    max_sequences=1000  # Ограничить количество последовательностей
)

# Загрузка из директории
sequences = loader.load_from_directory('data/fasta', recursive=True)

# Статистика
stats = loader.get_statistics(sequences)
print(f"Всего последовательностей: {stats['count']}")
print(f"Средняя длина: {stats['mean_length']:.1f}")
print(f"GC-состав: {stats['mean_gc_content']:.3f}")

# Печать подробной сводки
loader.print_summary(sequences)
```

## Пример полного конвейера

```python
from bio_music_pipeline.data import UniversalDataLoader, MusicDataset
from bio_music_pipeline.extractors import load_user_fasta_dataset, BioVectorExtractor

# 1. Настройка и сканирование данных
setup_user_datasets('/workspace')
loader = UniversalDataLoader()
loader.print_data_summary()

# 2. Загрузка FASTA последовательностей
fasta_sequences = load_user_fasta_dataset(['data/fasta'])

# 3. Извлечение био-векторов
extractor = BioVectorExtractor()
bio_vectors = []
for seq in fasta_sequences:
    if len(seq.sequence) >= 100:
        features = extractor.extract_features(seq.sequence)
        bio_vector = extractor.create_bio_vector(features, target_dim=128)
        bio_vectors.append(bio_vector)

print(f"Извлечено {len(bio_vectors)} био-векторов")

# 4. Подготовка MIDI датасета
music_dataset = MusicDataset(
    data_dir='data/midi',
    train_split=0.7,
    val_split=0.15,
    test_split=0.15
)

print(f"MIDI датасет: {len(music_dataset.train_data)} train, "
      f"{len(music_dataset.val_data)} val, "
      f"{len(music_dataset.test_data)} test")

# 5. Обучение модели (см. run_pipeline.py)
```

## Устранение неполадок

### Файлы не обнаруживаются

Убедитесь, что:
- Файлы имеют правильные расширения (.mid, .midi для MIDI; .fasta, .fa для FASTA)
- Файлы находятся в правильной директории или поддиректории
- Рекурсивный поиск включён (по умолчанию включён)

### Ошибки при чтении файлов

- **MIDI**: Проверьте, что файлы не повреждены и являются корректными MIDI файлами
- **FASTA**: Убедитесь, что файлы в правильном формате FASTA (заголовки начинаются с '>')

### Недостаточно данных

Если файлов мало, конвейер может создать синтетические данные для демонстрации. 
Для лучших результатов рекомендуется иметь:
- Минимум 50-100 MIDI файлов для обучения
- Минимум 50-100 FASTA последовательностей длиной >100 bp

## Дополнительные ресурсы

- Документация mido: https://mido.readthedocs.io/
- Формат FASTA: https://en.wikipedia.org/wiki/FASTA_format
- NCBI Handbook: https://www.ncbi.nlm.nih.gov/books/NBK21102/
