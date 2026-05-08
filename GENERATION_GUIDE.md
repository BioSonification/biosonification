# Руководство по генерации музыки из геномов

**Дата создания:** 2026-05-07  
**Статус:** Модель обучена и готова к генерации

---

## Быстрый старт

### Генерация из одного генома

```powershell
# Активация виртуального окружения и генерация
.\.venv\Scripts\python.exe generate_from_fasta_v2.py `
  --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
  --fasta data\fasta\refseq_genomes\GCF_000005845.2_genomic.fna `
  --output results\generated_music\my_output.mid `
  --record-index 0 `
  --config configs\pipeline_v2_medium_rtx2060_fast.json
```

**Важно:** Всегда используйте `.\.venv\Scripts\python.exe` для запуска, иначе будет ошибка `ModuleNotFoundError: No module named 'torch'`

---

## Параметры генерации

### Основные параметры

| Параметр | Описание | Пример |
|----------|----------|--------|
| `--checkpoint` | Путь к обученной модели | `results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt` |
| `--fasta` | Путь к FASTA файлу или директории | `data\fasta\refseq_genomes\GCF_000005845.2_genomic.fna` |
| `--output` | Путь для сохранения MIDI | `results\generated_music\output.mid` |
| `--record-index` | Индекс записи в FASTA (0-based) | `0` |
| `--config` | Конфигурация (опционально) | `configs\pipeline_v2_medium_rtx2060_fast.json` |
| `--device` | Устройство для генерации | `auto` (по умолчанию), `cuda`, `cpu` |

### Как работает адаптивная длительность

**Формула:** `num_bars = max(8, min(32, sequence_length // 200))`

| Длина последовательности | Количество тактов | Примерная длительность |
|--------------------------|-------------------|------------------------|
| 1600 bp | 8 тактов | ~16 секунд |
| 1800 bp | 9 тактов | ~18 секунд |
| 3000 bp | 15 тактов | ~30 секунд |
| 6400 bp | 32 такта | ~64 секунды |

**Токены для мелодии:**
- `melody_max_tokens = num_bars * 48`
- `melody_min_tokens = num_bars * 16`

Это гарантирует, что мелодия заполняет все сгенерированные аккорды без "висячих" аккордов в конце.

---

## Доступные геномы

### Референсные геномы в `data/fasta/refseq_genomes/`

| Организм | Файл | Размер | Записей |
|----------|------|--------|---------|
| E. coli | `GCF_000005845.2_genomic.fna` | 4.6 MB | 1 |
| Yeast | `GCF_000146045.2_genomic.fna` | 12 MB | 17 |
| Drosophila | `GCF_000001215.4_genomic.fna` | 141 MB | 1870 |
| C. elegans | `GCF_000002985.6_genomic.fna` | 99 MB | 7 |
| Arabidopsis | `GCF_000001735.4_genomic.fna` | 117 MB | 7 |

---

## Примеры использования

### 1. Генерация из E. coli (короткая)

```powershell
.\.venv\Scripts\python.exe generate_from_fasta_v2.py `
  --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
  --fasta data\fasta\refseq_genomes\GCF_000005845.2_genomic.fna `
  --output results\generated_music\ecoli.mid `
  --record-index 0 `
  --config configs\pipeline_v2_medium_rtx2060_fast.json
```

**Результат:** ~9 тактов, ~36 нот, ~18 секунд

### 2. Генерация из Drosophila (длинная)

```powershell
.\.venv\Scripts\python.exe generate_from_fasta_v2.py `
  --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
  --fasta data\fasta\refseq_genomes\GCF_000001215.4_genomic.fna `
  --output results\generated_music\drosophila_long.mid `
  --record-index 0 `
  --config configs\pipeline_long_fragment.json
```

**Результат:** ~32 такта, ~74 ноты, ~64 секунды

### 3. Генерация из разных записей одного файла

```powershell
# Первая хромосома yeast
.\.venv\Scripts\python.exe generate_from_fasta_v2.py `
  --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
  --fasta data\fasta\refseq_genomes\GCF_000146045.2_genomic.fna `
  --output results\generated_music\yeast_chr1.mid `
  --record-index 0

# Вторая хромосома yeast
.\.venv\Scripts\python.exe generate_from_fasta_v2.py `
  --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
  --fasta data\fasta\refseq_genomes\GCF_000146045.2_genomic.fna `
  --output results\generated_music\yeast_chr2.mid `
  --record-index 1
```

### 4. Batch генерация для всех геномов

```powershell
# Создать скрипт для генерации всех геномов
$genomes = @(
    @{name="ecoli"; file="GCF_000005845.2_genomic.fna"; index=0},
    @{name="yeast"; file="GCF_000146045.2_genomic.fna"; index=0},
    @{name="drosophila"; file="GCF_000001215.4_genomic.fna"; index=0},
    @{name="worm"; file="GCF_000002985.6_genomic.fna"; index=0},
    @{name="arabidopsis"; file="GCF_000001735.4_genomic.fna"; index=0}
)

foreach ($genome in $genomes) {
    Write-Host "Generating $($genome.name)..."
    .\.venv\Scripts\python.exe generate_from_fasta_v2.py `
      --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
      --fasta "data\fasta\refseq_genomes\$($genome.file)" `
      --output "results\generated_music\$($genome.name)_adaptive.mid" `
      --record-index $genome.index `
      --config configs\pipeline_v2_medium_rtx2060_fast.json
}
```

---

## Конфигурации

### Стандартная конфигурация (короткие фрагменты)

**Файл:** `configs/pipeline_v2_medium_rtx2060_fast.json`

```json
{
  "bio": {
    "fragment_length": 1800,
    "fragment_stride": 1800
  }
}
```

**Результат:** 8-9 тактов (~18 секунд)

### Длинные фрагменты

**Файл:** `configs/pipeline_long_fragment.json`

```json
{
  "bio": {
    "fragment_length": 6400,
    "fragment_stride": 6400
  }
}
```

**Результат:** 32 такта (~64 секунды)

### Создание кастомной конфигурации

```powershell
# Копировать базовую конфигурацию
Copy-Item configs\pipeline_v2_medium_rtx2060_fast.json configs\my_config.json

# Отредактировать вручную или через sed
cat configs\pipeline_v2_medium_rtx2060_fast.json | `
  sed 's/"fragment_length": 1800/"fragment_length": 3600/' > configs\medium_fragment.json
```

---

## Метаданные генерации

Каждая генерация создает JSON файл с метаданными:

```json
{
  "sequence_id": "NC_000913.3::frag000",
  "sequence_type": "dna",
  "cleaned_sequence_length": 1800,
  "translated_protein_length": 488,
  "tonic_pc_hint": 1,
  "tempo_bpm": 89.38,
  "num_bars": 9,
  "harmony_max_tokens": 72,
  "melody_max_tokens": 432,
  "melody_min_tokens": 144,
  "generated_melody_note_count": 36,
  "generated_harmony_bars": [...]
}
```

**Сохранение метаданных:**

```powershell
.\.venv\Scripts\python.exe generate_from_fasta_v2.py `
  --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
  --fasta data\fasta\refseq_genomes\GCF_000005845.2_genomic.fna `
  --output results\generated_music\ecoli.mid `
  --metadata-output results\generated_music\ecoli_metadata.json `
  --record-index 0
```

---

## Troubleshooting

### Ошибка: `ModuleNotFoundError: No module named 'torch'`

**Причина:** Используется системный Python вместо виртуального окружения

**Решение:** Всегда используйте `.\.venv\Scripts\python.exe`

```powershell
# ❌ Неправильно
python generate_from_fasta_v2.py ...

# ✅ Правильно
.\.venv\Scripts\python.exe generate_from_fasta_v2.py ...
```

### Ошибка: `IndexError: record_index=X is outside the valid range`

**Причина:** Указан несуществующий индекс записи

**Решение:** Проверьте количество записей в FASTA файле

```powershell
# Посмотреть количество записей
.\.venv\Scripts\python.exe -c "from Bio import SeqIO; print(len(list(SeqIO.parse('data/fasta/refseq_genomes/GCF_000005845.2_genomic.fna', 'fasta'))))"
```

### Ошибка: `CUDA out of memory`

**Причина:** Недостаточно GPU памяти

**Решение:** Используйте CPU для генерации

```powershell
.\.venv\Scripts\python.exe generate_from_fasta_v2.py `
  --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
  --fasta data\fasta\refseq_genomes\GCF_000005845.2_genomic.fna `
  --output results\generated_music\ecoli.mid `
  --device cpu
```

### Проблема: Harmony повторяется на длинных композициях

**Причина:** Модель обучалась на 4-тактовых сегментах

**Решение:** Это ожидаемое поведение для композиций >8 тактов. Для лучших результатов на длинных композициях нужно переобучить модель на более длинных сегментах.

---

## Сгенерированные файлы (2026-05-07)

### Адаптивная длительность (9 тактов)

| Файл | Организм | Такты | Ноты | Темп |
|------|----------|-------|------|------|
| `ecoli_adaptive.mid` | E. coli | 9 | 36 | 89.4 BPM |
| `yeast_adaptive.mid` | Yeast | 9 | 35 | 87.4 BPM |
| `drosophila_adaptive.mid` | Drosophila | 9 | 28 | 91.1 BPM |
| `worm_adaptive.mid` | C. elegans | 9 | 36 | 74.1 BPM |
| `arabidopsis_adaptive.mid` | Arabidopsis | 9 | 34 | 87.4 BPM |

### Длинная композиция (32 такта)

| Файл | Организм | Такты | Ноты | Темп |
|------|----------|-------|------|------|
| `drosophila_very_long.mid` | Drosophila | 32 | 74 | 88.4 BPM |

---

## Следующие шаги

### 1. Evaluation

Оценить качество сгенерированной музыки:

```powershell
.\.venv\Scripts\python.exe tools\evaluate_structured_v2.py `
  --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
  --fasta data\fasta\refseq_genomes\GCF_000005845.2_genomic.fna `
  --output-dir results\evaluation `
  --max-records 20
```

### 2. Обучение на полном датасете

Для еще лучших результатов:

```powershell
# Предобработка с большим количеством фрагментов
.\.venv\Scripts\python.exe tools\preprocess_bio.py `
  --config configs\pipeline_v2_medium_rtx2060.json `
  --output data\cache\bio_fragments_medium.pkl

# Обучение (~12-18 часов)
.\.venv\Scripts\python.exe train_bio_music_v2.py `
  --config configs\pipeline_v2_medium_rtx2060.json `
  --bio-cache data\cache\bio_fragments_medium.pkl `
  --music-cache data\cache\music_segments_pop909.pkl
```

### 3. Генерация для всех хромосом

Сгенерировать музыку для каждой хромосомы отдельно:

```powershell
# Пример для yeast (17 хромосом)
for ($i=0; $i -lt 17; $i++) {
    .\.venv\Scripts\python.exe generate_from_fasta_v2.py `
      --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
      --fasta data\fasta\refseq_genomes\GCF_000146045.2_genomic.fna `
      --output "results\generated_music\yeast_chr$i.mid" `
      --record-index $i
}
```

---

## Технические детали

### Архитектура модели

- **Bio encoder:** k-mer features + protein features → 256D embedding
- **Harmony model:** Bio-conditioned transformer (384D, 6 heads, 6 layers)
- **Melody model:** Bio-conditioned transformer (384D, 6 heads, 6 layers)

### Обучение

- **Датасет:** 3,746 bio фрагментов × 25,940 music сегментов = 18,730 пар
- **Время обучения:** 52 минуты на RTX 2060
- **Harmony loss:** 0.145 (val), 0.139 (test)
- **Melody loss:** 0.157 (val), 0.165 (test)

### Генерация

1. **Bio encoding:** FASTA → k-mers + protein → 256D vector
2. **Harmony generation:** Bio vector → chord progression (8 tokens/bar)
3. **Melody generation:** Bio vector + harmony → melody notes (48 tokens/bar)
4. **Rendering:** Tokens → MIDI file

---

**Автор:** Claude Code  
**Последнее обновление:** 2026-05-07 04:00 UTC
