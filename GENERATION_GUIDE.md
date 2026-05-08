# Руководство по генерации музыки из геномов

**Дата создания:** 2026-05-07  
**Последнее обновление:** 2026-05-08  
**Статус:** Фрагментированная генерация с 4-тактовой моделью

---

## Быстрый старт

### Фрагментированная генерация (рекомендуется)

```powershell
# Генерация с автоматическим разбиением на фрагменты
.\.venv\Scripts\python.exe generate_from_fasta_v2_fragmented.py `
  --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
  --fasta data\fasta\refseq_genomes\GCF_000005845.2_genomic.fna `
  --output results\generated_music\ecoli_fragmented.mid `
  --bars-per-fragment 4 `
  --metadata-output results\generated_music\ecoli_fragmented.json
```

**Преимущества:**
- ✅ Стабильное качество для любой длины последовательности
- ✅ Длинные последовательности → длинные композиции
- ✅ Больше разнообразия (каждый фрагмент имеет свой био-вектор)

### Baseline генерация (для сравнения)

```powershell
# Генерация без фрагментации (использует только первый фрагмент)
.\.venv\Scripts\python.exe generate_from_fasta_v2.py `
  --checkpoint results\v2_medium_rtx2060_long\checkpoints\structured_pipeline.pt `
  --fasta data\fasta\refseq_genomes\GCF_000005845.2_genomic.fna `
  --output results\generated_music\ecoli_baseline.mid `
  --config configs\pipeline_v2_medium_rtx2060_long.json
```

**Важно:** Всегда используйте `.\.venv\Scripts\python.exe` для запуска, иначе будет ошибка `ModuleNotFoundError: No module named 'torch'`

---

## Сравнение подходов

### Baseline (generate_from_fasta_v2.py)

**Как работает:**
- Использует только первый фрагмент последовательности (1800 bp)
- Генерирует одну композицию за раз
- Адаптивная длина: `num_bars = max(8, min(32, sequence_length // 200))`

**Проблемы:**
- ❌ Для длинных последовательностей (>1800 bp) модель выходит за пределы обучающего распределения
- ❌ Качество падает: гармония повторяется, мелодия становится монотонной
- ❌ Не использует всю информацию из длинной последовательности

**Пример результата (10000 bp):**
- Использует: 1800 bp (первый фрагмент)
- Генерирует: 9 тактов, 63 ноты
- Качество: низкое для длинных композиций

### Фрагментированная генерация (generate_from_fasta_v2_fragmented.py)

**Как работает:**
1. Разбивает входную последовательность на фрагменты по 1800 bp
2. Для каждого фрагмента генерирует короткий сегмент (4 или 8 тактов)
3. Склеивает все сегменты в одну композицию

**Преимущества:**
- ✅ Каждый фрагмент генерируется в пределах обучающего распределения → стабильное качество
- ✅ Использует всю последовательность
- ✅ Больше разнообразия (каждый фрагмент имеет свой био-вектор и темп)

**Пример результата (10000 bp):**
- Использует: 10000 bp (6 фрагментов)
- Генерирует: 24 такта, 166 нот
- Качество: стабильное на протяжении всей композиции

---

## Параметры фрагментированной генерации

### Основные параметры

| Параметр | Описание | Значение по умолчанию |
|----------|----------|----------------------|
| `--checkpoint` | Путь к обученной модели | Обязательный |
| `--fasta` | Путь к FASTA файлу | Обязательный |
| `--output` | Путь для сохранения MIDI | Обязательный |
| `--bars-per-fragment` | Количество тактов на фрагмент (4 или 8) | `4` (рекомендуется) |
| `--record-index` | Индекс записи в FASTA (0-based) | `0` |
| `--config` | Конфигурация (опционально) | Из checkpoint |
| `--metadata-output` | Путь для сохранения метаданных JSON | Опционально |
| `--device` | Устройство для генерации | `auto` |

### Выбор bars-per-fragment

**4 такта (рекомендуется):**
- Модель: `v2_medium_rtx2060_fast`
- Validation loss: Harmony 0.145, Melody 0.157 (лучше!)
- Больше фрагментов → больше разнообразия
- Лучшее качество генерации

**8 тактов:**
- Модель: `v2_medium_rtx2060_long`
- Validation loss: Harmony 0.179, Melody 0.215
- Меньше фрагментов → меньше переходов
- Более длинные сегменты

### Как работает фрагментация

```
Длинная последовательность (10000 bp)
↓
Разбить на фрагменты по 1800 bp (stride 1800)
↓
Fragment 1 (0-1800)    → Bio vector 1 → Generate 4 bars → MIDI segment 1
Fragment 2 (1800-3600) → Bio vector 2 → Generate 4 bars → MIDI segment 2
Fragment 3 (3600-5400) → Bio vector 3 → Generate 4 bars → MIDI segment 3
Fragment 4 (5400-7200) → Bio vector 4 → Generate 4 bars → MIDI segment 4
Fragment 5 (7200-9000) → Bio vector 5 → Generate 4 bars → MIDI segment 5
Fragment 6 (9000-10000) → Bio vector 6 → Generate 4 bars → MIDI segment 6
↓
Concatenate all MIDI segments → Final MIDI (24 bars)
```

### Расчет длины композиции

| Длина последовательности | Фрагментов | Тактов (4-bar) | Тактов (8-bar) | Примерная длительность |
|--------------------------|------------|----------------|----------------|------------------------|
| 1800 bp | 1 | 4 | 8 | 8-16 секунд |
| 3600 bp | 2 | 8 | 16 | 16-32 секунды |
| 10000 bp | 6 | 24 | 48 | 48-96 секунд |
| 20000 bp | 12 | 48 | 96 | 96-192 секунды |

---

## Доступные модели

### 4-тактовая модель (рекомендуется)

**Checkpoint:** `results/v2_medium_rtx2060_fast/checkpoints/structured_pipeline.pt`  
**Config:** `configs/pipeline_v2_medium_rtx2060_fast.json`

**Характеристики:**
- Архитектура: 384D, 6 heads, 6 layers
- Обучающие данные: 4-тактовые сегменты из POP909
- Validation loss: Harmony 0.145, Melody 0.157
- Время обучения: 52 минуты на RTX 2060

**Использование:**
```powershell
.\.venv\Scripts\python.exe generate_from_fasta_v2_fragmented.py `
  --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
  --fasta your_sequence.fna `
  --output output.mid `
  --bars-per-fragment 4
```

### 8-тактовая модель

**Checkpoint:** `results/v2_medium_rtx2060_long/checkpoints/structured_pipeline.pt`  
**Config:** `configs/pipeline_v2_medium_rtx2060_long.json`

**Характеристики:**
- Архитектура: 384D, 6 heads, 6 layers
- Обучающие данные: 8-тактовые сегменты из POP909
- Validation loss: Harmony 0.179, Melody 0.215
- Время обучения: ~60 минут на RTX 2060

**Использование:**
```powershell
.\.venv\Scripts\python.exe generate_from_fasta_v2_fragmented.py `
  --checkpoint results\v2_medium_rtx2060_long\checkpoints\structured_pipeline.pt `
  --fasta your_sequence.fna `
  --output output.mid `
  --bars-per-fragment 8
```

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

### 1. Короткая последовательность (E. coli, 1800 bp)

```powershell
.\.venv\Scripts\python.exe generate_from_fasta_v2_fragmented.py `
  --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
  --fasta data\fasta\refseq_genomes\GCF_000005845.2_genomic.fna `
  --output results\generated_music\ecoli_short.mid `
  --bars-per-fragment 4 `
  --metadata-output results\generated_music\ecoli_short.json
```

**Ожидаемый результат:**
- 1 фрагмент
- 4 такта
- ~20 нот мелодии
- ~8 секунд

### 2. Средняя последовательность (3600 bp)

```powershell
# Создать тестовую последовательность
.\.venv\Scripts\python.exe -c "from Bio import SeqIO; r = list(SeqIO.parse('data/fasta/refseq_genomes/GCF_000005845.2_genomic.fna', 'fasta'))[0]; print(f'>{r.id}_3600bp\n{str(r.seq)[:3600]}')" | Out-File -Encoding utf8 "data\test\ecoli_3600bp.fna"

# Генерация
.\.venv\Scripts\python.exe generate_from_fasta_v2_fragmented.py `
  --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
  --fasta data\test\ecoli_3600bp.fna `
  --output results\generated_music\ecoli_medium.mid `
  --bars-per-fragment 4
```

**Ожидаемый результат:**
- 2 фрагмента
- 8 тактов
- ~48 нот мелодии
- ~16 секунд

### 3. Длинная последовательность (Drosophila, 10000 bp)

```powershell
# Создать тестовую последовательность
.\.venv\Scripts\python.exe -c "from Bio import SeqIO; r = list(SeqIO.parse('data/fasta/refseq_genomes/GCF_000001215.4_genomic.fna', 'fasta'))[0]; print(f'>{r.id}_10000bp\n{str(r.seq)[:10000]}')" | Out-File -Encoding utf8 "data\test\drosophila_10000bp.fna"

# Генерация
.\.venv\Scripts\python.exe generate_from_fasta_v2_fragmented.py `
  --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
  --fasta data\test\drosophila_10000bp.fna `
  --output results\generated_music\drosophila_long.mid `
  --bars-per-fragment 4
```

**Ожидаемый результат:**
- 6 фрагментов
- 24 такта
- ~150-170 нот мелодии
- ~48 секунд

### 4. Сравнение baseline vs fragmented

```powershell
# Baseline (только первый фрагмент)
.\.venv\Scripts\python.exe generate_from_fasta_v2.py `
  --checkpoint results\v2_medium_rtx2060_long\checkpoints\structured_pipeline.pt `
  --fasta data\test\drosophila_10000bp.fna `
  --output results\comparison\drosophila_baseline.mid `
  --config configs\pipeline_v2_medium_rtx2060_long.json

# Fragmented (все фрагменты)
.\.venv\Scripts\python.exe generate_from_fasta_v2_fragmented.py `
  --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
  --fasta data\test\drosophila_10000bp.fna `
  --output results\comparison\drosophila_fragmented.mid `
  --bars-per-fragment 4
```

**Сравнение результатов:**

| Метрика | Baseline | Fragmented |
|---------|----------|------------|
| Использовано bp | 1800 | 10000 |
| Фрагментов | 1 | 6 |
| Тактов | 9 | 24 |
| Нот мелодии | ~63 | ~166 |
| Качество | Низкое (повторения) | Стабильное |

### 5. Batch генерация для всех геномов

```powershell
$genomes = @(
    @{name="ecoli"; file="GCF_000005845.2_genomic.fna"; index=0},
    @{name="yeast"; file="GCF_000146045.2_genomic.fna"; index=0},
    @{name="drosophila"; file="GCF_000001215.4_genomic.fna"; index=0},
    @{name="worm"; file="GCF_000002985.6_genomic.fna"; index=0},
    @{name="arabidopsis"; file="GCF_000001735.4_genomic.fna"; index=0}
)

foreach ($genome in $genomes) {
    Write-Host "Generating $($genome.name)..."
    .\.venv\Scripts\python.exe generate_from_fasta_v2_fragmented.py `
      --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
      --fasta "data\fasta\refseq_genomes\$($genome.file)" `
      --output "results\generated_music\$($genome.name)_fragmented.mid" `
      --bars-per-fragment 4 `
      --metadata-output "results\generated_music\$($genome.name)_fragmented.json"
}
```

---

## Метаданные генерации

Фрагментированная генерация создает подробный JSON файл с метаданными:

```json
{
  "sequence_id": "NC_000913.3_3600bp",
  "full_sequence_length": 3600,
  "fragment_length": 1800,
  "stride": 1800,
  "num_fragments": 2,
  "bars_per_fragment": 4,
  "total_bars": 8,
  "total_melody_notes": 48,
  "output_midi": "C:\\...\\ecoli_3600_fragmented.mid",
  "checkpoint_path": "results\\v2_medium_rtx2060_fast\\checkpoints\\structured_pipeline.pt",
  "config_source": "checkpoint",
  "device": "cuda",
  "fragments": [
    {
      "fragment_index": 0,
      "start_position": 0,
      "fragment_length": 1800,
      "num_bars": 4,
      "tempo_bpm": 89.38,
      "harmony_bars": 4,
      "melody_notes": 25
    },
    {
      "fragment_index": 1,
      "start_position": 1800,
      "fragment_length": 1800,
      "num_bars": 4,
      "tempo_bpm": 92.56,
      "harmony_bars": 4,
      "melody_notes": 26
    }
  ]
}
```

**Ключевые поля:**
- `num_fragments` — количество фрагментов
- `total_bars` — общее количество тактов
- `total_melody_notes` — общее количество нот мелодии
- `fragments` — детали по каждому фрагменту (темп, количество нот)

---

## Troubleshooting

### Ошибка: `ModuleNotFoundError: No module named 'torch'`

**Причина:** Используется системный Python вместо виртуального окружения

**Решение:** Всегда используйте `.\.venv\Scripts\python.exe`

```powershell
# ❌ Неправильно
python generate_from_fasta_v2_fragmented.py ...

# ✅ Правильно
.\.venv\Scripts\python.exe generate_from_fasta_v2_fragmented.py ...
```

### Ошибка: `IndexError: record_index=X is outside the valid range`

**Причина:** Указан несуществующий индекс записи

**Решение:** Проверьте количество записей в FASTA файле

```powershell
.\.venv\Scripts\python.exe -c "from Bio import SeqIO; print(len(list(SeqIO.parse('your_file.fna', 'fasta'))))"
```

### Ошибка: `CUDA out of memory`

**Причина:** Недостаточно GPU памяти

**Решение:** Используйте CPU для генерации

```powershell
.\.venv\Scripts\python.exe generate_from_fasta_v2_fragmented.py `
  --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
  --fasta your_sequence.fna `
  --output output.mid `
  --device cpu
```

### Ошибка: `Checkpoint/config mismatch`

**Причина:** Несоответствие архитектуры модели и конфига

**Решение:** Используйте правильный конфиг для модели

```powershell
# Для 4-тактовой модели
--config configs\pipeline_v2_medium_rtx2060_fast.json

# Для 8-тактовой модели
--config configs\pipeline_v2_medium_rtx2060_long.json
```

### Проблема: MIDI файл пустой (106 байт)

**Причина:** Эта проблема была исправлена в последней версии

**Решение:** Убедитесь, что используете актуальную версию кода

```powershell
# Проверить размер сгенерированного MIDI
(Get-Item "output.mid").Length
# Должно быть >300 байт для коротких последовательностей
```

---

## Технические детали

### Архитектура модели

- **Bio encoder:** k-mer features + protein features + ESM embeddings → 256D embedding
- **Harmony model:** Bio-conditioned transformer (384D, 6 heads, 6 layers)
- **Melody model:** Bio-conditioned transformer (384D, 6 heads, 6 layers)

### Обучение 4-тактовой модели

- **Датасет:** 3,746 bio фрагментов × 15,226 music сегментов (4 bars)
- **Время обучения:** 52 минуты на RTX 2060
- **Harmony loss:** 0.145 (val), 0.139 (test)
- **Melody loss:** 0.157 (val), 0.165 (test)

### Процесс фрагментированной генерации

1. **Фрагментация:** Разбить последовательность на фрагменты по 1800 bp
2. **Bio encoding:** Для каждого фрагмента: FASTA → k-mers + protein → 256D vector
3. **Harmony generation:** Bio vector → chord progression (8 tokens/bar)
4. **Melody generation:** Bio vector + harmony → melody notes (48 tokens/bar)
5. **Concatenation:** Склеить все MIDI сегменты с правильным смещением по времени
6. **Rendering:** Финальный Score → MIDI file

---

## Следующие шаги

### 1. Веб-интерфейс

Запустить веб-интерфейс с фрагментированной генерацией:

```powershell
.\.venv\Scripts\python.exe -m web.app
# Откройте http://localhost:5001
```

### 2. Evaluation

Оценить качество сгенерированной музыки:

```powershell
.\.venv\Scripts\python.exe tools\evaluate_structured_v2.py `
  --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
  --fasta data\fasta\refseq_genomes\GCF_000005845.2_genomic.fna `
  --output-dir results\evaluation `
  --max-records 20
```

### 3. Генерация для всех хромосом

Сгенерировать музыку для каждой хромосомы отдельно:

```powershell
# Пример для yeast (17 хромосом)
for ($i=0; $i -lt 17; $i++) {
    .\.venv\Scripts\python.exe generate_from_fasta_v2_fragmented.py `
      --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
      --fasta data\fasta\refseq_genomes\GCF_000146045.2_genomic.fna `
      --output "results\generated_music\yeast_chr$i.mid" `
      --record-index $i `
      --bars-per-fragment 4
}
```

---

**Автор:** Claude Code  
**Последнее обновление:** 2026-05-08
