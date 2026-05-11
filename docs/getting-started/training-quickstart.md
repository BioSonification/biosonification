# Быстрый старт обучения

## Текущее состояние

✅ **Предобработка завершена!**

- Биологические данные: `data/cache/bio_fragments_fast.pkl` (3,746 фрагментов, 26.3 MB)
- Музыкальные данные: `data/cache/music_segments_pop909.pkl` (25,940 сегментов, 31.4 MB)
- Ожидаемые пары: ~18,730

**Время предобработки:** 2.2 минуты (вместо часов!)

---

## Запуск обучения

### Вариант 1: С использованием кэша (рекомендуется)

```powershell
cd C:\Users\vlasi\Documents\biosonification

python train_bio_music_v2.py `
  --config configs\pipeline_v2_medium_rtx2060_fast.json `
  --bio-cache data\cache\bio_fragments_fast.pkl `
  --music-cache data\cache\music_segments_pop909.pkl
```

**Преимущества:**
- ✅ Обучение начинается **сразу** (без ожидания предобработки)
- ✅ Можно запускать многократно с разными гиперпараметрами
- ✅ Легко отлаживать

**Ожидаемое время:** 8-12 часов на RTX 2060

### Вариант 2: Без кэша (классический)

```powershell
python train_bio_music_v2.py `
  --config configs\pipeline_v2_medium_rtx2060_fast.json
```

Скрипт сам извлечет данные (займет ~2-3 минуты), затем начнет обучение.

---

## Мониторинг обучения

### Логи

```powershell
# Смотреть логи в реальном времени
Get-Content results\v2_medium_rtx2060_fast\train.log -Wait

# Смотреть метрики
Get-Content results\v2_medium_rtx2060_fast\metrics.json | ConvertFrom-Json
```

### GPU

```powershell
# Мониторинг GPU
nvidia-smi -l 1
```

### Checkpoints

Модель сохраняется в:
- `results/v2_medium_rtx2060_fast/checkpoints/harmony_model_best.pt`
- `results/v2_medium_rtx2060_fast/checkpoints/melody_model_best.pt`
- `results/v2_medium_rtx2060_fast/checkpoints/structured_pipeline.pt`

---

## Ожидаемые метрики

### Во время обучения

**Harmony model (30 эпох):**
- Начальный loss: ~1.5
- Финальный loss: ~0.3-0.4
- Время: ~3-4 часа

**Melody model (35 эпох):**
- Начальный loss: ~1.8
- Финальный loss: ~0.4-0.5
- Время: ~5-8 часов

### После обучения

**Целевые метрики (по сравнению с baseline):**
- Pitch range: >22 (было 14.6)
- Unique pitches: >16 (было 10.8)
- Pitch entropy: >3.1 (было 2.86)
- Chord-tone ratio: >68% (было 60%)

---

## Генерация музыки

После обучения можно сгенерировать музыку:

```powershell
python generate_from_fasta_v2.py `
  --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
  --fasta data\fasta\refseq_genomes\GCF_000005845.2_genomic.fna `
  --output results\generated_music\ecoli.mid `
  --max-records 5
```

---

## Evaluation

```powershell
python tools\evaluate_structured_v2.py `
  --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
  --fasta data\fasta\refseq_genomes\GCF_000005845.2_genomic.fna `
  --output-dir results\evaluation `
  --max-records 20
```

---

## Troubleshooting

### Out of Memory

Если GPU памяти не хватает:

1. Уменьшить `batch_size` в конфиге (8 → 4)
2. Уменьшить `d_model` (384 → 256)
3. Отключить `mixed_precision: false`

### Медленное обучение

Если обучение идет медленно:

1. Проверить GPU utilization: `nvidia-smi`
2. Увеличить `num_workers` (2 → 4)
3. Проверить что используется кэш

### Loss не падает

Если loss не падает после 5 эпох:

1. Проверить learning rate (может быть слишком большой/маленький)
2. Проверить что данные загружаются корректно
3. Попробовать уменьшить `weight_decay`

---

## Следующие шаги

После успешного обучения:

1. ✅ Запустить evaluation
2. ✅ Сгенерировать музыку из всех 5 геномов
3. ✅ Сравнить метрики с baseline
4. ✅ Провести ablation study (опционально)
5. ✅ Обучить на полном датасете (medium/large config)

---

## Конфигурации

### Fast (текущая)
- Bio: 3,746 фрагментов
- Music: 25,940 сегментов
- Пары: ~18,730
- Время обучения: 8-12 часов

### Medium (следующий шаг)
- Bio: ~18,000 фрагментов (50 на геном)
- Music: 25,940 сегментов
- Пары: ~90,000
- Время обучения: 12-18 часов

### Large (для A100/H100)
- Bio: ~36,000 фрагментов (100 на геном)
- Music: ~40,000 сегментов (POP909 + maestro)
- Пары: ~180,000
- Время обучения: 24-48 часов

---

**Дата:** 2026-05-06  
**Статус:** Готово к обучению! 🚀
