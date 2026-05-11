# Быстрая предобработка данных

## Обзор

Новая система предобработки разделяет извлечение данных и обучение:
- **Предобработка** (один раз, долго) → сохраняет в кэш
- **Обучение/Evaluation** (многократно, быстро) → загружает из кэша

## Шаг 1: Предобработка биологических данных

```powershell
python tools\preprocess_bio.py `
  --config configs\pipeline_v2_medium_rtx2060_fast.json `
  --output data\cache\bio_fragments_fast.pkl
```

**Время:** ~30 секунд (fast config) или ~1-2 часа (full config)

**Результат:** `data/cache/bio_fragments_fast.pkl` (26 MB)

**Оптимизации в fast config:**
- `max_fragments_per_record: 10` (вместо 50)
- `fragment_stride: 1800` (без перекрытия)
- `use_vienna_rna: false` (отключена медленная структура РНК)

## Шаг 2: Предобработка музыкальных данных

```powershell
python tools\preprocess_music.py `
  --config configs\pipeline_v2_medium_rtx2060_fast.json `
  --output data\cache\music_segments_pop909.pkl
```

**Время:** ~2-5 минут (POP909 только)

**Результат:** `data/cache/music_segments_pop909.pkl`

**Оптимизации:**
- Только POP909 (909 файлов вместо 4,174)
- Быстрое определение тональности (без `score.analyze("key")`)
- Пропуск файлов >2MB

## Шаг 3: Быстрый отчет по датасету

```powershell
python tools\quick_dataset_report.py `
  --bio-cache data\cache\bio_fragments_fast.pkl `
  --music-cache data\cache\music_segments_pop909.pkl `
  --output-dir results\quick_report
```

**Время:** <1 секунда

**Результат:** `results/quick_report/quick_report.json`

## Шаг 4: Обучение (TODO - нужно обновить train_bio_music_v2.py)

```powershell
python train_bio_music_v2.py `
  --config configs\pipeline_v2_medium_rtx2060_fast.json `
  --bio-cache data\cache\bio_fragments_fast.pkl `
  --music-cache data\cache\music_segments_pop909.pkl
```

**Время:** Обучение начинается сразу (без ожидания предобработки)

## Конфигурации

### Fast (для быстрых экспериментов)
- **Конфиг:** `configs/pipeline_v2_medium_rtx2060_fast.json`
- **Bio фрагменты:** ~3,700 (10 на геном)
- **Music сегменты:** ~5,000-10,000 (POP909)
- **Предобработка:** ~3-5 минут
- **Обучение:** ~8-12 часов

### Medium (оригинальный)
- **Конфиг:** `configs/pipeline_v2_medium_rtx2060.json`
- **Bio фрагменты:** ~18,000 (50 на геном)
- **Music сегменты:** ~5,000-10,000 (POP909)
- **Предобработка:** ~1-2 часа
- **Обучение:** ~12-18 часов

### Large (для A100/H100)
- **Конфиг:** `configs/pipeline_v2_large.json`
- **Bio фрагменты:** ~36,000 (100 на геном)
- **Music сегменты:** ~15,000-20,000 (POP909 + maestro)
- **Предобработка:** ~3-5 часов
- **Обучение:** ~24-48 часов

## Переиспользование кэша

Кэш можно использовать многократно:

```powershell
# Эксперимент 1: baseline
python train_bio_music_v2.py --config config1.json --bio-cache cache.pkl --music-cache cache.pkl

# Эксперимент 2: другой learning rate
python train_bio_music_v2.py --config config2.json --bio-cache cache.pkl --music-cache cache.pkl

# Эксперимент 3: другая архитектура
python train_bio_music_v2.py --config config3.json --bio-cache cache.pkl --music-cache cache.pkl
```

Данные извлекаются один раз, используются многократно!

## Принудительная переобработка

Если нужно обновить кэш:

```powershell
python tools\preprocess_bio.py --config config.json --output cache.pkl --force
```

## Проверка кэша

```powershell
# Проверить что в кэше (без переобработки)
python tools\preprocess_bio.py --config config.json --output cache.pkl
# Выведет: "Cache already exists, use --force to reprocess"
```

## Преимущества новой системы

✅ **Быстрые эксперименты** - обучение начинается сразу  
✅ **Отладка по частям** - можно отдельно проверить bio или music  
✅ **Прогресс-бары** - видно что происходит  
✅ **Переиспользование** - один кэш для многих экспериментов  
✅ **Экономия времени** - предобработка 1 раз вместо каждого запуска  

## Следующие шаги

1. ✅ Создать `preprocess_bio.py`
2. ✅ Создать `preprocess_music.py`
3. ✅ Создать `quick_dataset_report.py`
4. ⏳ Обновить `train_bio_music_v2.py` для работы с кэшем
5. ⏳ Обновить `evaluate_structured_v2.py` для работы с кэшем
