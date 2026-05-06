# План улучшения проекта BioSonification

**Версия:** 1.0  
**Дата создания:** 2026-05-05  
**Статус:** В процессе выполнения

---

## 📋 Оглавление

1. [Текущее состояние проекта](#текущее-состояние-проекта)
2. [Критические проблемы](#критические-проблемы)
3. [План улучшений по фазам](#план-улучшений-по-фазам)
4. [Детальный план работ](#детальный-план-работ)
5. [Ожидаемые результаты](#ожидаемые-результаты)
6. [Чек-лист выполнения](#чек-лист-выполнения)

---

## 🎯 Текущее состояние проекта

### Оценка качества: 6/10

**Сильные стороны:**
- ✅ Грамотная архитектура (иерархический Bio → Harmony → Melody)
- ✅ Чистый модульный код
- ✅ Полный CI/CD пайплайн
- ✅ Документация и тесты
- ✅ Web-интерфейс

**Слабые стороны:**
- ❌ Катастрофически малый датасет (30 train pairs)
- ❌ Модель переобучена и генерирует скучную музыку
- ❌ Биологический conditioning не работает
- ❌ Узкий диапазон генерируемых мелодий

### Текущие метрики (small model)

| Метрика | Значение | Проблема |
|---------|----------|----------|
| Train pairs | 30 | Слишком мало для обобщения |
| Pitch range | 14.6 | В 2.3 раза уже baseline (34.2) |
| Unique pitches | 10.8 | В 2 раза меньше baseline (22.8) |
| Pitch entropy | 2.86 | Менее разнообразна чем baseline (3.28) |
| Chord-tone ratio | 60% | Хорошо, но можно лучше |
| Self-similarity | 0.14 | Более повторяющаяся чем baseline (0.11) |

**Диагноз:** Модель запомнила 30 примеров и генерирует безопасные, но скучные мелодии.

---

## 🚨 Критические проблемы

### Проблема 1: Недостаток данных

**Было:**
- 12 биологических последовательностей
- 265 музыкальных сегментов
- 30 обучающих пар

**Решение (✅ ВЫПОЛНЕНО):**
- Скачано 5 референсных геномов (433k фрагментов)
- Используется POP909 + maestro (4,174 MIDI, ~15k-20k сегментов)
- Ожидается 50k-100k обучающих пар

### Проблема 2: Биологический conditioning не работает

**Симптомы:**
- Все генерации имеют почти одинаковые `calibrated_profile`
- Модель игнорирует биологические признаки

**Причина:**
- Слишком мало данных для обучения связи bio → music
- Простая проекция bio вектора недостаточна

**Решение:**
- Увеличить датасет (✅ выполнено)
- Включить ESM embeddings для семантических признаков
- Добавить contrastive learning для bio-music pairing
- Провести ablation study для проверки

### Проблема 3: Модель слишком простая

**Текущая архитектура:**
- d_model: 256
- n_layers: 4
- batch_size: 4

**Решение:**
- Увеличить capacity: d_model=512, n_layers=8
- Увеличить batch_size=32 для стабильности
- Добавить regularization (dropout=0.2, weight_decay=0.05)

---

## 📅 План улучшений по фазам

### ✅ Фаза 1: Расширение датасета (ЗАВЕРШЕНА)

**Срок:** 2026-05-05  
**Статус:** ✅ Выполнено

**Задачи:**
- [x] Скачать референсные геномы (E.coli, yeast, fly, worm, arabidopsis)
- [x] Проверить наличие MIDI корпусов (POP909, maestro)
- [x] Создать конфиг pipeline_v2_large.json
- [x] Создать документацию

**Результат:**
- 433,473 биологических фрагмента
- ~15,000-20,000 музыкальных сегментов
- Ожидается 50k-100k обучающих пар

---

### 🔄 Фаза 2: Проверка и подготовка (В ПРОЦЕССЕ)

**Срок:** 2026-05-05  
**Статус:** 🔄 В процессе

**Задачи:**
- [ ] Проверить конфигурацию pipeline_v2_large.json
- [ ] Сгенерировать dataset report
- [ ] Проверить, что ESM модель доступна
- [ ] Убедиться в наличии GPU с достаточной памятью

**Ожидаемый результат:**
- Подтверждение ~15k-20k музыкальных сегментов
- Подтверждение ~433k биологических фрагментов
- Готовность к обучению

---

### 🎓 Фаза 3: Обучение базовой модели

**Срок:** 2-3 дня (на A100)  
**Статус:** ⏳ Ожидает

**Задачи:**
- [ ] Обучить модель на полном датасете (50 эпох)
- [ ] Мониторить метрики (loss, GPU usage)
- [ ] Сохранить checkpoints
- [ ] Провести smoke-test генерацию

**Требования:**
- GPU: A100 (40GB) или H100
- RAM: 64GB+
- Диск: 15GB свободного места
- Время: 24-48 часов

**Команда:**
```powershell
.\.venv\Scripts\python.exe train_bio_music_v2.py `
  --config configs\pipeline_v2_large.json
```

---

### 📊 Фаза 4: Evaluation и анализ

**Срок:** 1 день  
**Статус:** ⏳ Ожидает

**Задачи:**
- [ ] Запустить evaluation на тестовом наборе
- [ ] Сравнить метрики с baseline
- [ ] Сгенерировать музыку из всех 5 геномов
- [ ] Проверить разнообразие генераций

**Целевые метрики:**
- Pitch range: >25 (было 14.6)
- Unique pitches: >18 (было 10.8)
- Pitch entropy: >3.2 (было 2.86)
- Chord-tone ratio: >70% (было 60%)
- Self-similarity: <0.10 (было 0.14)

**Команда:**
```powershell
.\.venv\Scripts\python.exe tools\evaluate_structured_v2.py `
  --checkpoint results\v2_large_dataset\checkpoints\structured_pipeline.pt `
  --fasta data\fasta\refseq_genomes\GCF_000005845.2_genomic.fna `
  --output-dir results\v2_large_evaluation `
  --max-records 20
```

---

### 🔬 Фаза 5: Ablation Study

**Срок:** 3-4 дня  
**Статус:** ⏳ Ожидает

**Цель:** Проверить, что биологический conditioning действительно работает.

**Задачи:**
- [ ] Обучить модель БЕЗ bio conditioning (no_bio)
- [ ] Обучить модель с RANDOM bio vectors (random_bio)
- [ ] Обучить модель с REAL bio vectors (real_bio) — уже есть
- [ ] Сравнить метрики всех трех моделей

**Ожидаемый результат:**
- `real_bio` должна быть лучше `random_bio`
- `real_bio` должна генерировать разную музыку для разных организмов
- Если `real_bio` ≈ `random_bio`, то conditioning не работает

**Варианты моделей:**

1. **no_bio** (baseline без биологии)
```json
// configs/pipeline_v2_large_no_bio.json
"bio": {
  "embedding_dim": 0  // отключить bio conditioning
}
```

2. **random_bio** (случайные векторы)
```python
# Модифицировать structured_train.py
bio_vector = torch.randn_like(bio_vector)
```

3. **real_bio** (настоящие биологические признаки)
```json
// configs/pipeline_v2_large.json — уже есть
```

---

### 🚀 Фаза 6: Улучшение архитектуры

**Срок:** 1-2 недели  
**Статус:** ⏳ Ожидает

**Задачи:**

#### 6.1 Cross-Attention между Bio и Music
```python
class BioMusicCrossAttention(nn.Module):
    def forward(self, music_hidden, bio_vector):
        # Позволить модели выборочно "смотреть" на bio признаки
        return cross_attn(query=music_hidden, key=bio_vector, value=bio_vector)
```

#### 6.2 Contrastive Learning для Pairing
```python
# Добавить InfoNCE loss
contrastive_loss = InfoNCE(bio_vector, music_embedding)
total_loss = generation_loss + 0.1 * contrastive_loss
```

#### 6.3 Variational Conditioning (VAE)
```python
mu, logvar = self.bio_encoder(bio_vector)
z = reparameterize(mu, logvar)
kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
```

#### 6.4 Data Augmentation
```python
# Для музыки
- Транспозиция (±6 полутонов)
- Tempo scaling (0.8x - 1.2x)
- Octave shift

# Для биологии
- Reverse complement (DNA/RNA)
- Circular permutation
- Sliding window с меньшим stride
```

---

### 🎼 Фаза 7: Accompaniment Generation

**Срок:** 1 неделя  
**Статус:** ⏳ Ожидает

**Цель:** Добавить третью модель для генерации аккомпанемента.

**Архитектура:**
```
Bio + Harmony + Melody → Bass + Drums + Chord Voicing
```

**Задачи:**
- [ ] Извлечь bass и drums из MIDI корпуса
- [ ] Обучить accompaniment model
- [ ] Интегрировать в пайплайн
- [ ] Рендерить 4-дорожечный MIDI

---

### 📝 Фаза 8: Документация и публикация

**Срок:** 1 неделя  
**Статус:** ⏳ Ожидает

**Задачи:**
- [ ] Написать научную статью
- [ ] Создать демо-видео
- [ ] Подготовить презентацию
- [ ] Опубликовать код на GitHub
- [ ] Создать интерактивное демо

---

## 🛠️ Детальный план работ

### Этап 2.1: Проверка конфигурации

**Время:** 10 минут

```powershell
# Проверить пути к данным
Test-Path "C:\Users\vlasi\Documents\biosonification\data\fasta\refseq_genomes"
Test-Path "C:\Users\vlasi\Documents\biosonification\data\midi\POP909"
Test-Path "C:\Users\vlasi\Documents\biosonification\data\midi\maestro-v3.0.0"

# Проверить конфиг
Get-Content configs\pipeline_v2_large.json | ConvertFrom-Json

# Проверить GPU
nvidia-smi
```

**Ожидаемый результат:**
- Все пути существуют
- Конфиг валиден
- GPU доступен

---

### Этап 2.2: Генерация Dataset Report

**Время:** 30-60 минут

```powershell
cd C:\Users\vlasi\Documents\biosonification

.\.venv\Scripts\python.exe tools\report_structured_dataset.py `
  --config configs\pipeline_v2_large.json `
  --output-dir results\v2_large_dataset_report
```

**Что делает:**
1. Загружает все FASTA файлы из `refseq_genomes/`
2. Извлекает биологические фрагменты (fragment_length=1800, stride=900)
3. Загружает все MIDI файлы из POP909 и maestro
4. Извлекает музыкальные сегменты (bars_per_segment=4, hop=2)
5. Строит bio-music pairing
6. Сохраняет manifest

**Ожидаемый результат:**
```
results/v2_large_dataset_report/
├── dataset_report.json
└── dataset_report.md
```

**Ключевые метрики в отчете:**
- `n_bio_sequences`: 5
- `n_bio_fragments`: ~433,473
- `n_music_segments`: ~15,000-20,000
- `n_train_pairs`: ~50,000-100,000
- `n_val_pairs`: ~5,000-10,000
- `n_test_pairs`: ~2,500-5,000

---

### Этап 2.3: Проверка ESM модели

**Время:** 5-10 минут

```powershell
# Проверить, что ESM модель может загрузиться
.\.venv\Scripts\python.exe -c @"
from transformers import AutoModel, AutoTokenizer
import torch

model_name = 'facebook/esm2_t33_650M_UR50D'
print(f'Loading {model_name}...')

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

if torch.cuda.is_available():
    model = model.cuda()
    print('Model loaded on GPU')
else:
    print('Model loaded on CPU')

print('ESM model ready!')
"@
```

**Ожидаемый результат:**
- Модель загружается без ошибок
- Если GPU доступен, модель на GPU

---

### Этап 3: Обучение модели

**Время:** 24-48 часов на A100

#### 3.1 Подготовка

```powershell
# Создать директорию для результатов
New-Item -ItemType Directory -Path "results\v2_large_dataset" -Force

# Проверить свободное место
Get-PSDrive C | Select-Object Used,Free
```

#### 3.2 Запуск обучения

```powershell
# Полное обучение
.\.venv\Scripts\python.exe train_bio_music_v2.py `
  --config configs\pipeline_v2_large.json

# Или быстрый тест (5% данных, 5 эпох)
# Отредактировать configs/pipeline_v2_large.json:
# "max_fragments_per_record": 10
# "num_epochs": 5
```

#### 3.3 Мониторинг

```powershell
# Терминал 1: Смотреть логи
Get-Content results\v2_large_dataset\train.log -Wait

# Терминал 2: Смотреть метрики
while ($true) {
    Clear-Host
    Get-Content results\v2_large_dataset\metrics.json | ConvertFrom-Json | Format-List
    Start-Sleep -Seconds 10
}

# Терминал 3: Смотреть GPU
nvidia-smi -l 1
```

#### 3.4 Ожидаемые метрики

**Хорошие признаки:**
- `harmony_loss`: 1.5 → 0.3-0.4 (50 эпох)
- `melody_loss`: 1.8 → 0.4-0.5 (60 эпох)
- GPU utilization: >80%
- Нет NaN или Inf

**Плохие признаки:**
- Loss не падает после 5 эпох → проблема с learning rate
- GPU utilization <50% → увеличить batch_size
- Out of memory → уменьшить batch_size или d_model

---

### Этап 4: Evaluation

**Время:** 2-3 часа

```powershell
# Evaluation на тестовом наборе
.\.venv\Scripts\python.exe tools\evaluate_structured_v2.py `
  --checkpoint results\v2_large_dataset\checkpoints\structured_pipeline.pt `
  --fasta data\fasta\refseq_genomes\GCF_000005845.2_genomic.fna `
  --output-dir results\v2_large_evaluation `
  --max-records 20 `
  --device auto
```

**Анализ результатов:**

```powershell
# Прочитать отчет
Get-Content results\v2_large_evaluation\evaluation_report.md

# Сравнить с baseline
$report = Get-Content results\v2_large_evaluation\evaluation_report.json | ConvertFrom-Json
$model = $report.aggregates.structured_v2.metrics
$baseline = $report.aggregates.random_baseline.metrics

Write-Host "Pitch range: $($model.pitch_range.mean) vs $($baseline.pitch_range.mean)"
Write-Host "Unique pitches: $($model.unique_pitches.mean) vs $($baseline.unique_pitches.mean)"
Write-Host "Chord-tone ratio: $($model.chord_tone_ratio.mean) vs $($baseline.chord_tone_ratio.mean)"
```

---

### Этап 5: Ablation Study

**Время:** 3-4 дня

#### 5.1 Обучить no_bio модель

```json
// configs/pipeline_v2_large_no_bio.json
{
  "output_dir": "results/v2_large_no_bio",
  "bio": {
    "embedding_dim": 0  // отключить bio
  }
}
```

```powershell
.\.venv\Scripts\python.exe train_bio_music_v2.py `
  --config configs\pipeline_v2_large_no_bio.json
```

#### 5.2 Обучить random_bio модель

```python
# Создать configs/pipeline_v2_large_random_bio.json
# Модифицировать bio_music_pipeline/v2/structured_train.py:

# В функции обучения добавить:
if config.get("use_random_bio", False):
    bio_vector = torch.randn_like(bio_vector)
```

```powershell
.\.venv\Scripts\python.exe train_bio_music_v2.py `
  --config configs\pipeline_v2_large_random_bio.json
```

#### 5.3 Сравнить результаты

```powershell
# Сгенерировать из одного и того же генома всеми тремя моделями
$fasta = "data\fasta\refseq_genomes\GCF_000005845.2_genomic.fna"

# real_bio
.\.venv\Scripts\python.exe generate_from_fasta_v2.py `
  --checkpoint results\v2_large_dataset\checkpoints\structured_pipeline.pt `
  --fasta $fasta --output results\ablation\real_bio.mid

# no_bio
.\.venv\Scripts\python.exe generate_from_fasta_v2.py `
  --checkpoint results\v2_large_no_bio\checkpoints\structured_pipeline.pt `
  --fasta $fasta --output results\ablation\no_bio.mid

# random_bio
.\.venv\Scripts\python.exe generate_from_fasta_v2.py `
  --checkpoint results\v2_large_random_bio\checkpoints\structured_pipeline.pt `
  --fasta $fasta --output results\ablation\random_bio.mid
```

**Анализ:**
- Если `real_bio` лучше `random_bio` → conditioning работает ✅
- Если `real_bio` ≈ `random_bio` → conditioning НЕ работает ❌

---

## 📈 Ожидаемые результаты

### После Фазы 3 (Обучение базовой модели)

| Метрика | Было (small) | Ожидается (large) | Улучшение |
|---------|--------------|-------------------|-----------|
| Pitch range | 14.6 | 25+ | +71% |
| Unique pitches | 10.8 | 18+ | +67% |
| Pitch entropy | 2.86 | 3.2+ | +12% |
| Chord-tone ratio | 60% | 70%+ | +17% |
| Self-similarity | 0.14 | <0.10 | -29% |

### После Фазы 5 (Ablation Study)

**Если bio conditioning работает:**
- `real_bio` генерирует разную музыку для E.coli vs yeast
- `real_bio` метрики лучше `random_bio`
- Корреляция между bio features и music features >0.3

**Если НЕ работает:**
- Все модели генерируют похожую музыку
- `real_bio` ≈ `random_bio` по метрикам
- Нужна Фаза 6 (улучшение архитектуры)

### После Фазы 6 (Улучшение архитектуры)

- Cross-attention: +10-15% к разнообразию
- Contrastive learning: bio-music correlation >0.5
- VAE: более плавные переходы между организмами
- Data augmentation: +20-30% к размеру датасета

---

## ✅ Чек-лист выполнения

### Фаза 1: Расширение датасета ✅

- [x] Создан скрипт `download_refseq_genomes.py`
- [x] Скачано 5 геномов (372 MB, ~433k фрагментов)
- [x] Проверено наличие MIDI (4,174 файла)
- [x] Создан `configs/pipeline_v2_large.json`
- [x] Создана документация (DATA_SUMMARY.md, LARGE_DATASET_QUICKSTART.md)

### Фаза 2: Проверка и подготовка 🔄

- [ ] Проверена конфигурация
- [ ] Сгенерирован dataset report
- [ ] Проверена доступность ESM модели
- [ ] Проверено наличие GPU

### Фаза 3: Обучение базовой модели ⏳

- [ ] Запущено обучение на полном датасете
- [ ] Мониторинг метрик
- [ ] Сохранены checkpoints
- [ ] Проведен smoke-test

### Фаза 4: Evaluation ⏳

- [ ] Запущен evaluation
- [ ] Сравнены метрики с baseline
- [ ] Сгенерирована музыка из всех 5 геномов
- [ ] Проверено разнообразие

### Фаза 5: Ablation Study ⏳

- [ ] Обучена no_bio модель
- [ ] Обучена random_bio модель
- [ ] Сравнены все три модели
- [ ] Подтверждено, что bio conditioning работает

### Фаза 6: Улучшение архитектуры ⏳

- [ ] Добавлен cross-attention
- [ ] Добавлен contrastive learning
- [ ] Добавлен VAE conditioning
- [ ] Реализован data augmentation

### Фаза 7: Accompaniment ⏳

- [ ] Извлечены bass и drums
- [ ] Обучена accompaniment model
- [ ] Интегрирована в пайплайн

### Фаза 8: Публикация ⏳

- [ ] Написана статья
- [ ] Создано демо-видео
- [ ] Подготовлена презентация
- [ ] Опубликован код

---

## 🎯 Критерии успеха

### Минимальный успех (MVP)

- ✅ Датасет расширен до 50k+ пар
- ✅ Модель обучена без ошибок
- ✅ Метрики лучше baseline
- ✅ Генерирует валидный MIDI

### Хороший успех

- ✅ Pitch range >25
- ✅ Unique pitches >18
- ✅ Chord-tone ratio >70%
- ✅ Bio conditioning работает (ablation study)

### Отличный успех

- ✅ Все метрики выше baseline
- ✅ Разные организмы → разная музыка
- ✅ Bio-music correlation >0.5
- ✅ Accompaniment generation работает

---

## 📞 Контакты и ресурсы

### Документация проекта

- [README.md](README.md)
- [DATA_SUMMARY.md](DATA_SUMMARY.md)
- [LARGE_DATASET_QUICKSTART.md](docs/LARGE_DATASET_QUICKSTART.md)
- [STAGE_1_COMPLETE.md](STAGE_1_COMPLETE.md)

### Внешние ресурсы

- [NCBI RefSeq](https://ftp.ncbi.nlm.nih.gov/genomes/refseq/)
- [POP909 Dataset](https://github.com/music-x-lab/POP909-Dataset)
- [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)
- [ESM Models](https://github.com/facebookresearch/esm)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## 📝 История изменений

| Дата | Версия | Изменения |
|------|--------|-----------|
| 2026-05-05 | 1.0 | Создан план улучшений |
| 2026-05-05 | 1.0 | Завершена Фаза 1 (расширение датасета) |

---

**Следующий шаг:** Фаза 2 — Проверка и подготовка к обучению
