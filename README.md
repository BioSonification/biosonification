# BioSonification

[![CI](https://github.com/username/biosonification/workflows/CI/badge.svg)](https://github.com/username/biosonification/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Проект генерирует музыку из биологических последовательностей (ДНК, РНК, белки). Система анализирует биологические признаки и создает двухдорожечную композицию: гармонию и мелодию.

## Как это работает

```
FASTA файл → Биологические признаки → Нейросеть → MIDI файл
```

Система работает в три этапа:

1. **Анализ биологии**: извлекает характеристики последовательности (состав, структура, белковые свойства)
2. **Генерация гармонии**: создает аккордовую сетку на основе биологических признаков
3. **Генерация мелодии**: создает мелодическую линию поверх гармонии

Для длинных последовательностей система автоматически разбивает их на фрагменты и генерирует музыку для каждого, сохраняя стабильное качество.

## Быстрый старт

### За 10 минут

Полное руководство для начинающих: [Быстрый старт](docs/quickstart.md)

Краткая версия:

```powershell
# 1. Установка
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Генерация музыки
.\.venv\Scripts\python.exe generate_from_fasta_v2_fragmented.py `
  --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
  --fasta data\fasta\refseq_genomes\GCF_000005845.2_genomic.fna `
  --output my_music.mid

# 3. Веб-интерфейс (опционально)
.\.venv\Scripts\python.exe -m web.app
# Откройте http://localhost:5001
```

### Подробные руководства

- [Генерация музыки](docs/guides/generation.md) — все способы генерации
- [Обучение модели](docs/getting-started/training-quickstart.md) — обучение на своих данных
- [Веб-интерфейс](web/README.md) — работа через браузер
- [Полная установка](docs/getting-started/run-from-scratch.md) — детальная инструкция

## Доступные модели

Проект включает две обученные модели:

| Модель | Качество | Длина фрагмента | Рекомендация |
|--------|----------|-----------------|--------------|
| 4-тактовая | Лучше (val_loss 0.145/0.157) | 4 такта | Рекомендуется |
| 8-тактовая | Хорошо (val_loss 0.179/0.215) | 8 тактов | Альтернатива |

4-тактовая модель показывает лучшее качество и больше разнообразия при генерации.

## Примеры

Сгенерируйте музыку из генома E. coli:

```powershell
.\.venv\Scripts\python.exe generate_from_fasta_v2_fragmented.py `
  --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
  --fasta data\fasta\refseq_genomes\GCF_000005845.2_genomic.fna `
  --output ecoli_music.mid `
  --bars-per-fragment 4
```

Для последовательности длиной 10000 bp система создаст композицию из 24 тактов (6 фрагментов по 4 такта).

## Документация

### Начало работы
- [Быстрый старт за 10 минут](docs/quickstart.md) — для новичков
- [Запуск с нуля](docs/getting-started/run-from-scratch.md) — полная инструкция
- [Быстрый старт обучения](docs/getting-started/training-quickstart.md) — обучение модели
- [FAQ и Troubleshooting](docs/faq.md) — частые вопросы и решение проблем

### Руководства
- [Руководство по генерации](docs/guides/generation.md) — генерация музыки из геномов
- [Руководство по данным](docs/guides/data-guide.md) — биологические и музыкальные данные
- [Руководство по предобработке](docs/guides/preprocessing-guide.md) — кэширование и оптимизация

### Техническая документация
- [Архитектура и методология](docs/technical/architecture-and-science.md) — научное обоснование
- [Разбор кода](docs/technical/code-walkthrough.md) — описание модулей
- [Структура проекта](docs/technical/project-structure.md) — карта файлов

### Развертывание
- [Production развертывание](docs/deployment/README.md) — запуск в production
- [HTTPS настройка](docs/deployment/https-setup.md) — публичный доступ

### Исследование
- [Сводка экспериментов](docs/thesis/experiment-summary.md) — результаты экспериментов
- [Полный текст диплома](docs/thesis/diploma-work.md) — дипломная работа

Полная навигация: [docs/README.md](docs/README.md)

## Требования

- Python 3.10+
- NVIDIA GPU с CUDA (для обучения и быстрой генерации)
- 6+ GB VRAM (проверено на RTX 2060)
- Windows 10/11 (основная платформа разработки)

Генерация работает и на CPU, но медленнее.

## Разработка

### Запуск тестов

```powershell
# Все тесты
.\.venv\Scripts\python.exe -m pytest

# С покрытием
.\.venv\Scripts\python.exe -m pytest --cov

# Конкретный модуль
.\.venv\Scripts\python.exe -m pytest tests/test_v2_pipeline.py
```

### Проверка кода

```powershell
# Форматирование
.\.venv\Scripts\python.exe -m black .
.\.venv\Scripts\python.exe -m isort .

# Линтинг
.\.venv\Scripts\python.exe -m flake8 bio_music_pipeline web tests

# Проверка типов
.\.venv\Scripts\python.exe -m mypy bio_music_pipeline web --ignore-missing-imports
```

## Ограничения

- Система генерирует музыку в стиле обучающего корпуса (POP909, MAESTRO)
- Биологические признаки используются как управляющие сигналы, но не доказывают причинную связь между генами и музыкой
- Текущая версия создает только гармонию и мелодию (без аккомпанемента)
- ESM embeddings отключены по умолчанию для экономии памяти GPU

## Лицензия

MIT License. См. [LICENSE](LICENSE) для деталей.

## Благодарности

- [POP909 Dataset](https://github.com/music-x-lab/POP909-Dataset) — музыкальный корпус
- [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro) — классическая музыка
- [NCBI RefSeq](https://ftp.ncbi.nlm.nih.gov/genomes/refseq/) — референсные геномы
- [Biopython](https://biopython.org/) — биоинформатические инструменты
- [music21](http://web.mit.edu/music21/) — музыкальный анализ
