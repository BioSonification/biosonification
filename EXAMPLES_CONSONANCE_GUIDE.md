# Руководство: Добавление оценки консонансности к примерам

## Краткий обзор

Вы можете добавить оценку консонансности к примерам в галерее композиций. Это позволит пользователям видеть, насколько консонансными звучат композиции из разных организмов.

## Пошаговая инструкция

### Шаг 1: Подготовьте MIDI файлы примеров

Поместите MIDI файлы в директорию:
```
web/static/examples/midi/
```

Имена файлов должны совпадать с `midi_filename` в `web/examples_data.py`:
- `ecoli_adaptive.mid`
- `yeast_adaptive.mid`
- `drosophila_adaptive.mid`
- `worm_adaptive.mid`
- `arabidopsis_adaptive.mid`
- `drosophila_very_long.mid`

### Шаг 2: Классифицируйте примеры

Выполните команду:
```bash
cd /Users/aloha_kuino/Desktop/biosonification_rate_music
source venv/bin/activate
python -m web.classify_examples
```

**Вывод скрипта:**
```
🎵 Найдено 6 MIDI файлов

Классифицирую: ecoli_adaptive.mid... ✅ CONSONANT (87%)
Классифицирую: yeast_adaptive.mid... ✅ CONSONANT (92%)
Классифицирую: drosophila_adaptive.mid... ✅ DISSONANT (64%)
...

📊 Результаты классификации:
============================================================

ecoli_adaptive:
  Предсказание: consonant
  Уверенность:  87.2%
  Консонансность: 87.2%
  Диссонансность: 12.8%
...

💾 Результаты сохранены в web/static/examples/consonance_ratings.json

📝 Добавьте консонансность к примерам в examples_data.py:
============================================================

Примеры для копирования:

    # ecoli_adaptive
    "consonance": {
        "prediction": "consonant",
        "confidence": 0.8723,
        "consonant_score": 0.8723,
        "dissonant_score": 0.1277
    }

    # yeast_adaptive
    "consonance": {
        "prediction": "consonant",
        "confidence": 0.9234,
        ...
```

### Шаг 3: Обновите examples_data.py

Скопируйте вывод скрипта и добавьте `consonance` объект к каждому примеру в `web/examples_data.py`:

**До:**
```python
{
    "id": "ecoli",
    "organism": "E. coli",
    "scientific_name": "Escherichia coli",
    "description": "Bacterial genome, 1800 bp fragment",
    "midi_filename": "ecoli_adaptive.mid",
    "bars": 9,
    "duration_seconds": 18,
    "genome_size": "4.6 MB",
    "icon": "🦠",
    "consonance": None,
}
```

**После:**
```python
{
    "id": "ecoli",
    "organism": "E. coli",
    "scientific_name": "Escherichia coli",
    "description": "Bacterial genome, 1800 bp fragment",
    "midi_filename": "ecoli_adaptive.mid",
    "bars": 9,
    "duration_seconds": 18,
    "genome_size": "4.6 MB",
    "icon": "🦠",
    "consonance": {
        "prediction": "consonant",
        "confidence": 0.8723,
        "consonant_score": 0.8723,
        "dissonant_score": 0.1277
    }
}
```

### Шаг 4: Перезагрузите приложение

```bash
# Остановите текущий сервер (если он запущен)
pkill -f "python -m web.app"

# Запустите снова
source venv/bin/activate
python -m web.app
```

### Шаг 5: Проверьте результаты

1. Откройте браузер на `http://localhost:5001`
2. Перейдите на вкладку **"Примеры"**
3. Проверьте, что каждый пример показывает:
   - 🎵 иконку с оценкой консонансности
   - Зелёный или красный badge с CONSONANT/DISSONANT
   - Процент уверенности модели

## Интеграция с фронтенд

Обновлённый фронтенд автоматически отобразит:

### В карточке примера:
```
┌─────────────────────────────┐
│ 🦠  E. coli                 │
│     Escherichia coli        │
│                             │
│ Bacterial genome, 1800 bp...│
│                             │
│ [Audio Player]              │
│                             │
│ 📊 9 bars  ⏱️ ~18s  🎵 Consonant │
│                             │
│ ┌─────────────────────────┐ │
│ │ CONSONANT     87%       │ │
│ └─────────────────────────┘ │
│                             │
│ [Download MIDI]             │
└─────────────────────────────┘
```

### CSS стили для консонансности:
- ✅ **Consonant** (зелёный) - rgba(16, 185, 129, 0.2)
- ❌ **Dissonant** (красный) - rgba(239, 68, 68, 0.2)

## Структура consonance_ratings.json

Скрипт сохраняет результаты в `web/static/examples/consonance_ratings.json`:

```json
{
  "ecoli_adaptive": {
    "prediction": "consonant",
    "confidence": 0.8723,
    "scores": {
      "consonant": 0.8723,
      "dissonant": 0.1277
    }
  },
  "yeast_adaptive": {
    "prediction": "consonant",
    "confidence": 0.9234,
    "scores": {
      "consonant": 0.9234,
      "dissonant": 0.0766
    }
  },
  ...
}
```

Этот файл служит для справки и может быть использован для анализа паттернов консонансности.

## Автоматизация процесса

Если вы хотите автоматизировать добавление примеров, можно модифицировать `web/examples_data.py` для загрузки консонансности из JSON:

```python
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
CONSONANCE_FILE = BASE_DIR / "static" / "examples" / "consonance_ratings.json"

def load_consonance_ratings():
    """Load consonance ratings from JSON file."""
    if CONSONANCE_FILE.exists():
        with open(CONSONANCE_FILE) as f:
            return json.load(f)
    return {}

_consonance_ratings = load_consonance_ratings()

EXAMPLES = [
    {
        "id": "ecoli",
        ...
        "consonance": _consonance_ratings.get("ecoli_adaptive"),
    },
    ...
]
```

## Использование API

Примеры также выводятся через `/api/examples`:

```bash
curl http://localhost:5001/api/examples
```

Ответ включает консонансность:
```json
{
  "examples": [
    {
      "id": "ecoli",
      "organism": "E. coli",
      ...
      "consonance": {
        "prediction": "consonant",
        "confidence": 0.8723,
        "consonant_score": 0.8723,
        "dissonant_score": 0.1277
      }
    },
    ...
  ]
}
```

## Производительность

- **Классификация одного примера**: 0.1-0.5 сек
- **Классификация 6 примеров**: 1-3 сек
- **Фронтенд отрисовка**: мгновенно

## Обработка ошибок

### Если модель не найдена:
```
❌ Классификатор не готов: Model not found at ...
```

**Решение:** Убедитесь, что обученная модель находится по пути:
```
/Users/aloha_kuino/Desktop/diploma/crnn_music_classifier/models/best_crnn.pth
```

### Если MIDI файлы не найдены:
```
⚠️  MIDI файлы не найдены в web/static/examples/midi/
   Поместите MIDI файлы в эту папку и запустите скрипт снова
```

**Решение:** Создайте директорию и добавьте MIDI файлы:
```bash
mkdir -p web/static/examples/midi/
# Скопируйте MIDI файлы сюда
```

### Если классификация не удаётся:
```
Классифицирую: file.mid... ❌ Ошибка: Classification failed: ...
```

**Решение:** 
- Проверьте, что MIDI файл не повреждён
- Попробуйте с другим MIDI файлом
- Проверьте логи Flask приложения

## Примеры команд

### Классифицировать примеры и показать результаты:
```bash
cd /Users/aloha_kuino/Desktop/biosonification_rate_music
source venv/bin/activate
python -m web.classify_examples
```

### Просмотреть сохранённые результаты:
```bash
cat web/static/examples/consonance_ratings.json | python -m json.tool
```

### Перезагрузить приложение:
```bash
pkill -f "python -m web.app"
source venv/bin/activate
python -m web.app
```
