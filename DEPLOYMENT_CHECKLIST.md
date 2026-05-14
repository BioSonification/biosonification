# Чеклист интеграции для продакшена

## 📋 Предусловия

### 1. Структура директорий
```
/production/
├── biosonification_rate_music/      ← Основной репозиторий
│   ├── web/
│   ├── configs/
│   ├── requirements.txt
│   ├── venv/
│   ├── consonance_classifier.py     ← Новая интеграция
│   └── classify_examples.py         ← Новая интеграция
│
└── diploma/crnn_music_classifier/   ← Модель классификации
    ├── models/best_crnn.pth         ← ОБЯЗАТЕЛЬНО наличие!
    ├── configs/
    ├── utils/
    └── requirements.txt
```

## 🔧 Шаг 1: Установка зависимостей

```bash
# Перейти в папку проекта
cd /production/biosonification_rate_music

# Создать виртуальное окружение
python3 -m venv venv
source venv/bin/activate

# Установить все зависимости
pip install -r requirements.txt

# Установить дополнительные зависимости для классификатора
pip install librosa pretty_midi mido scikit-learn

# Проверить что всё установилось
python -c "import torch, librosa, mido; print('✅ All deps OK')"
```

## 🎯 Шаг 2: Проверить пути

### Убедитесь что модель на месте:
```bash
ls -l /production/diploma/crnn_music_classifier/models/best_crnn.pth
# Должен быть файл ~5.7 MB
```

### Обновить пути в коде (если другие директории):

**Файл: `web/consonance_classifier.py`** (строка ~14)
```python
# Если diploma в другом месте:
DIPLOMA_ROOT = Path("/production/diploma")  # ← Обновить путь
```

**Файл: `classify_examples.py`** (строка ~11)
```python
# Обновить абсолютные пути:
BIOSONIFICATION_ROOT = Path("/production/biosonification_rate_music").resolve()
DIPLOMA_ROOT = Path("/production/diploma").resolve()
```

## 📊 Шаг 3: Подготовить примеры (опционально)

Если используете примеры с консонансностью:

```bash
# Разместить MIDI файлы примеров
mkdir -p web/static/examples/midi/
# cp path/to/midi/*.mid web/static/examples/midi/

# Классифицировать примеры
python3 classify_examples.py

# Скопировать результаты в examples_data.py
# (вывод скрипта подскажет что копировать)
```

## 🚀 Шаг 4: Запустить приложение

```bash
source venv/bin/activate

# Развертывание с gunicorn (для продакшена)
gunicorn -w 4 -b 0.0.0.0:5001 'web.app:app'

# ИЛИ локально для тестирования
python -m web.app
```

## 🔐 Шаг 5: Переменные окружения (опционально)

```bash
# .env файл
export BIOSONIFICATION_HOST=0.0.0.0
export BIOSONIFICATION_PORT=5001
export BIOSONIFICATION_DEBUG=false
export BIOSONIFICATION_LOG_LEVEL=INFO

# Если используются нестандартные пути:
export BIOSONIFICATION_STRUCTURED_CHECKPOINT=/path/to/checkpoint.pt
export BIOSONIFICATION_CONFIG_PATH=/path/to/config.json
```

## ✅ Шаг 6: Проверка

### 1. Проверить что классификатор готов:
```bash
curl http://localhost:5001/api/status | jq '.consonance_classifier'

# Должно быть:
# {
#   "ready": true,
#   "error": null
# }
```

### 2. Проверить примеры:
```bash
curl http://localhost:5001/api/examples | jq '.examples[0].consonance'

# Должно быть консонансность данные или null (если не классифицировано)
```

### 3. Тест классификации генерируемой композиции:
```bash
# Если есть модель генерации - сгенерировать композицию
# Проверить что в ответе есть поле "consonance"

curl -X POST http://localhost:5001/api/generate \
  -H "Content-Type: application/json" \
  -d '{"fasta": ">test\nATCGATCG..."}' | jq '.consonance'
```

## ⚠️ Потенциальные проблемы и решения

### Проблема: "No module named 'crnn_music_classifier'"
**Решение:**
```bash
# Проверить что diploma в PYTHONPATH
python -c "import sys; sys.path.insert(0, '/production/diploma'); from crnn_music_classifier.configs.config import BEST_MODEL_PATH; print(BEST_MODEL_PATH)"

# Если не работает - обновить пути в consonance_classifier.py
```

### Проблема: "Model not found at /path/to/model"
**Решение:**
```bash
# Проверить что модель на месте
find /production/diploma -name "best_crnn.pth"

# Если нет - тренировать модель или скопировать из другого места
```

### Проблема: "Failed to import librosa"
**Решение:**
```bash
source venv/bin/activate
pip install librosa
```

### Проблема: Медленная первая классификация
**Это нормально!** Первая классификация загружает модель (~1-2 сек), потом она кэшируется в памяти.

## 📈 Масштабирование

### Для высоких нагрузок:

```bash
# Использовать gunicorn с несколькими рабочими процессами
gunicorn -w 4 -b 0.0.0.0:5001 \
  --timeout 60 \
  --access-logfile - \
  'web.app:app'

# Или через nginx + gunicorn
```

### Оптимизация памяти:

Классификатор использует ~2-3 GB памяти при загрузке модели.

Рекомендуемые specs:
- **CPU**: 2+ cores
- **RAM**: 8+ GB (4 GB минимум)
- **Disk**: 1 GB (для моделей + логи)
- **GPU**: Optional (ускорит классификацию в 5-10x раз)

## 🎯 Финальный чеклист

- [ ] Оба репозитория на месте (biosonification_rate_music + diploma)
- [ ] Установлены все зависимости (pip install -r requirements.txt)
- [ ] Модель best_crnn.pth есть в diploma/crnn_music_classifier/models/
- [ ] Пути обновлены в consonance_classifier.py и classify_examples.py
- [ ] MIDI примеры классифицированы (опционально)
- [ ] examples_data.py обновлен с consonance данными
- [ ] Приложение запущено и отвечает на /api/status
- [ ] Консонансность отображается в /api/examples
- [ ] Тестирована генерация (если модель доступна)

## 📞 Команды для быстрого старта

```bash
# 1. Установка
cd /production/biosonification_rate_music
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install librosa pretty_midi mido scikit-learn

# 2. Тест классификатора
python -c "from web.consonance_classifier import get_classifier; c = get_classifier(); print('Ready!' if c.is_ready() else c.get_error())"

# 3. Запуск приложения
python -m web.app

# 4. Проверка
curl http://localhost:5001/api/status | jq '.consonance_classifier'
```

## 🚀 Готово!

Если все чекпойнты пройдены - интеграция завершена и приложение готово к использованию! 🎉
