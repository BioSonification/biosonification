# Интеграция CRNN классификатора консонансности

## Описание

На сайт `biosonification_rate_music` интегрирована модель CRNN из проекта `crnn_music_classification` для оценки консонансности каждой сгенерированной композиции.

## Что было изменено

### 1. Backend интеграция (`web/consonance_classifier.py`)
- Новый модуль `ConsonanceClassifier` для загрузки и использования CRNN модели
- Ленивая загрузка зависимостей (PyTorch и другие) для оптимизации памяти
- Graceful handling ошибок при отсутствии зависимостей
- Метод `classify(midi_path)` возвращает:
  - `prediction`: "consonant" или "dissonant"
  - `confidence`: вероятность (0.0-1.0)
  - `scores`: словарь с вероятностями для каждого класса

### 2. API изменения (`web/app.py`)
- Добавлен импорт классификатора
- Модифицирован `/api/generate` endpoint для добавления консонансной оценки в ответ
- Обновлён `/api/status` endpoint для отображения статуса классификатора

### 3. Frontend обновления

#### HTML (`web/templates/index.html`)
- Добавлена новая секция "Анализ консонансности" в результатах
- Отображение:
  - Предсказания (CONSONANT/DISSONANT)
  - Уверенности (%)
  - Графики вероятностей для каждого класса

#### CSS (`web/static/css/style.css`)
- Стили для консонансной секции
- Визуализация confidence bars с анимацией shimmer
- Адаптивный дизайн для мобильных устройств
- Цветовая кодировка: зелёный для консонансности, красный для диссонансности

#### JavaScript (`web/static/js/app.js`)
- Обновлён `showResults()` для отображения консонансной оценки
- Динамическое обновление визуальных элементов
- Обработка ошибок классификации без прерывания основного потока

## Структура ответа API

```json
{
  "success": true,
  "session_id": "abc12345",
  "consonance": {
    "success": true,
    "prediction": "consonant",
    "confidence": 0.87,
    "scores": {
      "consonant": 0.87,
      "dissonant": 0.13
    }
  },
  "musical_params": {...},
  ...
}
```

## Как использовать

### Локальное тестирование
1. Убедитесь, что установлены зависимости для обоих проектов:
   ```bash
   pip install -r /Users/aloha_kuino/Desktop/biosonification_rate_music/requirements.txt
   pip install -r /Users/aloha_kuino/Desktop/diploma/crnn_music_classifier/requirements.txt
   ```

2. Запустите Flask приложение:
   ```bash
   cd /Users/aloha_kuino/Desktop/biosonification_rate_music
   python -m web.app
   ```

3. Откройте браузер на `http://localhost:5001`

4. Сгенерируйте композицию и просмотрите консонансную оценку в секции "Анализ консонансности"

### Обработка ошибок
- Если модель не загружается, классификация будет пропущена, но генерация продолжится
- Ошибки классификации логируются в консоль приложения
- Фронтенд скрывает секцию консонансности при ошибке классификации

## Требования

- PyTorch с поддержкой GPU (опционально, используется CPU если GPU недоступна)
- librosa, mido, pretty_midi (из requirements.txt крупа)
- CRNN модель должна быть обучена и находиться по пути:
  `/Users/aloha_kuino/Desktop/diploma/crnn_music_classifier/models/best_crnn.pth`

## Производительность

- Первая классификация после запуска может занять 1-2 секунды (загрузка модели)
- Последующие классификации обычно занимают 0.1-0.5 секунд
- Модель использует lazy loading для оптимизации памяти приложения
