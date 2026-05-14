# Гайд по мерджу репозиториев

## 📁 Новые файлы (нужно добавить)

```
biosonification_rate_music/
├── web/
│   ├── consonance_classifier.py      ← НОВЫЙ (классификатор CRNN)
│   └── classify_examples.py          ← НОВЫЙ (скрипт классификации)
│
├── classify_examples.py              ← НОВЫЙ (standalone скрипт)
├── CONSONANCE_CLASSIFIER_INTEGRATION.md
├── EXAMPLES_CONSONANCE_GUIDE.md
├── EXAMPLES_IMPLEMENTATION_SUMMARY.md
├── INTEGRATION_FLOW.md
├── DEPLOYMENT_CHECKLIST.md           ← ВЫ ЧИТАЕТЕ СЕЙЧАС
└── MERGE_GUIDE.md
```

## 🔄 Измененные файлы (нужны обновления)

### 1. `web/app.py`
**Строки для добавления:**

```python
# После строки 14 (импорты):
from .consonance_classifier import get_classifier

# В функции generate() - строка ~103:
# Добавить после создания WAV:
classifier = get_classifier()
if classifier.is_ready():
    classification = classifier.classify(midi_path)
    result["consonance"] = {
        "success": classification["success"],
        "prediction": classification["prediction"],
        "confidence": classification["confidence"],
        "scores": classification["scores"],
    }

# В функции status() - строка ~152:
# Добавить после synth_status:
classifier = get_classifier()

# И добавить в возвращаемый jsonify():
"consonance_classifier": {
    "ready": classifier.is_ready(),
    "error": classifier.get_error() if not classifier.is_ready() else None,
},
```

### 2. `web/templates/index.html`
**Добавить перед `<!-- Musical Parameters -->` секцией (строка ~174):**

```html
<!-- Consonance Rating -->
<div class="consonance-section" id="consonance-section">
    <h3>Анализ консонансности</h3>
    <div class="consonance-card">
        <div class="consonance-rating">
            <div class="rating-label" id="consonance-label">Загрузка...</div>
            <div class="rating-score">
                <span id="consonance-prediction" class="prediction-badge">-</span>
                <span id="consonance-confidence" class="confidence-value">-</span>
            </div>
        </div>
        <div class="rating-details" id="consonance-details">
            <div class="score-bar">
                <div class="score-label">Консонансность</div>
                <div class="bar-container">
                    <div class="bar-fill" id="consonant-bar" style="width: 0%"></div>
                </div>
                <div class="score-value" id="consonant-value">0%</div>
            </div>
            <div class="score-bar">
                <div class="score-label">Диссонансность</div>
                <div class="bar-container">
                    <div class="bar-fill dissonant" id="dissonant-bar" style="width: 0%"></div>
                </div>
                <div class="score-value" id="dissonant-value">0%</div>
            </div>
        </div>
    </div>
</div>
```

### 3. `web/static/css/style.css`
**Добавить перед `/* ============================================ Examples Gallery Section */` (строка ~615):**

```css
/* Consonance Rating Section */
.consonance-section {
    margin: 32px 0;
    padding: 24px;
    background: var(--bg-input);
    border-radius: var(--radius);
    border: 1px solid var(--border);
}

.consonance-section h3 {
    margin-bottom: 20px;
    color: var(--text-primary);
}

.consonance-card {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 24px;
    align-items: center;
}

.consonance-rating {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 20px;
    background: var(--bg-secondary);
    border-radius: var(--radius-sm);
    min-width: 200px;
}

.rating-label {
    font-size: 0.875rem;
    color: var(--text-secondary);
    margin-bottom: 12px;
    text-align: center;
}

.rating-score {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
}

.prediction-badge {
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 0.875rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.prediction-badge.consonant {
    background: rgba(16, 185, 129, 0.2);
    color: var(--success);
    border: 1px solid var(--success);
}

.prediction-badge.dissonant {
    background: rgba(239, 68, 68, 0.2);
    color: var(--error);
    border: 1px solid var(--error);
}

.confidence-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-primary);
}

.rating-details {
    display: flex;
    flex-direction: column;
    gap: 16px;
    flex: 1;
}

.score-bar {
    display: grid;
    grid-template-columns: 100px 1fr 60px;
    align-items: center;
    gap: 12px;
}

.score-label {
    font-size: 0.875rem;
    color: var(--text-secondary);
    font-weight: 500;
}

.bar-container {
    background: var(--bg-secondary);
    border-radius: 4px;
    height: 24px;
    overflow: hidden;
    border: 1px solid var(--border);
}

.bar-fill {
    height: 100%;
    background: var(--success);
    transition: width 0.6s ease-out;
}

.bar-fill.dissonant {
    background: var(--error);
}

.score-value {
    text-align: right;
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--text-primary);
    min-width: 50px;
}

@media (max-width: 768px) {
    .consonance-card {
        grid-template-columns: 1fr;
        gap: 16px;
    }

    .consonance-rating {
        min-width: unset;
    }

    .score-bar {
        grid-template-columns: 80px 1fr 50px;
    }
}
```

### 4. `web/static/js/app.js`
**Добавить в DOM elements (строка ~36):**

```javascript
// Consonance section
consonanceSection: document.getElementById('consonance-section'),
```

**Заменить в функции `showResults()` (строка ~217), добавить ПЕРЕД параметрами:**

```javascript
// Consonance rating
if (data.consonance && data.consonance.success) {
    const consonance = data.consonance;
    const predictionBadge = document.getElementById('consonance-prediction');
    const confidenceValue = document.getElementById('consonance-confidence');
    const consonantBar = document.getElementById('consonant-bar');
    const dissonantBar = document.getElementById('dissonant-bar');
    const consonantValue = document.getElementById('consonant-value');
    const dissonantValue = document.getElementById('dissonant-value');
    const consonanceLabel = document.getElementById('consonance-label');

    predictionBadge.textContent = consonance.prediction.toUpperCase();
    predictionBadge.className = `prediction-badge ${consonance.prediction}`;
    confidenceValue.textContent = `${Math.round(consonance.confidence * 100)}%`;

    const consonantScore = (consonance.scores.consonant || 0) * 100;
    const dissonantScore = (consonance.scores.dissonant || 0) * 100;

    consonantBar.style.width = consonantScore + '%';
    dissonantBar.style.width = dissonantScore + '%';
    consonantValue.textContent = consonantScore.toFixed(1) + '%';
    dissonantValue.textContent = dissonantScore.toFixed(1) + '%';

    const predictionText = consonance.prediction === 'consonant'
        ? 'Композиция звучит консонансно'
        : 'Композиция звучит диссонансно';
    consonanceLabel.textContent = predictionText;

    document.getElementById('consonance-section').classList.remove('hidden');
} else if (data.consonance && !data.consonance.success) {
    console.warn('Consonance classification failed:', data.consonance.error);
    document.getElementById('consonance-section').classList.add('hidden');
}
```

### 5. `web/examples_data.py`
**Добавить `"consonance": {...}` для каждого примера:**

```python
{
    "id": "ecoli",
    "organism": "E. coli",
    ...
    "consonance": {
        "prediction": "consonant",
        "confidence": 0.6866,
        "consonant_score": 0.6866,
        "dissonant_score": 0.3134,
    },
}
```

## 🔗 Зависимости для requirements.txt

Убедитесь что в `requirements.txt` есть:

```
torch>=2.0.0
librosa>=0.10.0
pretty_midi>=0.2.9
mido>=1.2.10
numpy>=1.24.0
```

## 📋 Порядок мерджа

1. **Добавить новые файлы:**
   - `web/consonance_classifier.py`
   - `classify_examples.py`
   - `web/classify_examples.py` (опционально)

2. **Обновить существующие файлы:**
   - `web/app.py`
   - `web/templates/index.html`
   - `web/static/css/style.css`
   - `web/static/js/app.js`
   - `web/examples_data.py`
   - `requirements.txt` (добавить librosa, pretty_midi)

3. **Добавить документацию:**
   - Этот MERGE_GUIDE.md
   - DEPLOYMENT_CHECKLIST.md
   - Другие *.md документы

4. **Протестировать:**
   ```bash
   pip install -r requirements.txt
   python -m web.app
   curl http://localhost:5001/api/status
   ```

## 🚀 Git команды для мерджа

```bash
# Если вливаете как pull request:
git checkout main
git pull origin main
git merge feature/consonance-classification
git push origin main

# Или cherry-pick отдельные коммиты:
git cherry-pick commit-hash-1
git cherry-pick commit-hash-2
```

## ✅ После мерджа

- [ ] Все файлы на месте
- [ ] Зависимости установлены
- [ ] Приложение запускается без ошибок
- [ ] API /status показывает "consonance_classifier": {"ready": true}
- [ ] Можно классифицировать примеры: `python3 classify_examples.py`
- [ ] Генерация композиций показывает консонансность

Готово к продакшену! 🎉
