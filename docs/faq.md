# FAQ и Troubleshooting

Часто задаваемые вопросы и решение распространенных проблем.

## Общие вопросы

### Что делает этот проект?

BioSonification генерирует музыку из биологических последовательностей (ДНК, РНК, белки). Система анализирует биологические признаки и использует их для управления нейросетью, которая создает двухдорожечную MIDI композицию: гармонию и мелодию.

### Нужна ли мне GPU?

Нет, но рекомендуется. Генерация работает на CPU, но медленнее. Для обучения модели GPU значительно ускоряет процесс (часы вместо дней).

### Какие данные нужны для обучения?

- FASTA файлы с биологическими последовательностями
- Полифонический MIDI корпус (например, POP909, MAESTRO)

Проект включает примеры данных для быстрого старта.

### Сколько времени занимает генерация?

- На GPU: 5-30 секунд для последовательности 10000 bp
- На CPU: 1-5 минут для той же последовательности

### Можно ли использовать свои последовательности?

Да, любые FASTA файлы с DNA, RNA или protein последовательностями.

### Какой формат выходного файла?

MIDI файл с двумя дорожками:
- Дорожка 1: гармония (аккорды)
- Дорожка 2: мелодия (монофоническая линия)

## Установка и настройка

### ModuleNotFoundError: No module named 'torch'

**Проблема:** Используется системный Python вместо виртуального окружения.

**Решение:** Всегда используйте `.\.venv\Scripts\python.exe`:

```powershell
# Неправильно
python generate_from_fasta_v2_fragmented.py ...

# Правильно
.\.venv\Scripts\python.exe generate_from_fasta_v2_fragmented.py ...
```

### CUDA out of memory

**Проблема:** Недостаточно GPU памяти.

**Решение 1:** Используйте CPU:

```powershell
.\.venv\Scripts\python.exe generate_from_fasta_v2_fragmented.py `
  --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
  --fasta your_sequence.fasta `
  --output output.mid `
  --device cpu
```

**Решение 2:** Уменьшите batch size в конфиге (для обучения).

### ImportError: DLL load failed

**Проблема:** Несовместимая версия PyTorch или CUDA.

**Решение:** Переустановите PyTorch:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall torch
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

### Тесты не проходят

**Проблема:** Отсутствуют зависимости или неправильная установка.

**Решение:**

```powershell
# Переустановить зависимости
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Запустить тесты
.\.venv\Scripts\python.exe -m pytest tests/test_v2_pipeline.py -v
```

## Генерация музыки

### FileNotFoundError: checkpoint not found

**Проблема:** Файл модели не существует.

**Решение:** Проверьте путь к checkpoint:

```powershell
Test-Path results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt
```

Если файла нет, обучите модель или скачайте предобученную.

### IndexError: record_index is outside the valid range

**Проблема:** Указан несуществующий индекс записи в FASTA файле.

**Решение:** Проверьте количество записей:

```powershell
.\.venv\Scripts\python.exe -c "from Bio import SeqIO; print(len(list(SeqIO.parse('your_file.fasta', 'fasta'))))"
```

Используйте индекс от 0 до (количество записей - 1).

### Checkpoint/config mismatch

**Проблема:** Несоответствие архитектуры модели и конфига.

**Решение:** Используйте правильный конфиг для модели:

```powershell
# Для 4-тактовой модели
--config configs\pipeline_v2_medium_rtx2060_fast.json

# Для 8-тактовой модели
--config configs\pipeline_v2_medium_rtx2060_long.json
```

### Генерация слишком медленная

**Проблема:** Используется CPU или большая последовательность.

**Решение 1:** Используйте GPU:

```powershell
--device cuda
```

**Решение 2:** Уменьшите длину последовательности или используйте меньше фрагментов.

### Музыка звучит странно

**Возможные причины:**
- Модель не обучена или обучена плохо
- Используется неподходящий checkpoint
- Биологическая последовательность слишком короткая или нетипичная

**Решение:**
- Используйте рекомендованную 4-тактовую модель
- Попробуйте другую последовательность
- Проверьте метрики обучения модели

## Обучение модели

### Training loss не уменьшается

**Возможные причины:**
- Слишком высокий learning rate
- Недостаточно данных
- Проблемы с pairing

**Решение:**
- Уменьшите learning rate в конфиге
- Увеличьте размер датасета
- Проверьте отчет по данным

### Validation loss растет (overfitting)

**Проблема:** Модель переобучается.

**Решение:**
- Увеличьте dropout в конфиге
- Уменьшите количество эпох
- Увеличьте размер датасета
- Добавьте регуляризацию

### Out of memory при обучении

**Проблема:** Недостаточно GPU памяти.

**Решение:**
- Уменьшите batch_size в конфиге
- Увеличьте gradient_accumulation_steps
- Используйте mixed_precision: true
- Отключите ESM embeddings (use_esm_embedding: false)

### Предобработка данных слишком долгая

**Проблема:** ViennaRNA или ESM замедляют процесс.

**Решение:** Используйте fast конфиг:

```json
{
  "bio": {
    "use_vienna_rna": false,
    "use_esm_embedding": false,
    "max_fragments_per_record": 10
  }
}
```

Или используйте кэширование:

```powershell
# Предобработка один раз
.\.venv\Scripts\python.exe tools\preprocess_bio.py --config your_config.json --output data\cache\bio.pkl
.\.venv\Scripts\python.exe tools\preprocess_music.py --config your_config.json --output data\cache\music.pkl

# Обучение с кэшем
.\.venv\Scripts\python.exe train_bio_music_v2.py --config your_config.json --bio-cache data\cache\bio.pkl --music-cache data\cache\music.pkl
```

## Веб-интерфейс

### Веб-интерфейс не запускается

**Проблема:** Порт занят или отсутствуют зависимости.

**Решение 1:** Проверьте зависимости:

```powershell
.\.venv\Scripts\python.exe -m pip install flask
```

**Решение 2:** Используйте другой порт:

```powershell
$env:FLASK_RUN_PORT = "5002"
.\.venv\Scripts\python.exe -m web.app
```

### Генерация в веб-интерфейсе не работает

**Проблема:** Модель не найдена или неправильный путь.

**Решение:** Установите переменные окружения:

```powershell
$env:BIOSONIFICATION_STRUCTURED_CHECKPOINT = "C:\full\path\to\structured_pipeline.pt"
$env:BIOSONIFICATION_CONFIG_PATH = "C:\full\path\to\config.json"
.\.venv\Scripts\python.exe -m web.app
```

### Примеры в галерее не воспроизводятся

**Проблема:** Отсутствует fluidsynth или timidity.

**Решение:** Скачайте MIDI файлы и откройте в MIDI-плеере.

## Данные

### FASTA файл не читается

**Проблема:** Неправильный формат или кодировка.

**Решение:**
- Проверьте, что файл начинается с `>`
- Убедитесь, что кодировка UTF-8
- Проверьте расширение файла (.fasta, .fa, .fna)

### MIDI файлы не обрабатываются

**Проблема:** Поврежденные файлы или неподдерживаемый формат.

**Решение:**
- Используйте стандартные MIDI файлы (.mid, .midi)
- Проверьте файлы в MIDI-плеере
- Удалите поврежденные файлы из корпуса

### Недостаточно данных для обучения

**Проблема:** Мало FASTA записей или MIDI файлов.

**Решение:**
- Скачайте дополнительные геномы из NCBI RefSeq
- Добавьте MIDI корпусы (POP909, MAESTRO)
- Увеличьте фрагментацию (уменьшите stride)

## Production развертывание

### Служба не запускается

**Проблема:** Неправильные права или пути.

**Решение:**
- Запустите PowerShell от администратора
- Проверьте пути в скриптах
- Проверьте логи: `Get-Content logs\service-stderr.log -Tail 30`

### HTTPS не работает

**Проблема:** Неправильная настройка Caddy или firewall.

**Решение:**
- Проверьте port forwarding на роутере (80, 443)
- Проверьте firewall: `Get-NetFirewallRule -DisplayName "Caddy*"`
- Проверьте логи Caddy

### Генерация не работает в production

**Проблема:** GPU недоступна в службе.

**Решение:** Переключите на CPU в `.env`:

```
BIOSONIFICATION_DEVICE=cpu
```

Затем перезапустите службу.

## Дополнительная помощь

### Где найти больше информации?

- [Документация](../docs/README.md) — полная документация
- [GitHub Issues](https://github.com/username/biosonification/issues) — сообщить о проблеме
- [Архитектура](technical/architecture-and-science.md) — техническое описание

### Как сообщить о проблеме?

Создайте issue на GitHub с:
- Описанием проблемы
- Шагами для воспроизведения
- Версией Python и PyTorch
- Логами ошибок
- Конфигурацией (без секретов)

### Как внести вклад?

См. [CONTRIBUTING.md](../CONTRIBUTING.md) (если существует) или создайте pull request с:
- Описанием изменений
- Тестами
- Обновленной документацией
