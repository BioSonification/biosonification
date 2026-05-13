# Быстрый старт

Это руководство поможет вам начать работу с BioSonification за 10 минут.

## Что вам понадобится

- Python 3.10 или новее
- NVIDIA GPU с CUDA (опционально, но рекомендуется)
- 10 GB свободного места на диске

## Шаг 1: Установка

Клонируйте репозиторий и создайте виртуальное окружение:

```powershell
git clone https://github.com/username/biosonification.git
cd biosonification

# Создать виртуальное окружение
python -m venv .venv

# Установить зависимости
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Шаг 2: Проверка установки

Убедитесь, что CUDA доступна (если используете GPU):

```powershell
@'
import torch
print("PyTorch:", torch.__version__)
print("CUDA доступна:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
'@ | .\.venv\Scripts\python.exe -
```

Запустите тесты:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_v2_pipeline.py
```

## Шаг 3: Генерация музыки

Используйте предобученную модель для генерации музыки из примера:

```powershell
.\.venv\Scripts\python.exe generate_from_fasta_v2_fragmented.py `
  --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
  --fasta data\fasta\refseq_genomes\GCF_000005845.2_genomic.fna `
  --output my_first_music.mid `
  --bars-per-fragment 4
```

Это создаст MIDI файл `my_first_music.mid` из генома E. coli.

## Шаг 4: Прослушивание результата

Откройте `my_first_music.mid` в любом MIDI-плеере или DAW (например, MuseScore, GarageBand, FL Studio).

## Шаг 5: Веб-интерфейс (опционально)

Для более удобной работы запустите веб-приложение:

```powershell
.\.venv\Scripts\python.exe -m web.app
```

Откройте http://localhost:5001 в браузере. Здесь вы можете:
- Вставить свою последовательность или загрузить FASTA файл
- Сгенерировать музыку в один клик
- Прослушать примеры из галереи
- Скачать MIDI файлы

## Что дальше?

### Хотите обучить свою модель?
Следуйте [Руководству по обучению](getting-started/training-quickstart.md)

### Хотите понять, как это работает?
Прочитайте [Архитектуру и методологию](technical/architecture-and-science.md)

### Хотите использовать свои данные?
Изучите [Руководство по данным](guides/data-guide.md)

### Хотите развернуть в production?
См. [Production развертывание](deployment/README.md)

## Troubleshooting

### Ошибка: ModuleNotFoundError: No module named 'torch'

Убедитесь, что используете Python из виртуального окружения:

```powershell
# Неправильно
python generate_from_fasta_v2_fragmented.py ...

# Правильно
.\.venv\Scripts\python.exe generate_from_fasta_v2_fragmented.py ...
```

### Ошибка: CUDA out of memory

Используйте CPU для генерации:

```powershell
.\.venv\Scripts\python.exe generate_from_fasta_v2_fragmented.py `
  --checkpoint results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt `
  --fasta your_sequence.fasta `
  --output output.mid `
  --device cpu
```

### Модель не найдена

Убедитесь, что файл checkpoint существует:

```powershell
Test-Path results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt
```

Если файла нет, вам нужно сначала обучить модель или скачать предобученную.

## Полезные команды

```powershell
# Проверить версию Python
python --version

# Проверить установленные пакеты
.\.venv\Scripts\python.exe -m pip list

# Запустить все тесты
.\.venv\Scripts\python.exe -m pytest

# Проверить код
.\.venv\Scripts\python.exe -m black . --check
.\.venv\Scripts\python.exe -m flake8 bio_music_pipeline web tests
```
