# Полный запуск с нуля (только FASTA + MIDI)

Этот документ — пошаговая инструкция, как запустить проект с нуля, если у вас есть только:
- датасет биопоследовательностей (`.fasta/.fa/.fna/...`)
- датасет MIDI (`.mid/.midi`)

## 1) Требования

- macOS / Linux
- `python3` (рекомендуется 3.10+)
- `pip`
- (опционально) `ffmpeg` для некоторых аудио-сценариев веб-части

Проверка:

```bash
python3 --version
pip3 --version
```

## 2) Клонирование и вход в проект

```bash
git clone <URL_РЕПОЗИТОРИЯ>
cd biosonification
```

## 3) Окружение и зависимости

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 4) Подготовка структуры данных

Создайте стандартные папки:

```bash
mkdir -p data/fasta data/midi
```

Скопируйте данные:

- все FASTA-файлы в `data/fasta/`
- все MIDI-файлы в `data/midi/` (можно с подпапками)

Пример:

```bash
cp /path/to/your_fasta_dataset/* data/fasta/
cp -R /path/to/your_midi_dataset/* data/midi/
```

## 5) Проверка, что файлы обнаружены

```bash
python3 scan_datasets.py
```

Если нужно сканировать конкретный путь:

```bash
python3 scan_datasets.py --path /absolute/path/to/data
```

## 6) (Рекомендуется) очистка старых артефактов

```bash
./tools/clean_pipeline_artifacts.sh
./tools/clean_pipeline_artifacts.sh --apply
```

## 7) Создание paired-данных (важно для научного режима)

Лучший режим в этом проекте — обучение на парных данных MIDI↔bio.

```bash
python3 -m bio_music_pipeline.data.paired_dataset_creator \
  --midi-dir data/midi \
  --fasta-path data/fasta \
  --output-dir results/paired_data \
  --config configs/pipeline_full_paired.json
```

После этого должны появиться:
- `results/paired_data/paired_data.json`
- `results/paired_data/paired_bio_vectors.npy`
- `results/paired_data/paired_conditioning_vectors.npy`

## 8) Запуск пайплайна (полный режим)

```bash
python3 run_pipeline.py \
  --config configs/pipeline_full_paired.json \
  --midi-dir data/midi \
  --paired-data results/paired_data
```

Это выполнит:
1. извлечение bio-векторов из FASTA  
2. сонификацию (с калибровкой)  
3. подготовку MIDI-датасета  
4. обучение модели  
5. генерацию, оценку, визуализации, отчёты  

## 9) Где смотреть результаты

Основные артефакты:

- `results/full_paired_run/final_report.json`
- `results/full_paired_run/summary.txt`
- `results/full_paired_run/reports/evaluation_results.json`
- `results/full_paired_run/visualizations/` (графики)
- `results/full_paired_run/midi/` (сгенерированные MIDI)
- `results/full_paired_run/surveys/human_evaluation_survey.html`

## 10) Построение сводных графиков и отчёта по всем прогонам

```bash
python3 tools/generate_research_artifacts.py \
  --roots results \
  --output-dir results/research_artifacts
```

Получите:
- `results/research_artifacts/research_artifacts.md`
- `results/research_artifacts/artifacts_summary.csv`
- `results/research_artifacts/*.png`

## 11) (Опционально) мульти-seed эксперименты для стабильности выводов

```bash
python3 tools/run_multi_seed_experiments.py \
  --base-config configs/pipeline_full_paired.json \
  --seeds 7,42,123,2026,31415 \
  --midi-dir data/midi \
  --paired-data results/paired_data
```

Сводка будет в:
- `results/multi_seed/per_seed_summary.csv`
- `results/multi_seed/aggregate_metrics.json`

## 12) Самый короткий сценарий (если нужно просто «завести»)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/fasta data/midi
# скопировать свои fasta/midi в эти папки
python3 -m bio_music_pipeline.data.paired_dataset_creator --midi-dir data/midi --fasta-path data/fasta --output-dir results/paired_data --config configs/pipeline_full_paired.json
python3 run_pipeline.py --config configs/pipeline_full_paired.json --midi-dir data/midi --paired-data results/paired_data
python3 tools/generate_research_artifacts.py --roots results --output-dir results/research_artifacts
```

## 13) Частые проблемы

- `python: command not found`  
  Используйте `python3`.

- `No valid MIDI files found`  
  Проверьте расширения `.mid/.midi` и что файлы реально не пустые/не битые.

- `No biological sequences provided`  
  Проверьте, что в `data/fasta` есть корректные FASTA-файлы.

- CUDA/GPU не используется  
  Это не блокер: пайплайн работает и на CPU, просто медленнее.

