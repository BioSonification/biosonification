# Данные Для Structured V2

Этот документ описывает только актуальный контур проекта: `configs/pipeline_v2_small.json`, `train_bio_music_v2.py`, `generate_from_fasta_v2.py` и пакет `bio_music_pipeline/v2/`.

## FASTA

Основной FASTA-файл задаётся полем `fasta_path` в конфиге:

```json
{
  "fasta_path": "data/fasta/quick_sample.fa"
}
```

Поддерживаются расширения `.fasta`, `.fa`, `.fna`, `.ffn`, `.faa`, `.frn`. Энкодер принимает DNA, RNA и protein-like последовательности, очищает записи и режет длинные последовательности на фрагменты по параметрам из `bio`.

## MIDI

Полифонический MIDI-корпус задаётся списком `music.midi_dirs`:

```json
{
  "music": {
    "midi_dirs": ["data/midi/polyphonic_music21"]
  }
}
```

Корпус должен содержать `.mid`, `.midi`, `.xml`, `.mxl` или `.musicxml` файлы. Из них извлекаются структурированные сегменты:

- гармония по тактам;
- монофоническая мелодическая линия;
- музыкальные дескрипторы для pairing.

Встроенный `music21` fallback подходит для smoke-тестов и локальной проверки. Для экспериментов, на которые хочется опираться в отчётах, лучше использовать внешний лицензированный полифонический MIDI-корпус и явно указать его в `music.midi_dirs`.

## Sanity Report

Перед обучением удобно зафиксировать, какие данные реально попали в запуск:

```powershell
.\.venv\Scripts\python.exe tools\report_structured_dataset.py --config configs\pipeline_v2_small.json --output-dir results\v2_dataset_report
```

Команда создаёт:

- `results/v2_dataset_report/dataset_report.json`
- `results/v2_dataset_report/dataset_report.md`

Отчёт показывает количество FASTA records/fragments, количество MIDI-файлов, тип музыкального источника, число извлечённых structured-сегментов, распределения tempo/key/density и фактические параметры сегментации.

## Типичный Поток

1. Положите FASTA в `data/fasta/` или укажите абсолютный путь в своём конфиге.
2. Положите полифонические MIDI/XML-файлы в отдельный каталог.
3. Создайте копию `configs/pipeline_v2_small.json` и обновите `fasta_path` и `music.midi_dirs`.
4. Запустите dataset report.
5. Запустите обучение:

```powershell
.\.venv\Scripts\python.exe train_bio_music_v2.py --config configs\pipeline_v2_small.json
```

6. Сгенерируйте MIDI из FASTA:

```powershell
.\.venv\Scripts\python.exe generate_from_fasta_v2.py --checkpoint results\v2_music21_rtx2060\checkpoints\structured_pipeline.pt --fasta data\fasta\quick_sample.fa --output results\v2_generation\structured_from_fasta.mid --metadata-output results\v2_generation\structured_from_fasta.json
```

## Частые Проблемы

- MIDI-файлы не находятся: проверьте `music.midi_dirs` и расширения файлов.
- Сегментов слишком мало: увеличьте корпус или ослабьте ограничения `music.min_notes_per_segment`, `music.bars_per_segment`, `music.max_segments`.
- FASTA не читается: проверьте, что заголовки начинаются с `>` и последовательность содержит допустимые символы.
- Результаты трудно воспроизвести: сохраните конфиг, `dataset_report.*`, `metrics.json` и checkpoint. Runtime-каталоги `results/`, `outputs/`, `tmp/`, `web/output/` не хранятся в git.
