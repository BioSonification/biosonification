# BioSonification Repair Plan

Цель документа: держать единый план исправления проекта после аудита, чтобы двигаться последовательно и не смешивать актуальный `structured v2` пайплайн с legacy-слоями.

## Текущее состояние

- Основной CLI-контур `train_bio_music_v2.py` -> `generate_from_fasta_v2.py` работает.
- Smoke-тесты проходят: `tests/test_v2_pipeline.py`.
- Генерация из `results/v2_music21_rtx2060/checkpoints/structured_pipeline.pt` создаёт двухдорожечный MIDI: harmony + melody.
- Главный риск проекта сейчас не в ядре `structured v2`, а в рассинхроне вокруг него: web, часть документации, слабая оценка качества, слабые тесты и tracked generated artifacts.

## Приоритеты

1. Зафиксировать `structured v2` как единственный основной путь.
2. Убрать или явно изолировать legacy-слои.
3. Подключить web к актуальному checkpoint-формату.
4. Сделать воспроизводимую оценку качества, а не только smoke-запуск.
5. Усилить тесты вокруг CLI, inference, web и данных.
6. Навести порядок в репозитории и артефактах.

## Phase 1. Repo Hygiene

### Задачи

- [x] Обновить `.gitignore`: явно добавить `web/output/`, `outputs/`, `tmp/`, generated MIDI/WAV/PNG/PPTX artifacts, кроме намеренно хранимых demo-файлов.
- [x] Убрать уже отслеживаемые generated artifacts из git index через `git rm --cached`, не удаляя локальные файлы.
- [x] Проверить `git status`, чтобы в изменениях остались только исходники и документация.
- [x] Решить судьбу legacy outputs: удалить из индекса как runtime/generated artifacts.

### Definition Of Done

- `git ls-files` не показывает `tmp/`, `outputs/`, `web/output/`.
- Новые локальные генерации не попадают в `git status`.
- В README или `docs/project_structure.md` указано, где лежат runtime artifacts.

## Phase 2. Documentation Alignment

### Задачи

- [x] Обновить `web/README.md`: заменить legacy `run_pipeline.py` сценарий на `structured v2` или явно пометить web как legacy до переподключения.
- [x] Обновить `docs/thesis_chapters.md`: убрать утверждение, что `configs/pipeline_full_paired.json` является основной конфигурацией текущей версии.
- [ ] Разнести документы по статусам:
  - current: `README.md`, `RUN_FROM_SCRATCH.md`, `docs/architecture_and_science.md`, `docs/code_walkthrough.md`, `docs/project_structure.md`
  - legacy/archive: материалы про `run_pipeline.py`, `full_paired_run`, старый single-stream v2
- [x] Добавить краткий раздел "Что является актуальным" в README.
- [x] Добавить раздел "Что не доказано" или "Scientific limitations": bio-признаки являются conditioning signals, а не доказанной причинной связью между генами и музыкой.

### Definition Of Done

- Поиск по документации не находит противоречий, где legacy путь назван текущим основным.
- Web-документация честно соответствует фактическому состоянию web-кода.
- Все команды быстрого старта запускают актуальные entrypoint'ы.

## Phase 3. Web Migration To Structured V2

### Задачи

- [x] Переписать `web/generator.py` под `bio_music_pipeline.v2.generate_structured_music_from_fasta`.
- [x] Изменить поиск checkpoint:
  - основной: `results/v2_music21_rtx2060/checkpoints/structured_pipeline.pt`
  - fallback: newest `results/*/checkpoints/structured_pipeline.pt`
  - override: `BIOSONIFICATION_STRUCTURED_CHECKPOINT`
- [x] Изменить config resolution:
  - основной: `configs/pipeline_v2_small.json`
  - override: `BIOSONIFICATION_CONFIG_PATH`
- [x] Убрать зависимость web inference от legacy `BioVectorExtractor`, `SonificationMapper`, `MIDIPreprocessor`, `create_bio_music_model`.
- [x] Для web input сохранять временный FASTA в runtime directory и вызывать structured generator по этому файлу.
- [x] В API response вернуть metadata из structured generator:
  - `sequence_id`
  - `sequence_type`
  - `cleaned_sequence_length`
  - `tempo_bpm`
  - `tonic_pc_hint`
  - `generated_harmony_bars`
  - `generated_melody_note_count`
- [x] Обновить frontend labels: результат теперь harmony + melody MIDI, а не старый single-stream model.
- [x] Проверить `/api/status` через Flask test client. Полный ручной запуск `python -m web.app` остаётся smoke-проверкой перед демонстрацией.

### Definition Of Done

- Web стартует при наличии `structured_pipeline.pt`.
- `/api/status` показывает найденный structured checkpoint/config.
- `/api/generate` создаёт MIDI через `structured v2`.
- Web больше не требует `results/full_paired_run/models/best_model.pt`.
- `web/README.md` совпадает с реальным запуском.

## Phase 4. Checkpoint And Inference Robustness

### Задачи

- [x] Сделать checkpoint self-contained: при inference по умолчанию использовать `config` из checkpoint.
- [x] Внешний `--config` трактовать как override и валидировать совместимость:
  - vocab sizes
  - `descriptor_bins`
  - `steps_per_bar`
  - model dimensions
  - `embedding_dim`
  - max sequence lengths
- [x] Добавить явную ошибку при несовместимости checkpoint/config.
- [x] Добавить CLI флаг `--device` для `generate_from_fasta_v2.py`: `auto`, `cpu`, `cuda`.
- [x] Добавить metadata fields:
  - checkpoint config hash
  - effective config path/source
  - device
  - generation seed, если seed фиксируется

### Definition Of Done

- Старый корректный вызов generation продолжает работать.
- Если передать несовместимый config, inference падает с понятной ошибкой до `load_state_dict`.
- CPU-only inference можно принудительно запустить.

## Phase 5. Test Coverage

### Задачи

- [x] Добавить unit tests для config/checkpoint compatibility.
- [ ] Добавить CLI smoke test для `generate_from_fasta_v2.py` на маленьком fixture checkpoint или mock checkpoint.
- [ ] Добавить tests для FASTA edge cases:
  - missing file
  - too short after cleaning
  - multi-record record index
  - RNA/protein input
- [ ] Добавить tests для structured corpus fallback:
  - empty MIDI dir + fallback enabled
  - empty MIDI dir + fallback disabled
- [ ] Добавить tests для tokenizer invariants:
  - harmony decode always returns `num_bars`
  - melody decode stays bounded and monophonic
  - unknown/generated invalid token sequences do not crash decode
- [x] Добавить Flask tests:
  - `/api/status`
  - `/api/generate` happy path with monkeypatched generator
  - invalid input response
- [x] Добавить test command в README.

### Definition Of Done

- `pytest` покрывает не только модульные smoke cases, но и CLI/web boundaries.
- Быстрый тестовый набор запускается без GPU.
- Есть отдельный маркер или инструкция для slow/integration tests.

## Phase 6. Evaluation Framework

### Задачи

- [ ] Зафиксировать evaluation dataset:
  - held-out FASTA
  - held-out MIDI/music segments
  - deterministic sample list
- [x] Добавить baselines:
  - random/unconditioned harmony+melody
  - shuffled bio profiles
  - deterministic sonification legacy baseline
  - music-only prior baseline
- [ ] Добавить ablations:
  - no bio vector, only control profile
  - bio vector without protein/RNA features
  - no calibration
  - different pairing `top_k` and temperature
- [ ] Добавить multi-seed runner для structured v2.
- [x] Считать базовые метрики:
  - train/val/test loss по seed
  - note density
  - pitch range
  - chord change rate
  - chord-tone ratio
  - repetition/self-similarity
  - invalid/empty generation rate
  - MIDI duration/part count invariants
- [x] Сохранять evaluation report в JSON + Markdown.
- [ ] Добавить human survey flow или обновить существующие `/api/survey/*` под structured outputs.

### Definition Of Done

- Есть команда одной строкой для запуска evaluation.
- Результат можно вставить в статью/доклад без ручного пересчёта.
- Можно сравнить current model vs baseline vs ablations.
- Ограничения интерпретации явно записаны рядом с метриками.

## Phase 7. Data And Corpus Quality

### Задачи

- [x] Решить, fallback `music21` остаётся demo-only или допустим для основного эксперимента.
- [x] Если нужен серьёзный корпус, добавить documented ingestion path для внешнего polyphonic MIDI dataset.
- [x] Проверить `MusicDataConfig.music21_composers`: fallback теперь явно поддерживает только `bach` и сообщает об ошибке для неподдержанных значений.
- [x] Добавить dataset manifest:
  - source
  - license/status
  - file count
  - extracted segment count
  - filtering criteria
- [x] Добавить sanity report по корпусу: распределения темпа, тональностей, плотности, polyphony, длительности.

### Definition Of Done

- Документация не создаёт впечатление, что fallback Bach chorales являются полноценной универсальной обучающей базой.
- Для каждого результата понятно, на каком корпусе он обучен.
- Конфиг не содержит неработающих или вводящих в заблуждение полей.

## Phase 8. Legacy Strategy

### Задачи

- [x] Принять решение по `run_pipeline.py`, `generate_from_fasta.py`, legacy modules:
  - удалить
  - оставить в `legacy/`
  - оставить на месте, но пометить deprecated
- [x] Если legacy остаётся, добавить `docs/legacy.md`.
- [x] Убрать legacy imports из current path.
- [x] Проверить, что `bio_music_pipeline/v2/__init__.py` экспортирует только актуальные stable APIs.

### Definition Of Done

- Новому пользователю понятно, какой код запускать.
- Legacy-код не маскируется под актуальный.
- Web и README не зависят от legacy path.

## Suggested Execution Order

1. Phase 1: repo hygiene.
2. Phase 2: documentation alignment.
3. Phase 3: web migration.
4. Phase 4: checkpoint/inference robustness.
5. Phase 5: tests for the changed boundaries.
6. Phase 6: evaluation framework.
7. Phase 7: corpus quality.
8. Phase 8: final legacy cleanup.

## Quick Validation Commands

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_v2_pipeline.py -q
```

```powershell
.\.venv\Scripts\python.exe generate_from_fasta_v2.py --config configs\pipeline_v2_small.json --checkpoint results\v2_music21_rtx2060\checkpoints\structured_pipeline.pt --fasta data\fasta\quick_sample.fa --output results\v2_generation\structured_from_fasta.mid --metadata-output results\v2_generation\structured_from_fasta.json
```

```powershell
.\.venv\Scripts\python.exe tools\evaluate_structured_v2.py --checkpoint results\v2_music21_rtx2060\checkpoints\structured_pipeline.pt --fasta data\fasta\quick_sample.fa --output-dir results\v2_evaluation --max-records 4 --device auto
```

```powershell
.\.venv\Scripts\python.exe tools\report_structured_dataset.py --config configs\pipeline_v2_small.json --output-dir results\v2_dataset_report
```

```powershell
@'
from music21 import converter, chord, note
score = converter.parse("results/v2_generation/structured_from_fasta.mid")
print("highestTime:", float(score.highestTime))
print("parts:", len(score.parts))
for i, part in enumerate(score.parts):
    flat = list(part.flatten().notes)
    print("part", i, "events", len(flat), "notes", sum(isinstance(x, note.Note) for x in flat), "chords", sum(isinstance(x, chord.Chord) for x in flat))
'@ | .\.venv\Scripts\python.exe -
```
