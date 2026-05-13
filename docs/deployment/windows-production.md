# Production развертывание на Windows 11

Руководство по развертыванию BioSonification в production на Windows 11 с автоматическим перезапуском и мониторингом.

---

## Архитектура

```
[Windows Service (NSSM)] → [Waitress WSGI Server] → [Flask App] → [GPU/CPU Generation]
         ↓                          ↓                      ↓
    Auto-restart              Production-grade         Logging
    Monitoring                Multi-threaded           Error handling
```

---

## Требования к системе

### Минимальные требования

- **OS:** Windows 11 (или Windows 10)
- **Python:** 3.12
- **RAM:** 8 GB (16 GB рекомендуется)
- **Disk:** 15 GB свободного места
- **GPU:** Опционально (NVIDIA с CUDA support для ускорения)

### Программное обеспечение

- Python 3.12 с pip
- Git (для обновлений)
- NSSM (Non-Sucking Service Manager)
- CUDA Toolkit 12.6 (если используется GPU)

---

## Установка

### 1. Подготовка окружения

```powershell
# Клонировать репозиторий (если еще не сделано)
cd C:\Users\vlasi\Documents
git clone <repository-url> biosonification
cd biosonification

# Создать виртуальное окружение
python -m venv .venv

# Активировать
.\.venv\Scripts\Activate.ps1

# Обновить pip
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# Установить PyTorch с CUDA (для GPU)
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126

# Установить зависимости
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Установка NSSM

```powershell
# Скачать NSSM
# https://nssm.cc/download

# Распаковать в C:\Tools\nssm\

# Добавить в PATH (от администратора)
$env:Path += ";C:\Tools\nssm\win64"
[Environment]::SetEnvironmentVariable("Path", $env:Path, [EnvironmentVariableTarget]::Machine)

# Проверить установку
nssm --version
```

### 3. Конфигурация

```powershell
# Создать .env файл из примера
Copy-Item .env.example .env

# Отредактировать .env
notepad .env
```

**Основные параметры в .env:**

```bash
# Server
BIOSONIFICATION_HOST=127.0.0.1
BIOSONIFICATION_PORT=5001
BIOSONIFICATION_THREADS=4

# Model
BIOSONIFICATION_DEVICE=auto  # auto, cpu, или cuda

# Logging
BIOSONIFICATION_LOG_LEVEL=INFO
BIOSONIFICATION_LOG_FILE=logs/biosonification.log
```

### 4. Проверка работоспособности

```powershell
# Запустить в dev режиме
.\.venv\Scripts\python.exe -m web.app

# В другом терминале проверить
curl http://localhost:5001/health

# Открыть в браузере
start http://localhost:5001

# Протестировать генерацию
# (загрузить FASTA через веб-интерфейс)

# Остановить (Ctrl+C)
```

---

## Установка Windows Service

### 1. Установить службу

```powershell
# Запустить PowerShell от администратора
# Перейти в директорию проекта
cd C:\Users\vlasi\Documents\biosonification

# Запустить скрипт установки
.\scripts\install-service.ps1
```

Скрипт выполнит:
- Проверку прав администратора
- Проверку наличия NSSM
- Установку службы BioSonification
- Настройку автозапуска
- Настройку логирования
- Настройку автоматического перезапуска при сбоях

### 2. Запустить службу

```powershell
.\scripts\start-production.ps1
```

Скрипт:
- Запустит службу
- Проверит статус
- Выполнит health check
- Покажет URL веб-интерфейса

### 3. Проверить работу

```powershell
# Мониторинг
.\scripts\monitor.ps1

# Открыть веб-интерфейс
start http://localhost:5001

# Проверить логи
Get-Content logs\biosonification.log -Tail 20
```

---

## Управление службой

### Основные команды

```powershell
# Запуск
.\scripts\start-production.ps1

# Остановка
.\scripts\stop-production.ps1

# Перезапуск
.\scripts\restart-production.ps1

# Мониторинг
.\scripts\monitor.ps1
```

### Команды NSSM

```powershell
# Статус
nssm status BioSonification

# Запуск
nssm start BioSonification

# Остановка
nssm stop BioSonification

# Перезапуск
nssm restart BioSonification

# Конфигурация
nssm edit BioSonification
```

### Управление через Services.msc

1. Нажать `Win + R`
2. Ввести `services.msc`
3. Найти "BioSonification Web Service"
4. Правой кнопкой → Start/Stop/Restart

---

## Мониторинг

### Health Check

```powershell
# Проверить здоровье приложения
curl http://localhost:5001/health

# Ожидаемый ответ:
# {
#   "status": "healthy",
#   "timestamp": "2026-05-10T12:00:00",
#   "generator_ready": true,
#   "gpu_available": true
# }
```

### Логи

**Логи приложения:**
```powershell
# Последние 20 строк
Get-Content logs\biosonification.log -Tail 20

# Следить в реальном времени
Get-Content logs\biosonification.log -Tail 20 -Wait

# Поиск ошибок
Get-Content logs\biosonification.log | Select-String "ERROR" -Context 2
```

**Логи службы:**
```powershell
# Stdout
Get-Content logs\service-stdout.log -Tail 20

# Stderr
Get-Content logs\service-stderr.log -Tail 20
```

### GPU мониторинг

```powershell
# Текущее состояние
nvidia-smi

# Непрерывный мониторинг
watch -n 1 nvidia-smi

# Или в PowerShell
while ($true) { cls; nvidia-smi; Start-Sleep -Seconds 1 }
```

---

## Troubleshooting

### Служба не запускается

**Проблема:** `nssm status BioSonification` возвращает ошибку

**Решение:**
```powershell
# Проверить логи
Get-Content logs\service-stderr.log -Tail 30

# Проверить Python
.\.venv\Scripts\python.exe --version

# Проверить зависимости
.\.venv\Scripts\python.exe -c "import flask; import waitress; print('OK')"

# Переустановить службу
.\scripts\uninstall-service.ps1
.\scripts\install-service.ps1
```

### Health check не отвечает

**Проблема:** `curl http://localhost:5001/health` не работает

**Решение:**
```powershell
# Проверить, запущена ли служба
nssm status BioSonification

# Проверить порт
netstat -ano | findstr :5001

# Проверить логи
Get-Content logs\biosonification.log -Tail 30

# Проверить .env
Get-Content .env
```

### Генерация не работает

**Проблема:** Генерация MIDI завершается с ошибкой

**Решение:**
```powershell
# Проверить checkpoint
Test-Path results\v2_medium_rtx2060_fast\checkpoints\structured_pipeline.pt

# Проверить GPU
nvidia-smi

# Переключить на CPU (в .env)
# BIOSONIFICATION_DEVICE=cpu

# Перезапустить
.\scripts\restart-production.ps1
```

### CUDA out of memory

**Проблема:** Ошибка "CUDA out of memory" в логах

**Решение:**
```powershell
# Переключить на CPU в .env
notepad .env
# Изменить: BIOSONIFICATION_DEVICE=cpu

# Перезапустить
.\scripts\restart-production.ps1
```

### Служба не перезапускается автоматически

**Проблема:** После сбоя служба не перезапускается

**Решение:**
```powershell
# Проверить конфигурацию
nssm dump BioSonification | Select-String "AppExit"

# Должно быть: AppExit Default Restart

# Переустановить службу
.\scripts\uninstall-service.ps1
.\scripts\install-service.ps1
```

---

## Обновление приложения

### Процедура обновления

```powershell
# 1. Остановить службу
.\scripts\stop-production.ps1

# 2. Сделать backup (опционально)
$date = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item results\v2_medium_rtx2060_fast\checkpoints "backup\checkpoints_$date" -Recurse

# 3. Обновить код
git pull

# 4. Обновить зависимости (если изменились)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 5. Запустить службу
.\scripts\start-production.ps1

# 6. Проверить
.\scripts\monitor.ps1
```

---

## Безопасность

### Checklist

- [ ] .env не в git (проверить .gitignore)
- [ ] Firewall настроен (только localhost:5001)
- [ ] Логи ротируются автоматически (NSSM)
- [ ] Ограничение размера файлов (10MB в app.py)
- [ ] Регулярные backup чекпоинтов
- [ ] Мониторинг дискового пространства

### Рекомендации

1. **Доступ только с localhost:**
   - В .env: `BIOSONIFICATION_HOST=127.0.0.1`
   - Для внешнего доступа используйте reverse proxy (Nginx)

2. **Регулярные backup:**
   ```powershell
   # Создать backup скрипт
   $date = Get-Date -Format "yyyyMMdd"
   Copy-Item results\v2_medium_rtx2060_fast\checkpoints "backup\checkpoints_$date" -Recurse
   ```

3. **Мониторинг дискового пространства:**
   ```powershell
   # Проверить место
   Get-PSDrive C | Select-Object Used,Free
   ```

---

## Автозапуск при старте системы

Служба настроена на автоматический запуск при старте Windows.

**Проверка:**
```powershell
# Перезагрузить компьютер
Restart-Computer

# После загрузки проверить
.\scripts\monitor.ps1
```

**Отключить автозапуск:**
```powershell
nssm set BioSonification Start SERVICE_DEMAND_START
```

**Включить автозапуск:**
```powershell
nssm set BioSonification Start SERVICE_AUTO_START
```

---

## Производительность

### Оптимизация

**Количество потоков:**
```bash
# В .env
BIOSONIFICATION_THREADS=4  # Для CPU
BIOSONIFICATION_THREADS=2  # Для GPU (меньше конкуренции)
```

**Устройство генерации:**
```bash
# В .env
BIOSONIFICATION_DEVICE=cuda  # Использовать GPU
BIOSONIFICATION_DEVICE=cpu   # Использовать CPU
BIOSONIFICATION_DEVICE=auto  # Автоматический выбор
```

### Метрики

**Время генерации:**
- Короткая последовательность (1800 bp): ~10-30 секунд
- Средняя последовательность (3600 bp): ~20-60 секунд
- Длинная последовательность (10000 bp): ~60-180 секунд

**Использование ресурсов:**
- RAM: ~2-4 GB
- GPU Memory: ~2-4 GB (при использовании GPU)
- CPU: 50-100% во время генерации

---

## Удаление

### Удалить службу

```powershell
# Запустить от администратора
.\scripts\uninstall-service.ps1
```

### Полное удаление

```powershell
# Удалить службу
.\scripts\uninstall-service.ps1

# Удалить виртуальное окружение
Remove-Item .venv -Recurse -Force

# Удалить логи
Remove-Item logs -Recurse -Force

# Удалить проект
cd ..
Remove-Item biosonification -Recurse -Force
```

---

## Дополнительные ресурсы

- [NSSM Documentation](https://nssm.cc/)
- [Waitress Documentation](https://docs.pylonsproject.org/projects/waitress/)
- [Flask Production Deployment](https://flask.palletsprojects.com/en/latest/deploying/)
