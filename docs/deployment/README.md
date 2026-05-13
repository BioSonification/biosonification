# Развертывание в production - быстрый старт

Быстрое руководство по развертыванию BioSonification в production на Windows.

## Варианты развертывания

1. **[Локальное развертывание](#локальное-развертывание-task-scheduler)** - Task Scheduler, доступ через localhost
2. **[HTTPS-развертывание](#https-развертывание-с-caddy)** - Caddy и автоматический SSL для публичного доступа

---

## HTTPS-развертывание с Caddy

Публичный доступ через домен с автоматическим SSL-сертификатом от Let's Encrypt.

### Предварительные требования

- Windows 10/11
- Python 3.10+
- Домен, указывающий на ваш публичный IP
- Доступ к настройкам роутера для проброса портов
- Права администратора

### Быстрая установка

**1. Скачать Caddy:**

```powershell
Invoke-WebRequest -Uri "https://caddyserver.com/api/download?os=windows&arch=amd64" -OutFile "$env:USERPROFILE\Downloads\caddy.exe"
Move-Item "$env:USERPROFILE\Downloads\caddy.exe" "C:\Tools\caddy\caddy.exe"
```

**2. Настроить Windows Firewall от имени администратора:**

```powershell
.\scripts\setup-firewall.ps1
```

**3. Настроить проброс портов на роутере:**

- порт 80 -> ваш локальный IP -> порт 80
- порт 443 -> ваш локальный IP -> порт 443

**4. Установить службу Caddy от имени администратора:**

```powershell
.\scripts\install-caddy-service.ps1
```

**5. Запустить службы:**

```powershell
.\scripts\start-task.ps1  # BioSonification
Start-ScheduledTask -TaskName CaddyServer  # Caddy
```

**6. Проверить:**

```powershell
.\scripts\monitor-production.ps1
```

### Управление HTTPS

```powershell
# Мониторинг обеих служб
.\scripts\monitor-production.ps1

# Caddy
Start-ScheduledTask -TaskName CaddyServer
Stop-ScheduledTask -TaskName CaddyServer

# BioSonification
.\scripts\start-task.ps1
.\scripts\stop-task.ps1
.\scripts\restart-task.ps1
```

### Полная документация HTTPS

См. [https-setup.md](https-setup.md) - подробное руководство с разделом устранения неполадок.

---

## Локальное развертывание (Task Scheduler)

Локальный доступ через localhost без HTTPS.

### Предварительные требования

- Windows 10/11
- Python 3.10+
- Права администратора

## Быстрая установка

### 1. Установить NSSM

```powershell
# Скачать: https://nssm.cc/download
# Распаковать в C:\Tools\nssm\
# Добавить в PATH:
$env:Path += ";C:\Tools\nssm\win64"
[Environment]::SetEnvironmentVariable("Path", $env:Path, [EnvironmentVariableTarget]::Machine)
```

### 2. Установить зависимости

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Создать конфигурацию

```powershell
Copy-Item .env.example .env
# Отредактировать .env при необходимости
```

### 4. Установить службу

```powershell
# От имени администратора
.\scripts\install-service.ps1
```

### 5. Запустить

```powershell
.\scripts\start-production.ps1
```

## Управление

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

## Проверка

```powershell
# Проверка состояния
curl http://localhost:5001/health

# Веб-интерфейс
start http://localhost:5001

# Логи
Get-Content logs\biosonification.log -Tail 20
```

## Полная документация

См. [docs/deployment/windows-production.md](windows-production.md).

## Устранение неполадок

**Служба не запускается:**

```powershell
Get-Content logs\service-stderr.log -Tail 30
```

**Генерация не работает:**

```powershell
# Переключить на CPU в .env
BIOSONIFICATION_DEVICE=cpu
.\scripts\restart-production.ps1
```

**Удалить службу:**

```powershell
.\scripts\uninstall-service.ps1
```
