# Скрипты развертывания для Windows

PowerShell скрипты для развертывания и управления веб-сервисом BioSonification на Windows.

## Управление сервисами

### Приложение BioSonification

- **`install-task.ps1`** - Установить BioSonification как службу Windows Task Scheduler
- **`start-task.ps1`** - Запустить службу BioSonification
- **`stop-task.ps1`** - Остановить службу BioSonification
- **`restart-task.ps1`** - Перезапустить службу BioSonification
- **`uninstall-task.ps1`** - Удалить службу BioSonification
- **`monitor-task.ps1`** - Мониторинг статуса и логов службы BioSonification
- **`run-service.ps1`** - Обёртка службы с автоматическим перезапуском (используется внутри Task Scheduler)

### Reverse Proxy Caddy

- **`install-caddy-service.ps1`** - Установить Caddy как службу Windows Task Scheduler с автоматическим HTTPS
- **`restart-caddy.ps1`** - Перезапустить службу Caddy (использовать после изменения Caddyfile)

## Настройка сети

- **`setup-firewall.ps1`** - Настроить правила Windows Firewall для портов 80, 443 и 5001

## Мониторинг

- **`monitor-production.ps1`** - Мониторинг обеих служб (BioSonification и Caddy), проверка health endpoints и SSL-сертификата

## Требования

- Windows 10/11
- PowerShell 5.1+
- Права администратора (для установки служб и настройки firewall)

## Быстрый старт

1. Установить службы:
   ```powershell
   .\install-task.ps1
   .\install-caddy-service.ps1
   ```

2. Настроить firewall:
   ```powershell
   .\setup-firewall.ps1
   ```

3. Мониторинг:
   ```powershell
   .\monitor-production.ps1
   ```

## Примечания

- Все скрипты должны запускаться из корневой директории проекта
- Установка служб требует прав администратора
- Caddy автоматически получает и обновляет SSL-сертификаты от Let's Encrypt
- BioSonification работает на `localhost:5001`, Caddy проксирует на порты 80/443
