@echo off
setlocal
chcp 65001 >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_services.ps1"
set "stop_exit=%ERRORLEVEL%"
echo.
pause
exit /b %stop_exit%
