@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\launch_model.ps1" -Problem share25
exit /b %errorlevel%
