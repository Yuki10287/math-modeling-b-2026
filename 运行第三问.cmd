@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\launch_model.ps1" -Problem q3
exit /b %errorlevel%
