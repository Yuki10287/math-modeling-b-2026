@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\launch_model.ps1" -Problem q4
exit /b %errorlevel%
