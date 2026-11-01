@echo off
REM Double-clickable Windows wrapper for install.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
pause
