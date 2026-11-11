@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "PY="

python -c "import sys" >nul 2>nul
if not errorlevel 1 set "PY=python"

if not defined PY (
    py -3 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "PY=py -3"
)

if not defined PY (
    if exist "C:\Users\Administrator\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe" set "PY=C:\Users\Administrator\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
)

if not defined PY (
    echo Python not found.
    pause
    exit /b 1
)

echo.
%PY% service.py
echo.
pause
