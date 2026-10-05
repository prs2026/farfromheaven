@echo off
setlocal

if "%~1"=="" (
    echo Drag one or more .CDX1 files onto this launcher.
    pause
    exit /b 2
)

set "VENV_PYTHON=%~dp0..\.venv\Scripts\python.exe"
if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" "%~dp0cdx1_summary_png.py" %* --open
) else (
    where py >nul 2>nul
    if errorlevel 1 (
        python "%~dp0cdx1_summary_png.py" %* --open
    ) else (
        py -3 "%~dp0cdx1_summary_png.py" %* --open
    )
)

if errorlevel 1 (
    echo.
    echo The summary could not be generated.
    pause
    exit /b 1
)

echo.
echo Summary PNG generated beside the CDX1 file.
timeout /t 3 >nul
