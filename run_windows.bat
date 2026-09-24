@echo off
REM Double-click to start ASFQ.  The first run builds a Python
REM environment; later runs reuse it, refresh it when requirements.txt changes,
REM and make sure the browser opens onto the copy this script just started.
setlocal
cd /d "%~dp0"

if "%PORT%"=="" set PORT=8765

REM --- the interpreter --------------------------------------------------------
REM Windows ships no Python, and on a PC without one `python` is the Store stub:
REM it opens the Store, creates nothing, and the next line then fails with "the
REM system cannot find the path".  Ask each candidate for its version instead
REM and take the first that is new enough; the stub answers nothing at all.
set PYEXE=
for %%c in ("py -3" "python" "python3") do call :trypy %%~c
if "%PYEXE%"=="" (
    echo This app needs Python 3.10 or newer, and this PC has none on PATH.
    echo Install it from https://www.python.org/downloads/ -- tick
    echo "Add python.exe to PATH" in the installer -- then double-click this again.
    pause
    exit /b 1
)

if not exist ".venv" (
    echo First run: setting up a Python environment with %PYEXE% ^(a few minutes^)...
    %PYEXE% -m venv .venv
    .venv\Scripts\python -m pip install --upgrade pip
)

REM An interrupted first run leaves a .venv that exists and holds nothing.
if not exist ".venv\Scripts\python.exe" (
    echo Could not build the Python environment in .venv -- see the messages above.
    echo Delete that folder and double-click this file again.
    pause
    exit /b 1
)

REM A .venv that exists is not the same as a .venv that is current: a
REM dependency added after the environment was built would otherwise never be
REM installed, and the only symptom is a crash partway through whatever needs it.
fc /b requirements.txt ".venv\.requirements.txt" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    .venv\Scripts\python -m pip install -q -r requirements.txt
    if errorlevel 1 (
        echo Installing the dependencies failed -- see the messages above.
        pause
        exit /b 1
    )
    copy /y requirements.txt ".venv\.requirements.txt" >nul
)

REM A busy port is usually this app, still running from an earlier day: it goes
REM on serving the current interface off disk with its own older code behind it,
REM which looks like a broken app rather than a stale one.  Stop it.
:checkport
netstat -ano | findstr /r /c:"LISTENING" | findstr /c:"127.0.0.1:%PORT% " >nul 2>&1
if errorlevel 1 goto portfree
REM Ask the occupant who it is.  /api/health is the direct answer, but a copy
REM started before that route existed cannot give it -- and a server too old to
REM introduce itself is exactly the one worth replacing.  It still serves this
REM app's own index page off disk, so the page is the fallback identification.
curl -fsS -m 2 "http://127.0.0.1:%PORT%/api/health" 2>nul | findstr /r /c:"asfq" /c:"fibrosis-quantifier" >nul 2>&1
if not errorlevel 1 goto itsus
curl -fsS -m 2 "http://127.0.0.1:%PORT%/" 2>nul | findstr /i /c:"Fibrosis Quantifier" >nul 2>&1
if errorlevel 1 goto nextport
:itsus
echo Stopping the copy already running on port %PORT%.
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /c:"127.0.0.1:%PORT% " ^| findstr /c:"LISTENING"') do taskkill /f /pid %%p >nul 2>&1
timeout /t 2 /nobreak >nul
goto checkport

:portfree
REM Open the browser only once the server answers, not after a fixed wait, so
REM a failed launch does not still show a page.
start "" /b .venv\Scripts\python open_when_ready.py %PORT% 60
.venv\Scripts\python app.py
pause
goto :eof

:nextport
echo Port %PORT% is taken by something else; trying the next one.
set /a PORT=%PORT% + 1
goto checkport

REM Runs `<candidate> -c ...`; %* is the whole candidate, so "py -3" stays two
REM words.  Keeps the first one that answered, so later candidates cost nothing.
:trypy
if not "%PYEXE%"=="" goto :eof
%* -c "import sys; raise SystemExit(sys.version_info < (3, 10))" >nul 2>&1
if errorlevel 1 goto :eof
set PYEXE=%*
goto :eof
