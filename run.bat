@echo off
REM ===================================================================
REM  One-click runner for the copper smelting blending pipeline.
REM
REM  IMPORTANT: this file is intentionally pure ASCII.
REM  cmd.exe interprets non-ASCII bytes using the current console code
REM  page, so UTF-8 Chinese here gets split into bogus commands.
REM  All Chinese output is printed by Python (ml_pipeline\show_menu.py
REM  and the step scripts themselves), which handle encoding properly.
REM
REM  Interpreter: set TABPFN_PYTHON to override, e.g.
REM      set TABPFN_PYTHON=E:\miniconda3\envs\tabpfn\python.exe
REM  Otherwise "python" from PATH is used.
REM ===================================================================
setlocal
cd /d "%~dp0"
REM 65001 = UTF-8: makes Python's Chinese output render correctly even if the
REM console defaults to GBK. Safe here because this file is pure ASCII.
chcp 65001 >nul

set "PY=%TABPFN_PYTHON%"
if not defined PY set "PY=python"

"%PY%" -c "import tabpfn" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] the interpreter does not have tabpfn installed:
  echo         %PY%
  echo         Activate your environment first, or set TABPFN_PYTHON
  echo         to the full path of the python.exe inside it.
  pause
  exit /b 1
)

if not "%~1"=="" (
  set "PICK=%~1"
  goto dispatch
)

:menu
cls
"%PY%" -u ml_pipeline\show_menu.py
set "PICK="
set /p "PICK=  code: "
if not defined PICK set "PICK=1k"
if /i "%PICK%"=="q" exit /b 0

:dispatch
echo.
for %%A in (%PICK%) do call :one %%A
echo.
echo ==================================================================
echo   All done. Results are under experiments\^<dataset^>\.
echo   Blending optimisation (fixed 1200 C / 21 g O2):  run.bat o
echo ==================================================================
pause
exit /b 0

:one
set "CODE=%~1"
if /i "%CODE%"=="q" exit /b 0
set "TARGET="
if /i "%CODE%"=="1"   set "TARGET=ml_pipeline\step1_train_tabpfn.py"
if /i "%CODE%"=="1k"  set "TARGET=ml_pipeline\step1_train_tabpfn.py --targets key"
if /i "%CODE%"=="3"   set "TARGET=ml_pipeline\step3_multiseed_evaluation.py"
if /i "%CODE%"=="4"   set "TARGET=ml_pipeline\step4_shap_all_targets.py"
if /i "%CODE%"=="5"   set "TARGET=ml_pipeline\step5_shap_key_targets.py"
if /i "%CODE%"=="5b"  set "TARGET=ml_pipeline\step5b_shap_signed.py"
if /i "%CODE%"=="6"   set "TARGET=ml_pipeline\step6_learning_curve.py"
if /i "%CODE%"=="7"   set "TARGET=ml_pipeline\step7_make_figures.py"
if /i "%CODE%"=="o"   set "TARGET=scripts\optimization\nsga2_optimization.py"

if not defined TARGET (
  echo   [skip] unknown code: %CODE%
  goto :eof
)

echo ------------------------------------------------------------------
echo   running %CODE% :  %TARGET%
echo ------------------------------------------------------------------
"%PY%" -u %TARGET%
if errorlevel 1 goto fail
goto :eof

:fail
echo.
echo ==================================================================
echo   [STOP] step %CODE% returned an error; later steps were skipped.
echo   Fix it and re-run the same code: finished work is resumed,
echo   not recomputed.
echo ==================================================================
pause
exit /b 1
