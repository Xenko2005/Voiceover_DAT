@echo off
setlocal

cd /d "%~dp0\.."

set MODEL=qwen3.5:2b
set PHOST_TEXT=D:\datasets\PhoST_subset\text_data
set DATASET_DIR=datasets\phost_test_module1_full
set BASELINE_OUT=experiment_outputs\baseline_phost_test_full
set PROPOSED_OUT=experiment_outputs\proposed_phost_test_full
set METRIC_OUT=evaluation\phost_test_full_outputs
set OPTIMIZER_CONFIG=module_3_adaptive_translation\outputs\phost_optimizer_config_fulltrain.json

echo ============================================================
echo Step 0/4: Check fulltrain lightweight module optimizer config
echo ============================================================
if not exist "%OPTIMIZER_CONFIG%" (
  echo Missing %OPTIMIZER_CONFIG%
  echo Run:
  echo python experiments\train_phost_module_optimizers.py --text-root "%PHOST_TEXT%" --output "%OPTIMIZER_CONFIG%" --train-limit-files 3028 --dev-limit-files 31 --train-sample-rows 327370 --dev-sample-rows 1935
  exit /b 1
)

echo ============================================================
echo Step 1/4: Prepare PhoST test split
echo ============================================================
python experiments\prepare_phost_module1_json.py ^
  --text-root "%PHOST_TEXT%" ^
  --split test ^
  --output-dir "%DATASET_DIR%"
if errorlevel 1 exit /b 1

echo ============================================================
echo Step 2/4: Run baseline on full PhoST test split
echo ============================================================
python experiments\run_baseline_dataset.py ^
  --input-dir "%DATASET_DIR%" ^
  --output-root "%BASELINE_OUT%" ^
  --model "%MODEL%"
if errorlevel 1 exit /b 1

echo ============================================================
echo Step 3/4: Run proposed on full PhoST test split
echo ============================================================
python experiments\run_proposed_dataset.py ^
  --input-dir "%DATASET_DIR%" ^
  --output-root "%PROPOSED_OUT%" ^
  --model "%MODEL%" ^
  --batch-size 5 ^
  --optimizer-config "%OPTIMIZER_CONFIG%"
if errorlevel 1 exit /b 1

echo ============================================================
echo Step 4/4: Compare metrics
echo ============================================================
python experiments\compare_dataset_runs.py ^
  --baseline-root "%BASELINE_OUT%" ^
  --proposed-root "%PROPOSED_OUT%" ^
  --output-dir "%METRIC_OUT%"
if errorlevel 1 exit /b 1

echo.
echo DONE.
echo Metrics: %METRIC_OUT%\dataset_metrics_summary.json
echo CSV:     %METRIC_OUT%\combined_dataset_metrics.csv

endlocal
