# ============================================================================
#  一键运行 ml_pipeline 全部步骤（或指定步骤）
# ----------------------------------------------------------------------------
#  用法（在项目根目录）：
#     powershell -ExecutionPolicy Bypass -File ml_pipeline\run_all.ps1
#     powershell -ExecutionPolicy Bypass -File ml_pipeline\run_all.ps1 -Dataset 3000_fixed_temp
#     powershell -ExecutionPolicy Bypass -File ml_pipeline\run_all.ps1 -Steps 1,7
#     powershell -ExecutionPolicy Bypass -File ml_pipeline\run_all.ps1 -Run 第二组参数
#
#  说明：
#    - 每步都会打印进度；中断后重新运行同一命令会从断点继续。
#    - 默认只跑 1,7（训练 + 出图）；步骤 3、4 各需数小时。
#    - 必须加 --no-capture-output，否则 conda run 会缓存输出到进程结束。
# ============================================================================

param(
    [string]   $Dataset = "",
    [string]   $Run     = "",
    [string]   $Root    = "",
    [string]   $Steps   = "1,7"
)

$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'

# 定位项目根目录（本脚本位于 <项目根>\ml_pipeline\）
# 注意：PowerShell 变量名大小写不敏感，项目根目录必须用别的变量名，
#       否则会把 -Root 参数覆盖掉。
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$scripts = [ordered]@{
    '1'  = 'step1_train_tabpfn.py'
    '3'  = 'step3_multiseed_evaluation.py'
    '4'  = 'step4_shap_all_targets.py'
    '5'  = 'step5_shap_key_targets.py'
    '5b' = 'step5b_shap_signed.py'
    '6'  = 'step6_learning_curve.py'
    '7'  = 'step7_make_figures.py'
}

# 组装公共的覆盖参数
$extra = @()
if ($Dataset) { $extra += @('--dataset', $Dataset) }
if ($Run)     { $extra += @('--run',     $Run) }
if ($Root)    { $extra += @('--root',    $Root) }

# 解析要执行的步骤（-Steps 传进来的始终是字符串，这里自行拆分更稳）
$wanted = @($Steps -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$plan = @()
foreach ($key in $scripts.Keys) {
    if ($wanted -contains $key) { $plan += $key }
}
if ($plan.Count -eq 0) {
    Write-Host "没有匹配的步骤（-Steps '$Steps'）。可用步骤: $($scripts.Keys -join ', ')" -ForegroundColor Red
    exit 1
}

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host " ml_pipeline 一键运行" -ForegroundColor Cyan
Write-Host " 项目根目录 : $projectRoot"
Write-Host " 数据集     : $(if ($Dataset) { $Dataset } else { '（用 config.py 的 active_dataset）' })"
Write-Host " 运行名称   : $(if ($Run) { $Run } else { '（用 config.py 的 run）' })"
Write-Host " 执行步骤   : $($plan -join ', ')"
Write-Host "==============================================================" -ForegroundColor Cyan

$overall = [System.Diagnostics.Stopwatch]::StartNew()
foreach ($key in $plan) {
    $file = $scripts[$key]
    $path = Join-Path $PSScriptRoot $file
    if (-not (Test-Path -LiteralPath $path)) {
        Write-Host "[跳过] 步骤 $key 缺少文件 $file" -ForegroundColor Yellow
        continue
    }
    Write-Host ""
    Write-Host ">>> 步骤 $key : $file" -ForegroundColor Green
    $stepTimer = [System.Diagnostics.Stopwatch]::StartNew()
    $arguments = @('run', '-n', 'tabpfn', '--no-capture-output', 'python', $path) + $extra
    & conda @arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Host "步骤 $key 失败（退出码 $LASTEXITCODE）。修复后重新运行同一命令即可从断点续跑。" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host ("<<< 步骤 {0} 完成，用时 {1:mm\:ss}" -f $key, $stepTimer.Elapsed) -ForegroundColor Green
}

Write-Host ""
Write-Host ("全部步骤完成，总用时 {0:hh\:mm\:ss}" -f $overall.Elapsed) -ForegroundColor Cyan
