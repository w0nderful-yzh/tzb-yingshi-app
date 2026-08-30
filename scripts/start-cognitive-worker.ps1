# 一键启动认知 MMSE Worker：wav2vec2 声学回归（守护期间累计 60-120 s 有效语音即出分）。
# 模型资产在仓库外本机目录（HF 格式完整），权重不进入 Git；路径不同用 -ModelDir 覆盖。
# 依赖：Python 3.10 环境（torch/transformers/librosa）。
param(
    [string]$PythonPath = "E:\python3.10.9aaa\python.exe",
    [string]$ModelDir = "E:\创新实践\老人摔倒预警\心理模块\代码包\补充\模型\wav2vec2_base_adress",
    [string]$Device = "cpu",
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)
$ErrorActionPreference = "Stop"

$BackendDir = Join-Path $RepoRoot "backend"

if (-not (Test-Path $PythonPath)) {
    Write-Host "[失败] 找不到 Python 解释器：$PythonPath（用 -PythonPath 覆盖）" -ForegroundColor Red
    exit 1
}
# 必须指向模型顶层（config.json + model.safetensors + preprocessor_config.json），不是 checkpoint 子目录。
foreach ($f in @("config.json", "model.safetensors", "preprocessor_config.json")) {
    if (-not (Test-Path (Join-Path $ModelDir $f))) {
        Write-Host "[失败] $ModelDir 缺少 $f" -ForegroundColor Red
        Write-Host "       -ModelDir 必须指向 wav2vec2_base_adress 顶层，不是 checkpoint-N 子目录。"
        exit 1
    }
}

$env:COGNITIVE_MODEL_DIR = $ModelDir

Write-Host "[1/2] 在新窗口启动认知 MMSE Worker（device=$Device）..." -ForegroundColor Cyan
$cmd = "& '$PythonPath' -m app.modules.psychology.cognitive.worker --device $Device"
Start-Process powershell -WorkingDirectory $BackendDir -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $cmd

# 等 8 s 确认进程没有"启动即退出"（典型原因：模型目录错、依赖缺失）
Start-Sleep -Seconds 8
$alive = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -like "*cognitive.worker*" }
if ($null -eq $alive) {
    Write-Host "[失败] Worker 进程未存活，请看新窗口报错（常见：COGNITIVE_MODEL_DIR 错、缺 torch/transformers/librosa）。" -ForegroundColor Red
    exit 1
}

Write-Host "[完成] 认知 MMSE Worker 已运行（请勿关闭新窗口）。" -ForegroundColor Green
Write-Host "  成功标志：新窗口出现 'Cognitive Worker started runtime_root=...'；"
Write-Host "  出分流程：App 点开始守护 -> 现场有 60-120 s 有效语音 -> 新窗口出现 'Cognitive assessment completed ... score=' -> Care 页出分。"
