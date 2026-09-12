param([Parameter(Mandatory=$true)][ValidateSet('q3','q4','share25')][string]$Problem)
$ErrorActionPreference = 'Stop'
$runExitCode = 1
try {
    $projectRoot = Split-Path -Parent $PSScriptRoot
    Set-Location -LiteralPath $projectRoot
    $pythonPath = $null
    foreach ($candidate in @('D:\Python\python.exe','python')) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($command) {
            try {
                & $command.Source -c 'import sys,numpy,scipy; assert sys.version_info >= (3,11)' 2>$null
                if ($LASTEXITCODE -eq 0) { $pythonPath=$command.Source; break }
            } catch { continue }
        }
    }
    if (-not $pythonPath) { throw '需要安装NumPy和SciPy的Python 3.11或更新版本。' }
    Write-Host '准备所选方案并核验版本，此时不连接模拟器。'
    $runtimePath = & $pythonPath -X utf8 (Join-Path $PSScriptRoot 'run_study.py') prepare
    if ($LASTEXITCODE -ne 0) { throw '运行目录准备失败。' }
    $runtimePath = [string]$runtimePath
    switch ($Problem) {
        'q3' { $folder='src\q3_model_v2'; $verify='verify_main_solution.py'; $client='official_client.py'; $label='第三问 time/lean'; $module='问题3演练测试' }
        'q4' { $folder='src\q4_model'; $verify='verify_release.py'; $client='q4_official_client.py'; $label='第四问 shared'; $module='问题4演练测试' }
        'share25' { $folder='src\q4_model'; $verify='verify_share25_release.py'; $client='q4_share25_client.py'; $label='第四问 share25 候选'; $module='问题4演练测试' }
    }
    $modulePath = Join-Path $runtimePath $folder
    & $pythonPath -X utf8 (Join-Path $modulePath $verify)
    if ($LASTEXITCODE -ne 0) { throw '发布校验未通过；尚未连接模拟器。' }
    Write-Host $label -ForegroundColor Cyan
    $team = Read-Host '请输入与模拟器登录一致的参赛队号（不是手机号）'
    if ([string]::IsNullOrWhiteSpace($team)) { throw '参赛队号不能为空。' }
    $portText = Read-Host '模拟器端口（默认2026，直接回车）'
    if ([string]::IsNullOrWhiteSpace($portText)) { $portText='2026' }
    $port = 0
    if (-not [int]::TryParse($portText,[ref]$port) -or $port -lt 1 -or $port -gt 65535) { throw '端口须为1至65535的整数。' }
    Write-Host ('请在模拟器中手动启动“'+$module+'”，并等待倒计时完全结束。') -ForegroundColor Yellow
    $null = Read-Host '确认相应演练接口已就绪后按回车，程序才会执行本局'
    $logDir = Join-Path $projectRoot ('local_data\official_runs\'+$Problem)
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $suffix = [guid]::NewGuid().ToString('N').Substring(0,8)
    $logPath = Join-Path $logDir ($Problem+'-'+$stamp+'-'+$suffix+'.jsonl')
    & $pythonPath -X utf8 (Join-Path $modulePath $client) --robot-id $team --url ('http://127.0.0.1:'+$port) --log $logPath
    $runExitCode = $LASTEXITCODE
    if ($runExitCode -ne 0) { Write-Host '本次未确认正常完成，请保留日志并查看模拟器状态，不要直接重复启动同一局。' -ForegroundColor Red }
    Write-Host ('日志：'+$logPath)
    Write-Host '请同时保存官方源总数、清除数和模拟器汇总。'
}
catch { $runExitCode = 1; Write-Host $_.Exception.Message -ForegroundColor Red }
$null = Read-Host '按回车关闭窗口'

exit $runExitCode
