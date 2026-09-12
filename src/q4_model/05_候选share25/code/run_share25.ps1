$ErrorActionPreference = 'Stop'
$runExitCode = 1
try {
    Set-Location -LiteralPath $PSScriptRoot
    Write-Host '第四问候选：share25（25站覆盖、任务重排与服务停点复用）' -ForegroundColor Cyan
    Write-Host '此入口运行 share25 候选；原“运行第四问.cmd”仍对应 shared 稳定版。' -ForegroundColor Yellow
    Write-Host '请使用问题4演练模块；本窗口不会替你选择或启动模拟器测试。'
    $candidates = @('D:\Python\python.exe', 'python')
    $pythonPath = $null
    foreach ($candidate in $candidates) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($command) {
            try {
                & $command.Source -c 'import sys,numpy,scipy; assert sys.version_info >= (3,11)' 2>$null
                if ($LASTEXITCODE -eq 0) { $pythonPath = $command.Source; break }
            } catch { continue }
        }
    }
    if (-not $pythonPath) { throw '需要装有 NumPy 和 SciPy 的 Python 3.11 或更新版本，请先按操作说明配置环境。' }
    & $pythonPath -X utf8 (Join-Path $PSScriptRoot 'verify_share25_release.py')
    if ($LASTEXITCODE -ne 0) { throw 'share25 候选版本校验未通过，尚未连接模拟器。' }
    $team = Read-Host '请输入与模拟器登录一致的参赛队号（不是手机号）'
    if ([string]::IsNullOrWhiteSpace($team)) { throw '参赛队号不能为空。' }
    $portText = Read-Host '模拟器端口（默认2026，直接按回车）'
    if ([string]::IsNullOrWhiteSpace($portText)) { $portText = '2026' }
    $portNumber = 0
    if (-not [int]::TryParse($portText, [ref]$portNumber) -or $portNumber -lt 1 -or $portNumber -gt 65535) {
        throw '端口必须是1到65535的整数。'
    }
    Write-Host '现在在模拟器中选择“问题4演练测试”并启动。' -ForegroundColor Yellow
    Write-Host '等倒计时完全结束、测试窗口开启后，再回到此窗口。'
    $null = Read-Host '确认问题4演练接口已就绪后按回车，share25 候选才会连接并执行本局'
    $logDir = Join-Path (Join-Path $PSScriptRoot 'official_runs') 'share25'
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $suffix = [guid]::NewGuid().ToString('N').Substring(0,8)
    $logPath = Join-Path $logDir ('q4-share25-' + $stamp + '-' + $suffix + '.jsonl')
    $url = 'http://127.0.0.1:' + $portNumber
    & $pythonPath -X utf8 (Join-Path $PSScriptRoot 'q4_share25_client.py') --robot-id $team --url $url --log $logPath
    $runExitCode = $LASTEXITCODE
    if ($runExitCode -ne 0) {
        Write-Host ('share25 候选未确认正常完成，退出状态：' + $runExitCode) -ForegroundColor Red
        Write-Host '请保留窗口和日志，查看模拟器状态；不要直接在同一局重复启动。' -ForegroundColor Red
    } else {
        Write-Host 'share25 候选本次运行完成。请保存模拟器结果，并记下官方源总数、清除数和程序运行时间。'
    }
    Write-Host ('本次候选日志：' + $logPath)
}
catch {
    $runExitCode = 1
    Write-Host $_.Exception.Message -ForegroundColor Red
}
$null = Read-Host '按回车关闭窗口'
exit $runExitCode
