$ErrorActionPreference = 'Stop'
try {
    Set-Location -LiteralPath $PSScriptRoot
    Write-Host '第三问联合调度候选版：手动演练入口' -ForegroundColor Cyan
    Write-Host '仅支持问题3。建议先使用问题3演练测试。'
    Write-Host '本窗口不会替你在模拟器中选择或启动测试。'
    $candidates = @(
        (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'),
        'D:\Python\python.exe',
        'python'
    )
    $pythonPath = $null
    foreach ($candidate in $candidates) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($command) {
            try {
                & $command.Source -c 'import sys,numpy; assert sys.version_info >= (3,11)' 2>$null
                if ($LASTEXITCODE -eq 0) { $pythonPath = $command.Source; break }
            } catch { continue }
        }
    }
    if (-not $pythonPath) {
        throw '未找到装有NumPy的Python 3.11或更新版本。请先按官方测试操作说明配置环境，再启动模拟器测试。'
    }
    Write-Host ('运行环境检查通过：' + $pythonPath)
    $team = Read-Host '请输入与模拟器当前登录完全一致的参赛队号（不是手机号）'
    if ([string]::IsNullOrWhiteSpace($team)) { throw '参赛队号不能为空。' }
    $portText = Read-Host '模拟器端口（未修改设置直接按回车，默认2026）'
    if ([string]::IsNullOrWhiteSpace($portText)) { $portText = '2026' }
    $portNumber = 0
    if (-not [int]::TryParse($portText, [ref]$portNumber) -or $portNumber -lt 1 -or $portNumber -gt 65535) {
        throw '端口必须是1到65535的整数。'
    }
    Write-Host ''
    Write-Host '现在回到模拟器，选择“问题3演练测试”并启动。'
    Write-Host '等5秒倒计时完全结束、测试窗口开启后，再回到本窗口。' -ForegroundColor Yellow
    $null = Read-Host '确认模拟器已就绪后按回车，程序才会连接并执行本局'
    $logDir = Join-Path $PSScriptRoot 'official_runs'
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $suffix = [guid]::NewGuid().ToString('N').Substring(0,8)
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $logPath = Join-Path $logDir ('q3-joint-' + $stamp + '-' + $suffix + '.jsonl')
    $url = 'http://127.0.0.1:' + $portNumber
    Write-Host ('正在执行第三问，本次日志：' + $logPath)
    & $pythonPath -X utf8 (Join-Path $PSScriptRoot 'joint_official_client.py') --robot-id $team --url $url --log $logPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host '本次程序出现错误。保留窗口与日志，检查模拟器状态后再处理；不要在同一局直接重复启动。' -ForegroundColor Red
    }
    else {
        Write-Host '请同时查看模拟器的结束提示和演练干扰源总数。正式测试的加密日志需由模拟器导出。'
    }
}
catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
}
$null = Read-Host '按回车关闭此窗口'
