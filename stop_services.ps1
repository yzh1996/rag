$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$projectRoot = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\', '/')
$escapedRoot = [regex]::Escape($projectRoot)
$pythonPattern = '(?i)^"?' + $escapedRoot + '\\\.venv\\Scripts\\pythonw?\.exe"?(?=\s|$)'
$scriptPatterns = @{
    '后端' = @{
        Absolute = '(?i)(?:^|\s)"?' + $escapedRoot + '\\run_api\.py"?(?=\s|$)'
        Relative = '(?i)(?:^|\s)"?(?:\.\\)?run_api\.py"?(?=\s|$)'
    }
    '前端' = @{
        Absolute = '(?i)(?:^|\s)"?' + $escapedRoot + '\\run_frontend\.py"?(?=\s|$)'
        Relative = '(?i)(?:^|\s)"?(?:\.\\)?run_frontend\.py"?(?=\s|$)'
    }
}
$streamlitAbsolute = '(?i)\s-m\s+streamlit\s+run\s+"?' + $escapedRoot + '\\frontend\\app\.py"?(?=\s|$)'
$streamlitRelative = '(?i)\s-m\s+streamlit\s+run\s+"?(?:\.\\)?frontend\\app\.py"?(?=\s|$)'

function Get-ProjectServices {
    param([object[]]$Processes)

    foreach ($process in $Processes) {
        if ($process.Name -notmatch '^pythonw?\.exe$' -or -not $process.CommandLine) {
            continue
        }

        $command = $process.CommandLine.Replace('/', '\')
        $usesProjectPython = $command -match $pythonPattern
        foreach ($label in @('后端', '前端')) {
            $pattern = $scriptPatterns[$label]
            $matchesScript = $command -match $pattern.Absolute -or
                ($usesProjectPython -and $command -match $pattern.Relative)
            $matchesStreamlit = $label -eq '前端' -and
                ($command -match $streamlitAbsolute -or
                    ($usesProjectPython -and $command -match $streamlitRelative))
            if ($matchesScript -or $matchesStreamlit) {
                [PSCustomObject]@{ Process = $process; Label = $label }
                break
            }
        }
    }
}

try {
    Write-Host "项目：$projectRoot"
    Write-Host '正在查找本项目的前后端服务……'

    $processes = @(Get-CimInstance Win32_Process)
    $services = @(Get-ProjectServices -Processes $processes)
    $targets = @{}
    foreach ($service in $services) {
        $targets[[int]$service.Process.ProcessId] = $service.Process
    }

    # Include children so orphaned Python/Streamlit processes are also verified.
    do {
        $added = $false
        foreach ($process in $processes) {
            if (-not $targets.ContainsKey([int]$process.ProcessId) -and
                $targets.ContainsKey([int]$process.ParentProcessId) -and
                $process.CreationDate -ge $targets[[int]$process.ParentProcessId].CreationDate) {
                $targets[[int]$process.ProcessId] = $process
                $added = $true
            }
        }
    } while ($added)

    foreach ($label in @('后端', '前端')) {
        if (@($services | Where-Object { $_.Label -eq $label }).Count -eq 0) {
            Write-Host "$label 服务未运行，或未发现能确认属于本项目的进程。"
        }
    }

    # Stop only the highest matching launchers; taskkill also stops their children.
    $roots = @($services | Where-Object {
        -not $targets.ContainsKey([int]$_.Process.ParentProcessId)
    })
    foreach ($service in $roots) {
        $original = $service.Process
        $current = Get-CimInstance Win32_Process -Filter "ProcessId = $($original.ProcessId)"
        if ($null -eq $current) {
            continue
        }
        if ($current.CreationDate -ne $original.CreationDate -or
            $current.CommandLine -ne $original.CommandLine) {
            throw "进程 $($original.ProcessId) 已发生变化，为避免误关其他程序，已停止操作。"
        }

        Write-Host "正在停止$($service.Label)及相关子进程（PID $($original.ProcessId)）……"
        try {
            $killOutput = & "$env:SystemRoot\System32\taskkill.exe" /PID $original.ProcessId /T /F 2>&1
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "停止命令未成功，将检查进程是否仍在运行。$killOutput"
            }
        }
        catch {
            Write-Warning "停止命令未成功，将检查进程是否仍在运行。$($_.Exception.Message)"
        }
    }

    $after = @(Get-CimInstance Win32_Process)
    $remaining = @(Get-ProjectServices -Processes $after)
    $originalRemaining = @($after | Where-Object {
        $targets.ContainsKey([int]$_.ProcessId) -and
            $_.CreationDate -eq $targets[[int]$_.ProcessId].CreationDate
    })
    $failed = $remaining.Count -gt 0 -or $originalRemaining.Count -gt 0
    if ($failed) {
        Write-Host '部分服务未能停止。请关闭对应启动窗口；如提示权限不足，请右键以管理员身份运行此脚本。' -ForegroundColor Red
    }
    elseif ($services.Count -gt 0) {
        Write-Host '本项目的前后端服务及相关子进程已停止。' -ForegroundColor Green
    }

    # Ports are checked for information only; never kill a process by port alone.
    $ports = @{ API_PORT = 8000; STREAMLIT_PORT = 8501 }
    $envFile = Join-Path $projectRoot '.env'
    if (Test-Path -LiteralPath $envFile) {
        foreach ($line in Get-Content -LiteralPath $envFile -Encoding UTF8) {
            if ($line -match '^\s*(API_PORT|STREAMLIT_PORT)\s*=\s*["'']?(\d+)["'']?\s*(?:#.*)?$') {
                $port = [int]$Matches[2]
                if ($port -ge 1 -and $port -le 65535) {
                    $ports[$Matches[1]] = $port
                }
            }
        }
    }
    try {
        $listeners = @(Get-NetTCPConnection -State Listen -ErrorAction Stop)
        foreach ($key in @('API_PORT', 'STREAMLIT_PORT')) {
            $port = $ports[$key]
            $owners = @($listeners | Where-Object { $_.LocalPort -eq $port })
            if ($owners.Count -eq 0) {
                Write-Host "端口 $port 已释放。"
            }
            else {
                $ownerIds = ($owners | Select-Object -ExpandProperty OwningProcess -Unique) -join ', '
                Write-Host "端口 $port 仍被占用（PID $ownerIds）。未确认属于本项目的程序不会被关闭。" -ForegroundColor Yellow
            }
        }
    }
    catch {
        Write-Warning "无法检查端口状态；已完成进程检查。$($_.Exception.Message)"
    }

    if ($failed) {
        exit 1
    }
    exit 0
}
catch {
    Write-Host "停止服务失败：$($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
