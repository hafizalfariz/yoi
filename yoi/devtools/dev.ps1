param(
    [ValidateSet('up','up-cpu','up-gpu','build-cpu','build-gpu','run-local','up-builder','down-builder','restart','logs','watch','cleanup','qa','qa-fix','limits')]
    [string]$Action = 'up',
    [ValidateSet('cpu','gpu')]
    [string]$Profile = 'cpu',
    [switch]$NoBuild = $true,
    [int]$Tail = 120,
    [switch]$Follow,
    [ValidateSet('ffplay','vlc')]
    [string]$Player = 'vlc',
    [string]$Url = 'rtsp://localhost:6554/kluis-line',
    [int]$NetworkCachingMs = 150,
    [string]$Config = 'configs/app/dwelltime.yaml',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

# Stabilize local Windows Docker builds (avoid intermittent BuildKit snapshot errors)
$env:DOCKER_BUILDKIT = '0'
$env:COMPOSE_DOCKER_CLI_BUILD = '0'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$composeFiles = @('-f', (Join-Path $repoRoot 'docker-compose.yml'), '-f', (Join-Path $repoRoot 'docker-compose.dev.yml'))
$service = if ($Profile -eq 'gpu') { 'app-yoi-gpu' } else { 'app-yoi-cpu' }

function Get-DotEnvValue {
    param(
        [string]$Key,
        [string]$Default = ''
    )

    $envFile = Join-Path $repoRoot '.env'
    if (-not (Test-Path $envFile)) {
        return $Default
    }

    $line = Get-Content $envFile | Where-Object { $_ -match "^\s*$Key\s*=" } | Select-Object -First 1
    if (-not $line) {
        return $Default
    }

    $value = ($line -split '=', 2)[1].Trim()
    if ($value.StartsWith('"') -and $value.EndsWith('"')) {
        $value = $value.Trim('"')
    }
    return $value
}

function Get-ConfigValue {
    param(
        [string]$Key,
        [string]$Default = ''
    )

    $envValue = [System.Environment]::GetEnvironmentVariable($Key)
    if (-not [string]::IsNullOrWhiteSpace($envValue)) {
        return $envValue
    }

    return Get-DotEnvValue -Key $Key -Default $Default
}

function Get-LogicalCoreCount {
    return [int][System.Environment]::ProcessorCount
}

function Get-TotalMemoryMb {
    try {
        $os = Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction Stop
        $totalKb = [double]$os.TotalVisibleMemorySize
        if ($totalKb -gt 0) {
            return [int][math]::Floor($totalKb / 1024)
        }
    }
    catch {
        # Fallback below
    }

    try {
        $cs = Get-CimInstance -ClassName Win32_ComputerSystem -ErrorAction Stop
        $totalBytes = [double]$cs.TotalPhysicalMemory
        if ($totalBytes -gt 0) {
            return [int][math]::Floor($totalBytes / 1MB)
        }
    }
    catch {
        # Ignore and fail below
    }

    throw 'Cannot detect total system memory for MEMORY_LIMIT_PERCENT conversion'
}

function Convert-PercentToCpuLimit {
    param(
        [string]$PercentRaw,
        [int]$CoreCount
    )

    if ([string]::IsNullOrWhiteSpace($PercentRaw)) {
        return $null
    }

    $normalized = $PercentRaw.Trim().ToLower()
    if ($normalized -eq 'max') {
        return [math]::Round([double]$CoreCount, 1)
    }
    if ($normalized.EndsWith('%')) {
        $normalized = $normalized.TrimEnd('%')
    }

    $percent = 0.0
    if (-not [double]::TryParse($normalized, [ref]$percent)) {
        throw "Invalid percent value: '$PercentRaw'"
    }

    if ($percent -lt 10 -or $percent -gt 100) {
        throw "Percent must be between 10 and 100 (or 'max'), got '$PercentRaw'"
    }

    $limit = ($CoreCount * $percent) / 100.0
    return [math]::Round($limit, 1)
}

function Convert-PercentToMemoryLimit {
    param(
        [string]$PercentRaw,
        [int]$TotalMemoryMb
    )

    if ([string]::IsNullOrWhiteSpace($PercentRaw)) {
        return $null
    }

    $normalized = $PercentRaw.Trim().ToLower()
    if ($normalized.EndsWith('%')) {
        $normalized = $normalized.TrimEnd('%')
    }

    $percent = 0.0
    if ($normalized -eq 'max') {
        $percent = 100.0
    }
    elseif (-not [double]::TryParse($normalized, [ref]$percent)) {
        throw "Invalid MEMORY_LIMIT_PERCENT value: '$PercentRaw'"
    }

    if ($percent -lt 10 -or $percent -gt 100) {
        throw "MEMORY_LIMIT_PERCENT must be between 10 and 100 (or 'max'), got '$PercentRaw'"
    }

    $limitMb = [int][math]::Max(512, [math]::Floor(($TotalMemoryMb * $percent) / 100.0))
    return "${limitMb}m"
}

function Resolve-ComposeMemoryLimit {
    $percentRaw = Get-ConfigValue -Key 'MEMORY_LIMIT_PERCENT' -Default ''

    if ([string]::IsNullOrWhiteSpace($percentRaw)) {
        Remove-Item Env:MEMORY_LIMIT -ErrorAction SilentlyContinue
        Write-Host "[dev] MEMORY_LIMIT_PERCENT is empty; using compose default mem_limit"
        return
    }

    $totalMb = Get-TotalMemoryMb
    $resolved = Convert-PercentToMemoryLimit -PercentRaw $percentRaw -TotalMemoryMb $totalMb
    $env:MEMORY_LIMIT = $resolved
    Write-Host "[dev] Resolved MEMORY_LIMIT from MEMORY_LIMIT_PERCENT=$percentRaw => $resolved (host=${totalMb}m)"
}

function Show-CpuLimitTable {
    $coreCount = Get-LogicalCoreCount
    Write-Host "[dev] Logical cores detected: $coreCount"
    Write-Host "[dev] Percent -> CPU limit (cores)"

    for ($p = 10; $p -le 100; $p += 10) {
        $limit = Convert-PercentToCpuLimit -PercentRaw "$p" -CoreCount $coreCount
        Write-Host ("[dev] {0,3}% => {1}" -f $p, $limit)
    }

    $maxLimit = Convert-PercentToCpuLimit -PercentRaw 'max' -CoreCount $coreCount
    Write-Host "[dev] max => $maxLimit"
}

function Apply-LocalThreadLimitsFromPercent {
    param(
        [ValidateSet('cpu','gpu')]
        [string]$TargetProfile
    )

    $coreCount = Get-LogicalCoreCount
    $percentKey = if ($TargetProfile -eq 'gpu') { 'GPU_CPU_LIMIT_PERCENT' } else { 'CPU_LIMIT_PERCENT' }
    $percentRaw = Get-ConfigValue -Key $percentKey -Default ''

    if ([string]::IsNullOrWhiteSpace($percentRaw)) {
        return
    }

    $threadBudget = [int][math]::Max(
        1,
        [math]::Floor((Convert-PercentToCpuLimit -PercentRaw $percentRaw -CoreCount $coreCount))
    )
    $opencvThreads = [int][math]::Max(1, [math]::Min(4, [math]::Floor($threadBudget / 2)))

    $env:OMP_NUM_THREADS = "$threadBudget"
    $env:OPENBLAS_NUM_THREADS = "$threadBudget"
    $env:MKL_NUM_THREADS = "$threadBudget"
    $env:NUMEXPR_NUM_THREADS = "$threadBudget"
    $env:OPENCV_FOR_THREADS_NUM = "$opencvThreads"

    Write-Host "[dev] Local thread cap from $percentKey=$percentRaw => OMP/BLAS/MKL/NUMEXPR=$threadBudget, OpenCV=$opencvThreads"
}

function Invoke-Compose {
    param([string[]]$ComposeActionArgs)
    $cmd = 'docker compose ' + (($composeFiles + $ComposeActionArgs) -join ' ')
    Write-Host "[dev] $cmd"
    if (-not $DryRun) {
        docker compose @composeFiles @ComposeActionArgs
        if ($LASTEXITCODE -ne 0) {
            throw "docker compose failed with exit code $LASTEXITCODE"
        }
    }
}

function Resolve-PythonCommand {
    $venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
    if (Test-Path $venvPython) {
        return $venvPython
    }
    return 'python'
}

function Invoke-PythonCommand {
    param([string[]]$CommandArgs)

    $pythonCmd = Resolve-PythonCommand
    $displayCmd = "$pythonCmd $($CommandArgs -join ' ')"
    Write-Host "[dev] $displayCmd"
    if (-not $DryRun) {
        & $pythonCmd @CommandArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Python command failed with exit code $LASTEXITCODE"
        }
    }
}

function Start-Profile {
    param(
        [ValidateSet('cpu','gpu')]
        [string]$TargetProfile,
        [bool]$UseNoBuild = $true
    )

    $upArgs = @('--profile', $TargetProfile, 'up', '-d')
    if ($UseNoBuild) {
        $upArgs += '--no-build'
    } else {
        $upArgs += '--build'
    }

    Resolve-ComposeMemoryLimit

    $maxAttempts = 3
    for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
        try {
            Invoke-Compose -ComposeActionArgs $upArgs
            break
        }
        catch {
            if ($attempt -ge $maxAttempts) {
                throw
            }

            $delaySeconds = 4 * $attempt
            Write-Host "[dev] compose up attempt $attempt/$maxAttempts failed, retrying in ${delaySeconds}s ..."
            Start-Sleep -Seconds $delaySeconds
        }
    }

    Invoke-Compose -ComposeActionArgs @('ps')
    Write-Host "[dev] Profile '$TargetProfile' started (no-build=$UseNoBuild)"
}

function Start-LocalRuntime {
    param(
        [string]$ConfigPath,
        [ValidateSet('cpu','gpu')]
        [string]$TargetProfile = 'cpu'
    )

    $resolvedConfig = $ConfigPath
    if (-not [System.IO.Path]::IsPathRooted($ConfigPath)) {
        $resolvedConfig = Join-Path $repoRoot $ConfigPath
    }

    if (-not (Test-Path $resolvedConfig)) {
        throw "Config file not found: $resolvedConfig"
    }

    $env:YOI_RUNTIME_PROFILE = $TargetProfile
    if (-not $env:YOI_TARGET_DEVICE) {
        $env:YOI_TARGET_DEVICE = if ($TargetProfile -eq 'gpu') { 'cuda' } else { 'cpu' }
    }

    Apply-LocalThreadLimitsFromPercent -TargetProfile $TargetProfile

    Write-Host "[dev] Local runtime profile: $TargetProfile (target device: $env:YOI_TARGET_DEVICE)"
    Write-Host "[dev] Local runtime config: $resolvedConfig"
    Invoke-PythonCommand -CommandArgs @('src/app/main.py', '--config', $resolvedConfig)
}

function Cleanup-LegacyLogs {
    $patterns = @(
        (Join-Path $repoRoot 'logs\ffmpeg_startup_*.log')
    )

    foreach ($pattern in $patterns) {
        $files = Get-ChildItem -Path $pattern -File -ErrorAction SilentlyContinue
        foreach ($file in $files) {
            Write-Host "[dev] removing legacy log: $($file.FullName)"
            if (-not $DryRun) {
                Remove-Item -Path $file.FullName -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

function Cleanup-OldOutputRuns {
    $retentionDaysRaw = Get-DotEnvValue -Key 'YOI_OUTPUT_RETENTION_DAYS' -Default '30'
    $retentionDays = 30
    if (-not [int]::TryParse($retentionDaysRaw, [ref]$retentionDays)) {
        $retentionDays = 30
    }
    if ($retentionDays -lt 1) {
        return
    }

    $outputRootRaw = Get-DotEnvValue -Key 'OUTPUT_PATH' -Default './output'
    $outputRoot = $outputRootRaw
    if (-not [System.IO.Path]::IsPathRooted($outputRootRaw)) {
        $outputRoot = Join-Path $repoRoot $outputRootRaw
    }

    if (-not (Test-Path $outputRoot)) {
        return
    }

    $cutoff = (Get-Date).AddDays(-$retentionDays)
    $oldRuns = Get-ChildItem -Path $outputRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -lt $cutoff }

    foreach ($runDir in $oldRuns) {
        Write-Host "[dev] removing old output run (> $retentionDays days): $($runDir.FullName)"
        if (-not $DryRun) {
            Remove-Item -Path $runDir.FullName -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

Push-Location $repoRoot
try {
    Cleanup-LegacyLogs
    Cleanup-OldOutputRuns

    switch ($Action) {
        'cleanup' {
            Write-Host '[dev] cleanup completed (legacy logs + old output runs)'
        }

        'qa' {
            Invoke-PythonCommand -CommandArgs @('-m', 'ruff', 'check', 'yoi', 'src', 'config_builder', 'tests', '--select', 'I,F401,F841')
            Invoke-PythonCommand -CommandArgs @('-m', 'pytest', '-q')
            Write-Host '[dev] QA completed (ruff + pytest)'
        }

        'qa-fix' {
            Invoke-PythonCommand -CommandArgs @('-m', 'ruff', 'check', 'yoi', 'src', 'config_builder', 'tests', '--select', 'I', '--fix')
            Invoke-PythonCommand -CommandArgs @('-m', 'ruff', 'check', 'yoi', 'src', 'config_builder', 'tests', '--select', 'I,F401,F841')
            Invoke-PythonCommand -CommandArgs @('-m', 'pytest', '-q')
            Write-Host '[dev] QA-FIX completed (ruff --fix + ruff + pytest)'
        }

        'limits' {
            Show-CpuLimitTable
        }

        'up' {
            Start-Profile -TargetProfile $Profile -UseNoBuild:$NoBuild
            Write-Host '[dev] RTSP endpoint: rtsp://localhost:6554/kluis-line'

            $autoPopup = (Get-DotEnvValue -Key 'YOI_AUTO_POPUP_RTSP' -Default '0').ToLower()
            if ($autoPopup -in @('1','true','on','yes') -and -not $DryRun) {
                $popupPlayer = Get-DotEnvValue -Key 'YOI_AUTO_POPUP_RTSP_PLAYER' -Default $Player
                $popupUrl = Get-DotEnvValue -Key 'YOI_AUTO_POPUP_RTSP_URL' -Default $Url
                Write-Host "[dev] Auto popup RTSP enabled -> player=$popupPlayer url=$popupUrl"
                Start-Process powershell -ArgumentList @(
                    '-ExecutionPolicy', 'Bypass',
                    '-File', $PSCommandPath,
                    '-Action', 'watch',
                    '-Player', $popupPlayer,
                    '-Url', $popupUrl
                ) | Out-Null
            }
        }

        'up-cpu' {
            Start-Profile -TargetProfile 'cpu' -UseNoBuild:$true
            Write-Host '[dev] Fast CPU up complete (no build)'
        }

        'up-gpu' {
            Start-Profile -TargetProfile 'gpu' -UseNoBuild:$true
            Write-Host '[dev] Fast GPU up complete (no build)'
        }

        'build-cpu' {
            Start-Profile -TargetProfile 'cpu' -UseNoBuild:$false
            Write-Host '[dev] CPU build+up complete'
        }

        'build-gpu' {
            Start-Profile -TargetProfile 'gpu' -UseNoBuild:$false
            Write-Host '[dev] GPU build+up complete'
        }

        'run-local' {
            Start-LocalRuntime -ConfigPath $Config -TargetProfile $Profile
        }

        'up-builder' {
            $upArgs = @('--profile', 'builder', 'up', '-d', 'config-builder')
            if ($NoBuild) {
                $upArgs += '--no-build'
            } else {
                $upArgs += '--build'
            }
            Invoke-Compose -ComposeActionArgs $upArgs
            Invoke-Compose -ComposeActionArgs @('ps', 'config-builder')

            $builderPort = Get-DotEnvValue -Key 'CONFIG_BUILDER_PORT' -Default '8032'
            Write-Host "[dev] Config Builder endpoint: http://localhost:$builderPort"
        }

        'down-builder' {
            Invoke-Compose -ComposeActionArgs @('--profile', 'builder', 'stop', 'config-builder')
            Invoke-Compose -ComposeActionArgs @('ps', 'config-builder')
            Write-Host '[dev] Config Builder stopped'
        }

        'restart' {
            Invoke-Compose -ComposeActionArgs @('restart', $service)
            Invoke-Compose -ComposeActionArgs @('logs', '--tail=80', $service, 'mediamtx')
        }

        'logs' {
            $logArgs = @('logs', '--tail', $Tail)
            if ($Follow) { $logArgs += '-f' }
            $logArgs += @($service, 'mediamtx')
            Invoke-Compose -ComposeActionArgs $logArgs
        }

        'watch' {
            Write-Host "[dev] Watch RTSP -> $Url ($Player)"
            if ($Player -eq 'ffplay') {
                $ffplay = Get-Command ffplay -ErrorAction SilentlyContinue
                if (-not $ffplay) { throw 'ffplay tidak ditemukan di PATH' }
                if (-not $DryRun) {
                    & $ffplay.Source -rtsp_transport tcp -fflags nobuffer -flags low_delay -framedrop $Url
                    exit $LASTEXITCODE
                }
            } else {
                $vlc = Get-Command vlc -ErrorAction SilentlyContinue
                if (-not $vlc) { throw 'vlc tidak ditemukan di PATH' }
                if (-not $DryRun) {
                    & $vlc.Source "--network-caching=$NetworkCachingMs" $Url
                    exit $LASTEXITCODE
                }
            }
        }
    }
}
finally {
    Pop-Location
}
