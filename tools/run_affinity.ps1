param(
    [Parameter(Mandatory = $true)]
    [string]$Mask,
    [string]$MetadataPath,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TaskArgs
)

$ErrorActionPreference = 'Stop'
if (-not $TaskArgs -or $TaskArgs.Count -eq 0) { throw 'Supply the original tools/run.ps1 Python arguments after -Mask and optional -MetadataPath.' }
if ($TaskArgs[0] -eq '--') { $TaskArgs = $TaskArgs[1..($TaskArgs.Count - 1)] }
if (-not $TaskArgs -or $TaskArgs.Count -eq 0) { throw 'Missing Python task arguments.' }
if ($MetadataPath -and (Test-Path -LiteralPath $MetadataPath)) { throw 'MetadataPath already exists; preserve the earlier protocol record.' }

$requestedMask = if ($Mask -match '^0[xX]([0-9a-fA-F]+)$') {
    [Convert]::ToUInt64($Matches[1], 16)
} elseif ($Mask -match '^\d+$') {
    [UInt64]::Parse($Mask, [Globalization.CultureInfo]::InvariantCulture)
} else { throw 'Mask must be a positive decimal integer or 0x-prefixed hexadecimal integer.' }
if ($requestedMask -eq 0) { throw 'The affinity mask must select at least one available logical processor.' }

$currentProcess = [Diagnostics.Process]::GetCurrentProcess()
$originalAffinity = $currentProcess.ProcessorAffinity
$originalMask = [BitConverter]::ToUInt64([BitConverter]::GetBytes($originalAffinity.ToInt64()), 0)
if (($requestedMask -band $originalMask) -ne $requestedMask) {
    throw ('Requested mask 0x{0:X} is outside the current available mask 0x{1:X}.' -f $requestedMask, $originalMask)
}
$runner = Join-Path $PSScriptRoot 'run.ps1'
$shellPath = (Get-Process -Id $PID).Path
$startedUtc = [DateTime]::UtcNow
$timer = [Diagnostics.Stopwatch]::StartNew()
$taskExitCode = 1
$record = $null

function Save-AffinityRecord {
    param($Record)
    if ($MetadataPath) {
        $absoluteMetadataPath = [IO.Path]::GetFullPath($MetadataPath)
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($absoluteMetadataPath)) | Out-Null
        $Record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $absoluteMetadataPath -Encoding utf8
    }
}

try {
    $signedMask = [BitConverter]::ToInt64([BitConverter]::GetBytes($requestedMask), 0)
    $currentProcess.ProcessorAffinity = [IntPtr]::new($signedMask)
    $currentProcess.Refresh()
    $actualMask = [BitConverter]::ToUInt64([BitConverter]::GetBytes($currentProcess.ProcessorAffinity.ToInt64()), 0)
    if ($actualMask -ne $requestedMask) { throw 'Current-process affinity verification failed.' }

    # This independent child uses the unchanged runner and selected Python.
    # Its successful inherited mask is recorded before importing any worker.
    $probeCode = 'import ctypes,json,os; from ctypes import wintypes; k=ctypes.WinDLL("kernel32",use_last_error=True); k.GetCurrentProcess.restype=wintypes.HANDLE; k.GetProcessAffinityMask.argtypes=[wintypes.HANDLE,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]; a=ctypes.c_size_t(); s=ctypes.c_size_t(); ok=k.GetProcessAffinityMask(k.GetCurrentProcess(),ctypes.byref(a),ctypes.byref(s)); assert ok,ctypes.get_last_error(); print(json.dumps(dict(pid=os.getpid(),mask_decimal=a.value,mask_hex=hex(a.value),system_mask_hex=hex(s.value))))'
    $probeOutput = @(& $shellPath -NoProfile -File $runner -c $probeCode)
    if ($LASTEXITCODE -ne 0) { throw ('Affinity child probe failed with exit code ' + $LASTEXITCODE) }
    $probe = ($probeOutput -join "`n") | ConvertFrom-Json
    if ([UInt64]$probe.mask_decimal -ne $requestedMask) { throw 'Selected Python did not inherit the requested affinity mask.' }
    $record = [ordered]@{
        kind = 'explicit_process_affinity_protocol'
        status = 'running'
        started_utc = $startedUtc.ToString('o')
        wrapper_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
        unchanged_runner_sha256 = (Get-FileHash -LiteralPath $runner -Algorithm SHA256).Hash.ToLowerInvariant()
        wrapper_pid = $PID
        original_available_mask_hex = ('0x{0:X}' -f $originalMask)
        requested_mask_hex = ('0x{0:X}' -f $requestedMask)
        requested_mask_decimal = $requestedMask
        verified_current_mask_hex = ('0x{0:X}' -f $actualMask)
        python_child_probe = $probe
        python_arguments = @($TaskArgs)
        thread_environment_unchanged = @{ OMP_NUM_THREADS=$env:OMP_NUM_THREADS; MKL_NUM_THREADS=$env:MKL_NUM_THREADS }
        scope = 'Only this PowerShell process is changed; children inherit its mask. No other process, priority, power policy or thread-count setting is changed. Python confirmation is an independent preflight child using the same runner.'
    }
    Save-AffinityRecord $record
    Write-Output ('AFFINITY_METADATA ' + ($record | ConvertTo-Json -Depth 8 -Compress))
    & $shellPath -NoProfile -File $runner @TaskArgs
    $taskExitCode = $LASTEXITCODE
    $record.status = if ($taskExitCode -eq 0) { 'complete' } else { 'task_failed' }
    $record.task_exit_code = $taskExitCode
    $record.completed_utc = [DateTime]::UtcNow.ToString('o')
    $record.wrapper_elapsed_wall_s = $timer.Elapsed.TotalSeconds
    Save-AffinityRecord $record
} finally {
    # Restore only our own original affinity, even if the probe/task failed.
    $currentProcess.ProcessorAffinity = $originalAffinity
    $currentProcess.Dispose()
}
exit $taskExitCode
