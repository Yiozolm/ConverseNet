param([Parameter(ValueFromRemainingArguments=$true)][string[]]$TaskArgs)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$pythonPath = Join-Path $repoRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    $pythonPath = (Get-Command python -ErrorAction Stop).Source
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) { throw 'vswhere not found; install the Visual Studio C++ build tools' }
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) { throw 'MSVC installation not found' }
$devcmd = Join-Path $installation 'Common7/Tools/VsDevCmd.bat'
$versionArgument = ''
if ($env:CONVERSE_MSVC_VERSION) {
    if ($env:CONVERSE_MSVC_VERSION -notmatch '^\d+(\.\d+)*$') { throw 'CONVERSE_MSVC_VERSION must be a numeric toolset version such as 14.44' }
    $versionArgument = ' -vcvars_ver=' + $env:CONVERSE_MSVC_VERSION
}
$devvars = & $env:ComSpec /d /s /c ('"' + $devcmd + '" -arch=x64 -host_arch=x64' + $versionArgument + ' >nul && set')
if ($LASTEXITCODE -ne 0) { throw 'MSVC initialization failed' }
foreach ($line in $devvars) {
    if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
if (-not $env:CUDA_HOME -or -not (Test-Path -LiteralPath (Join-Path $env:CUDA_HOME 'bin/nvcc.exe'))) {
    if ($env:CUDA_PATH -and (Test-Path -LiteralPath (Join-Path $env:CUDA_PATH 'bin/nvcc.exe'))) {
        $env:CUDA_HOME = $env:CUDA_PATH
    } else {
        $toolkitRoot = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit/CUDA'
        $toolkit = Get-ChildItem -LiteralPath $toolkitRoot -Directory |
            Where-Object { $_.Name -match '^v\d+\.\d+$' } |
            Sort-Object { [version]$_.Name.Substring(1) } -Descending | Select-Object -First 1
        if (-not $toolkit) { throw 'CUDA toolkit not found; set CUDA_HOME explicitly' }
        $env:CUDA_HOME = $toolkit.FullName
    }
}
$env:CUDA_PATH = $env:CUDA_HOME
$env:PATH = "$env:CUDA_HOME\bin;$(Split-Path -Parent $pythonPath);$env:PATH"
$env:DISTUTILS_USE_SDK = '1'
$env:VSLANG = '1033'
if (-not $env:MAX_JOBS) { $env:MAX_JOBS = '2' }
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONNOUSERSITE = '1'
$env:TEMP = Join-Path $repoRoot '.build/training_full_spectrum_fusion/temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
& $pythonPath -X utf8 (Join-Path $PSScriptRoot 'loader.py') @TaskArgs
exit $LASTEXITCODE
