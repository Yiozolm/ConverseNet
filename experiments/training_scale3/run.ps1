param([Parameter(ValueFromRemainingArguments=$true)][string[]]$TaskArgs)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) { throw 'MSVC installation not found' }
$devcmd = Join-Path $installation 'Common7\Tools\VsDevCmd.bat'
$devvars = & $env:ComSpec /d /s /c ('"' + $devcmd + '" -arch=x64 -host_arch=x64 -vcvars_ver=14.44 >nul && set')
if ($LASTEXITCODE -ne 0) { throw 'MSVC initialization failed' }
foreach ($line in $devvars) {
    if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
}
if (-not $env:CUDA_HOME -or -not (Test-Path (Join-Path $env:CUDA_HOME 'bin/nvcc.exe'))) {
    $toolkit = Get-ChildItem 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA' -Directory |
        Where-Object { $_.Name -match '^v\d+\.\d+$' } | Sort-Object { [version]$_.Name.Substring(1) } -Descending | Select-Object -First 1
    if (-not $toolkit) { throw 'CUDA toolkit not found; set CUDA_HOME' }
    $env:CUDA_HOME = $toolkit.FullName
}
$env:CUDA_PATH = $env:CUDA_HOME
$env:PATH = "$env:CUDA_HOME\bin;$repoRoot\.venv\Scripts;$env:PATH"
$env:CONVERSE2D_BUILD_PATH = $env:PATH
$env:DISTUTILS_USE_SDK = '1'
$env:VSLANG = '1033'
$env:MAX_JOBS = '2'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONNOUSERSITE = '1'
$env:TEMP = Join-Path $repoRoot '.build/training_scale3/temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
& (Join-Path $repoRoot '.venv/Scripts/python.exe') -X utf8 @TaskArgs
exit $LASTEXITCODE
