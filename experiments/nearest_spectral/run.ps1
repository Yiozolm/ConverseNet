param(
    [string]$Python = 'F:\anaconda3\envs\vllm\python.exe',
    [string]$Msvc = 'F:\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.44.35207',
    [string]$Sdk = 'F:\Windows Kits\10',
    [string]$SdkVersion = '10.0.28000.0',
    [string]$Cuda = 'F:\NVIDIA\CUDA\v13.2',
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$StudyArgs
)
$ErrorActionPreference = 'Stop'
$env:PATH = "$Msvc\bin\Hostx64\x64;$Sdk\bin\$SdkVersion\x64;$Cuda\bin;$(Split-Path -Parent $Python);$(Split-Path -Parent $Python)\Scripts;$env:PATH"
$env:INCLUDE = "$Msvc\include;$Sdk\Include\$SdkVersion\ucrt;$Sdk\Include\$SdkVersion\shared;$Sdk\Include\$SdkVersion\um;$Sdk\Include\$SdkVersion\winrt"
$env:LIB = "$Msvc\lib\x64;$Sdk\Lib\$SdkVersion\ucrt\x64;$Sdk\Lib\$SdkVersion\um\x64"
$env:CUDA_HOME = $Cuda
$env:CUDA_PATH = $Cuda
$env:VSLANG = '1033'
$env:DISTUTILS_USE_SDK = '1'
$env:MAX_JOBS = '2'
$env:PYTHONDONTWRITEBYTECODE = '1'
& $Python (Join-Path $PSScriptRoot 'study.py') @StudyArgs
exit $LASTEXITCODE
