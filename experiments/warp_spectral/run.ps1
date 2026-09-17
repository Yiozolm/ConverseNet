param(
    [string]$PythonEnv = 'F:\anaconda3\envs\vllm',
    [string]$CudaRoot = 'F:\NVIDIA\CUDA\v13.2',
    [string]$Toolchain = 'F:\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.44.35207',
    [string]$WindowsSdk = 'F:\Windows Kits\10',
    [string]$SdkVersion = '10.0.28000.0',
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$TaskArgs
)
$ErrorActionPreference = 'Stop'
$env:CUDA_HOME = $CudaRoot
$env:CUDA_PATH = $CudaRoot
$env:INCLUDE = "$Toolchain\include;$WindowsSdk\Include\$SdkVersion\ucrt;$WindowsSdk\Include\$SdkVersion\shared;$WindowsSdk\Include\$SdkVersion\um;$WindowsSdk\Include\$SdkVersion\winrt"
$env:LIB = "$Toolchain\lib\x64;$WindowsSdk\Lib\$SdkVersion\ucrt\x64;$WindowsSdk\Lib\$SdkVersion\um\x64"
$env:PATH = "$Toolchain\bin\Hostx64\x64;$WindowsSdk\bin\$SdkVersion\x64;$CudaRoot\bin;$PythonEnv;$PythonEnv\Scripts;$PythonEnv\Library\bin;$env:PATH"
$env:VSLANG = '1033'
$env:CL = '/Zc:preprocessor /DWIN32_LEAN_AND_MEAN /DNOMINMAX'
$env:NVCC_PREPEND_FLAGS = '-Xcompiler /Zc:preprocessor -DWIN32_LEAN_AND_MEAN -DNOMINMAX'
$env:TORCH_CUDA_ARCH_LIST = '12.0'
$env:MAX_JOBS = '2'
$env:DISTUTILS_USE_SDK = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
& "$PythonEnv\python.exe" -u @TaskArgs
exit $LASTEXITCODE
