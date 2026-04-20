$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$runtimeDir = Join-Path $projectRoot ".jupyter_runtime"
$configDir = Join-Path $projectRoot ".jupyter_config"
$dataDir = Join-Path $projectRoot ".jupyter_data"
$ipythonDir = Join-Path $projectRoot ".ipython"
$mplConfigDir = Join-Path $projectRoot ".mplconfig"
$notebookPath = Join-Path $projectRoot "docs\image_dehazing_aodnet_residual_skip_notebook.ipynb"

New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
New-Item -ItemType Directory -Force -Path $configDir | Out-Null
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
New-Item -ItemType Directory -Force -Path $ipythonDir | Out-Null
New-Item -ItemType Directory -Force -Path $mplConfigDir | Out-Null

$env:JUPYTER_RUNTIME_DIR = $runtimeDir
$env:JUPYTER_CONFIG_DIR = $configDir
$env:JUPYTER_DATA_DIR = $dataDir
$env:IPYTHONDIR = $ipythonDir
$env:MPLCONFIGDIR = $mplConfigDir

& $venvPython -m notebook $notebookPath
