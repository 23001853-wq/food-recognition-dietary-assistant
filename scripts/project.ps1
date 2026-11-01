#Requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateSet('Status', 'DownloadData', 'PrepareData', 'VerifyData', 'TrainHog',
                 'TrainResNet', 'TrainEfficientNet', 'Evaluate', 'Select', 'Tests',
                 'MockSmoke', 'RealSmoke', 'Backend', 'Frontend', 'ComposeUp')]
    [string]$Step = 'Status',
    [string]$RunId = 'poc-v1-cpu-budget-v2-primal',
    [string]$ImagePath
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Project .venv is missing. Follow RUN_COMMANDS.md environment setup first.'
}

function Invoke-ProjectPython {
    param([string[]]$CommandArguments)
    & $projectPython @CommandArguments
    if ($LASTEXITCODE -ne 0) { throw "Python step failed with exit code $LASTEXITCODE" }
}

function Get-ExperimentArtifacts {
    $modelPaths = @()
    foreach ($model in @('hog_svm', 'resnet18', 'efficientnet_b0')) {
        $modelPath = "artifacts/models/$model/$RunId"
        $metadataPath = Join-Path $modelPath 'model_config.json'
        if (-not (Test-Path -LiteralPath $metadataPath)) { throw "Missing model: $modelPath" }
        $metadata = Get-Content -LiteralPath $metadataPath -Encoding UTF8 -Raw | ConvertFrom-Json
        if (-not $metadata.trained -or $metadata.smoke_only -or
            -not (Test-Path -LiteralPath (Join-Path $modelPath 'checksum.sha256'))) {
            throw "Model is incomplete/smoke-only: $modelPath. Finish all candidates before test evaluation."
        }
        $modelPaths += $modelPath
    }
    return $modelPaths
}

switch ($Step) {
    'Status' { Invoke-ProjectPython -CommandArguments @('scripts/status_project.py', '--docker') }
    'DownloadData' { Invoke-ProjectPython -CommandArguments @('scripts/download_food101_parallel.py', '--workers', '8') }
    'PrepareData' {
        if (Test-Path -LiteralPath 'data/splits/v1/dataset_manifest.csv') {
            Write-Host 'Frozen v1 manifest already exists; not overwritten.'
        } else { Invoke-ProjectPython -CommandArguments @('scripts/create_splits.py') }
    }
    'VerifyData' { Invoke-ProjectPython -CommandArguments @('scripts/verify_dataset.py') }
    'TrainHog' {
        Invoke-ProjectPython -CommandArguments @('-u', 'scripts/train_hog_svm.py', '--run-id', $RunId)
    }
    'TrainResNet' {
        Invoke-ProjectPython -CommandArguments @('-u', 'scripts/train_resnet18.py', '--run-id', $RunId,
            '--device', 'cpu', '--head-epochs', '1', '--fine-tune-epochs', '1', '--batch-size', '4')
    }
    'TrainEfficientNet' {
        Invoke-ProjectPython -CommandArguments @('-u', 'scripts/train_efficientnet_b0.py', '--run-id', $RunId,
            '--device', 'cpu', '--head-epochs', '1', '--fine-tune-epochs', '1', '--batch-size', '4')
    }
    'Evaluate' {
        $modelPaths = Get-ExperimentArtifacts
        $protocolPath = "artifacts/reports/$RunId/frozen_protocol.json"
        if (-not (Test-Path -LiteralPath $protocolPath)) {
            Invoke-ProjectPython -CommandArguments (@('scripts/freeze_poc.py','--output',$protocolPath,'--artifacts') + $modelPaths)
        }
        foreach ($modelPath in $modelPaths) {
            if ((Test-Path -LiteralPath (Join-Path $modelPath 'evaluation_test/evaluation_complete.json')) -and
                (Test-Path -LiteralPath (Join-Path $modelPath 'metrics_test.json'))) {
                Write-Host "Existing test evaluation preserved: $modelPath"
                continue
            }
            Invoke-ProjectPython -CommandArguments @('-u', 'scripts/evaluate_models.py',
                '--artifact', $modelPath, '--split', 'test', '--device', 'cpu','--protocol',$protocolPath)
        }
    }
    'Select' {
        $modelPaths = Get-ExperimentArtifacts
        Invoke-ProjectPython -CommandArguments (@('scripts/select_production_model.py', '--artifacts') + $modelPaths)
    }
    'Tests' { Invoke-ProjectPython -CommandArguments @('-m', 'pytest', '-q') }
    'MockSmoke' { Invoke-ProjectPython -CommandArguments @('scripts/smoke_test.py', '--model', 'mock') }
    'RealSmoke' {
        if (-not $ImagePath) { throw 'RealSmoke requires -ImagePath with a real food validation photo.' }
        Invoke-ProjectPython -CommandArguments @('scripts/smoke_test.py', '--model', 'production','--image',$ImagePath)
    }
    'Backend' {
        Invoke-ProjectPython -CommandArguments @('-m', 'uvicorn', 'food_assistant.api.main:app',
                                                '--host', '127.0.0.1', '--port', '8000')
    }
    'Frontend' {
        Invoke-ProjectPython -CommandArguments @('-m', 'streamlit', 'run', 'src/food_assistant/app.py')
    }
    'ComposeUp' {
        & docker compose up -d --build
        if ($LASTEXITCODE -ne 0) { throw 'Docker Compose failed' }
    }
}
