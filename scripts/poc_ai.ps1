#Requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateSet('Status','Audit','CUDA','TrainHog','TrainResNet','TrainEfficientNet','Freeze','Evaluate','Select')]
    [string]$Step = 'Status',
    [string]$HogRunId = 'poc-v1-cpu-budget-v2-primal',
    [string]$CnnRunId = 'poc10-rtx3050-v1'
)
$ErrorActionPreference = 'Stop'
$pocRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $pocRoot
$pocCpuPython = Join-Path $pocRoot '.venv/Scripts/python.exe'
$pocCudaPython = Join-Path $pocRoot '.venv-cuda/Scripts/python.exe'
$pocArtifacts = @("artifacts/models/hog_svm/$HogRunId",
                  "artifacts/models/resnet18/$CnnRunId",
                  "artifacts/models/efficientnet_b0/$CnnRunId")
$pocProtocol = "artifacts/reports/$CnnRunId/frozen_protocol.json"
function Invoke-Poc {
    param([string]$Python, [string[]]$Arguments)
    if (-not (Test-Path -LiteralPath $Python)) { throw 'Missing venv; follow POC_REAL_AI.md first.' }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "POC step failed: exit $LASTEXITCODE" }
}
switch ($Step) {
    'Status' { Invoke-Poc $pocCpuPython @('scripts/status_project.py') }
    'Audit' { Invoke-Poc $pocCpuPython @('scripts/audit_poc.py', '--output', 'data/splits/v1/poc_contract_audit.json') }
    'CUDA' { Invoke-Poc $pocCudaPython @('scripts/check_training_device.py', '--require-cuda', '--probe') }
    'TrainHog' {
        Invoke-Poc $pocCpuPython @('-u','scripts/train_hog_svm.py','--config','configs/poc/hog_svm_64.yaml','--run-id',$HogRunId)
    }
    'TrainResNet' {
        Invoke-Poc $pocCudaPython @('scripts/check_training_device.py', '--require-cuda')
        Invoke-Poc $pocCudaPython @('-u','scripts/train_resnet18.py','--config','configs/poc/resnet18_rtx3050.yaml','--run-id',$CnnRunId)
    }
    'TrainEfficientNet' {
        Invoke-Poc $pocCudaPython @('scripts/check_training_device.py', '--require-cuda')
        Invoke-Poc $pocCudaPython @('-u','scripts/train_efficientnet_b0.py','--config','configs/poc/efficientnet_b0_rtx3050.yaml','--run-id',$CnnRunId)
    }
    'Freeze' { Invoke-Poc $pocCudaPython (@('scripts/freeze_poc.py','--output',$pocProtocol,'--artifacts') + $pocArtifacts) }
    'Evaluate' {
        if (-not (Test-Path -LiteralPath $pocProtocol)) { throw 'Freeze the three candidates before Evaluate.' }
        foreach ($artifact in $pocArtifacts) {
            if ((Test-Path -LiteralPath "$artifact/metrics_test.json") -and
                (Test-Path -LiteralPath "$artifact/evaluation_test/evaluation_complete.json")) {
                Write-Host "Completed test evaluation preserved: $artifact"
                continue
            }
            Invoke-Poc $pocCudaPython @('-u','scripts/evaluate_models.py','--artifact',$artifact,'--split','test','--device','cpu','--protocol',$pocProtocol)
        }
    }
    'Select' { Invoke-Poc $pocCudaPython (@('scripts/select_production_model.py','--artifacts') + $pocArtifacts) }
}
