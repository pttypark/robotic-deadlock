param(
    [string]$PythonVersion = "3.12"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

function Get-BootstrapPython {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @("py", "-$PythonVersion")
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @("python")
    }

    throw "Could not find a Python launcher. Install Python $PythonVersion or newer first."
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$CommandParts
    )

    $CommandText = ($CommandParts -join " ")
    Write-Host ">> $CommandText" -ForegroundColor Cyan
    & $CommandParts[0] $CommandParts[1..($CommandParts.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $CommandText"
    }
}

if (-not (Test-Path $VenvPython)) {
    $BootstrapPython = Get-BootstrapPython
    Invoke-Checked ($BootstrapPython + @("-m", "venv", $VenvDir))
}

Invoke-Checked @($VenvPython, "-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked @($VenvPython, "-m", "pip", "install", "-r", (Join-Path $ProjectRoot "requirements.txt"))
Write-Host ">> Registering project path for local imports" -ForegroundColor Cyan
$SitePackages = Join-Path $VenvDir "Lib\site-packages"
$PthFile = Join-Path $SitePackages "robotic_deadlock_local.pth"
Set-Content -Path $PthFile -Value $ProjectRoot -Encoding ASCII

Write-Host ""
Write-Host "Environment is ready." -ForegroundColor Green
Write-Host "Activate it with:" -ForegroundColor Green
Write-Host ".\.venv\Scripts\Activate.ps1"
