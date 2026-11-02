# Windows counterpart of setup.sh: installs Ruby + Jekyll and serves this site locally.
#
# Run from PowerShell (no admin needed):
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1

$ErrorActionPreference = "Stop"

# PowerShell 5.1 does not stop on a failing native command, so check exit codes explicitly.
function Invoke-Native {
  param([string]$Exe, [string[]]$Arguments)
  & $Exe @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "'$Exe $($Arguments -join ' ')' failed with exit code $LASTEXITCODE"
  }
}

# Pick up PATH changes made by installers without restarting the terminal.
function Update-SessionPath {
  $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
  $user = [Environment]::GetEnvironmentVariable("Path", "User")
  $env:Path = "$machine;$user"
}

function Test-Command {
  param([string]$Name)
  return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

Write-Host "=== Step 1: Checking for winget (the package manager)... ==="
if (-not (Test-Command "winget")) {
  Write-Host "winget not found. Install 'App Installer' from the Microsoft Store, then run this script again."
  Write-Host "  https://apps.microsoft.com/detail/9NBLGGH4NNS1"
  exit 1
}

Write-Host ""
Write-Host "=== Step 2: Installing Ruby (RubyInstaller with DevKit) if needed... ==="
if (-not (Test-Command "ruby")) {
  Invoke-Native "winget" @("install", "--id", "RubyInstallerTeam.RubyWithDevKit.3.3", "-e",
                           "--accept-package-agreements", "--accept-source-agreements")
  Update-SessionPath

  # Fall back to the installer's default location if PATH was not updated.
  if (-not (Test-Command "ruby") -and (Test-Path "C:\Ruby33-x64\bin")) {
    $env:Path = "C:\Ruby33-x64\bin;$env:Path"
  }
  if (-not (Test-Command "ruby")) {
    Write-Host "Ruby was installed but is not on PATH yet. Open a new terminal and run this script again."
    exit 1
  }
} else {
  Write-Host "Ruby already installed."
}

Write-Host ""
Write-Host "Ruby version:"
Invoke-Native "ruby" @("-v")

Write-Host ""
Write-Host "=== Step 3: Installing the MSYS2 build toolchain (needed to compile some gems)... ==="
# A silent winget install skips RubyInstaller's 'ridk install' step, so run it here.
if (-not (Test-Command "ridk")) {
  Write-Host "ridk not found (Ruby was not installed with RubyInstaller). Skipping; gem builds may fail."
} else {
  # Redirecting a native command's stderr throws under 'Stop' in PowerShell 5.1.
  $ErrorActionPreference = "Continue"
  & ridk exec gcc --version *> $null
  $hasToolchain = ($LASTEXITCODE -eq 0)
  $ErrorActionPreference = "Stop"

  if (-not $hasToolchain) {
    Invoke-Native "ridk" @("install", "3")
  } else {
    Write-Host "Toolchain already installed."
  }
}

Write-Host ""
Write-Host "=== Step 4: Installing Jekyll and Bundler gems if needed... ==="
if (-not (Test-Command "jekyll") -or -not (Test-Command "bundle")) {
  Invoke-Native "gem" @("install", "bundler", "jekyll", "--no-document")
  Update-SessionPath
} else {
  Write-Host "Jekyll and Bundler already installed."
}

# The Jekyll site is the folder this script lives in (docs/).
$JekyllDir = $PSScriptRoot
if (-not (Test-Path (Join-Path $JekyllDir "_config.yml"))) {
  Write-Host "WARNING: Expected a Jekyll site (_config.yml) at:"
  Write-Host "  $JekyllDir"
  Write-Host "Keep this script in the docs/ folder of the FastFlowLM repo."
  exit 1
}

Write-Host ""
Write-Host "=== Step 5: Installing site dependencies and starting Jekyll server... ==="
Write-Host "When the server starts, open this in your browser:"
Write-Host "  http://127.0.0.1:4000"
Write-Host "Press Ctrl+C to stop it."
Write-Host ""
Set-Location $JekyllDir

if (Test-Path "Gemfile") {
  Write-Host "Gemfile found. Running bundle install..."
  Invoke-Native "bundle" @("install")
  Write-Host ""
  Write-Host "Starting Jekyll server with Bundler..."
  Invoke-Native "bundle" @("exec", "jekyll", "serve")
} else {
  Write-Host "No Gemfile found. Running jekyll serve directly..."
  # --incremental never notices _config.yml edits (e.g. the sidebar nav), so drop its
  # cache and do one full build per start; edits while running stay incremental.
  Remove-Item ".jekyll-metadata" -ErrorAction SilentlyContinue
  Invoke-Native "jekyll" @("serve", "--watch", "--incremental", "--livereload")
}
