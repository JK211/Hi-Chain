# Sweep UAV swarm sizes (run from NS-3 root on Windows).
# Usage:
#   .\run_swarm_scale_sweep.ps1 -Ns3Root D:\ns-3.25

param(
    [Parameter(Mandatory = $true)]
    [string]$Ns3Root,
    [int[]]$Scales = @(10, 20, 30, 40, 50),
    [double]$SimTime = 100,
    [string]$CsvOut = "hi-chain-swarm-metrics.csv"
)

Set-Location $Ns3Root
if (Test-Path $CsvOut) { Remove-Item $CsvOut }

foreach ($n in $Scales) {
    Write-Host "=== nUavs=$n ==="
    & python waf --run "scratch/hi-chain-uav-swarm-validation --nUavs=$n --simTime=$SimTime --csvOut=$CsvOut"
}

Write-Host "Done. Results: $(Join-Path $Ns3Root $CsvOut)"
