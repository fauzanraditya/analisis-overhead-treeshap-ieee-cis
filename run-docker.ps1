param([string]$DataPath, [ValidateSet('pilot','formal')][string]$RunKind='pilot', [string]$Image='fraud-benchmark:ieee-cis', [switch]$SkipBuild)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
if (-not $DataPath) {
    $DataPath = Join-Path $taskRoot '..\Dataset + Model\Dataset IEEE-CIS\Uji Bersama Dua Model'
}
$resolvedData = (Resolve-Path -LiteralPath $DataPath).Path
$resultsRoot = Join-Path $taskRoot '..\Hasil Benchmark'
New-Item -ItemType Directory -Path $resultsRoot -Force | Out-Null
$resolvedResults = (Resolve-Path -LiteralPath $resultsRoot).Path
$runName = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0,8) + '_ieee_' + $RunKind
$evidenceDir = Join-Path $resultsRoot ($runName + '_docker_evidence')
New-Item -ItemType Directory -Path $evidenceDir | Out-Null
if (-not $SkipBuild) {
    & docker build -t $Image $taskRoot
    if ($LASTEXITCODE -ne 0) { throw 'Docker build failed' }
}
$containerName = 'fraud-ieee-' + [guid]::NewGuid().ToString('N').Substring(0,12)
$dockerArguments = @('create', '--name', $containerName, '--cpus=2', '--memory=4g', '--memory-swap=4g',
    '--mount', "type=bind,source=$resolvedData,target=/data,readonly",
    '--mount', "type=bind,source=$resolvedResults,target=/out", $Image,
    'python', 'scripts/benchmark.py', '--data', '/data', '--run-kind', $RunKind, '--output-dir', "/out/$runName")
if ($RunKind -eq 'pilot') { $dockerArguments += @('--rows','1000','--replications','2','--warmups','1') }
@{ container=$containerName; image=$Image; arguments=$dockerArguments; hostOS=[Environment]::OSVersion.VersionString;
   processor=(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors);
   physicalMemoryBytes=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory } |
    ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $evidenceDir 'launch.json') -Encoding UTF8
$containerId = & docker @dockerArguments
if ($LASTEXITCODE -ne 0) { throw 'Docker container creation failed' }
& docker inspect $containerName | Set-Content -LiteralPath (Join-Path $evidenceDir 'inspect_before.json') -Encoding UTF8
if ($LASTEXITCODE -ne 0) { throw 'Docker inspection failed' }
& docker image inspect $Image | Set-Content -LiteralPath (Join-Path $evidenceDir 'image_inspect.json') -Encoding UTF8
& docker start --attach $containerName
$attachExit = $LASTEXITCODE
$inspection = & docker inspect $containerName
$inspection | Set-Content -LiteralPath (Join-Path $evidenceDir 'inspect_after.json') -Encoding UTF8
$exitCode = ($inspection | ConvertFrom-Json)[0].State.ExitCode
if ($attachExit -ne 0 -or $exitCode -ne 0) { throw "Benchmark failed; container $containerName retained for diagnosis" }
$runDir = Join-Path $resultsRoot $runName
Copy-Item -LiteralPath (Join-Path $evidenceDir 'launch.json'),(Join-Path $evidenceDir 'inspect_before.json'),(Join-Path $evidenceDir 'inspect_after.json'),(Join-Path $evidenceDir 'image_inspect.json') -Destination $runDir
& docker run --rm --cpus=2 --memory=4g --memory-swap=4g --mount "type=bind,source=$resolvedResults,target=/out" $Image python scripts/analyze_results.py --run-dir "/out/$runName"
if ($LASTEXITCODE -ne 0) { throw 'Analysis failed; benchmark CSVs retained' }
Write-Host "Completed run: $runDir"
Write-Host "Readable report: $(Join-Path $runDir 'analysis\Laporan Terbaca\Laporan Benchmark.html')"
Write-Host "Excel CSV: $(Join-Path $runDir 'analysis\Laporan Terbaca\Ringkasan Benchmark.csv')"
Write-Host "Stopped benchmark container retained: $containerName"
