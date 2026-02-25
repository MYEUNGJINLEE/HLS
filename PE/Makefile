SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -Command

.PHONY: pe pe-log pe-gui pe-tb pe-clean clean
.SILENT:

pe: pe-clean
	New-Item -ItemType Directory -Force -Path logs | Out-Null
	catapult -shell -file scripts/run_pe_catapult.tcl 2>&1 | Tee-Object -FilePath logs/catapult_pe.log

pe-log: pe

pe-gui:
	$prj = Get-ChildItem -Path . -File -Filter *.ccs -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
	if (-not $prj) {
		Write-Output "No PE.ccs found. Running batch flow once to create project..."
		catapult -shell -file scripts/run_pe_catapult.tcl
		$prj = Get-ChildItem -Path . -File -Filter *.ccs -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
	}
	if ($prj) {
		Start-Process catapult -ArgumentList $prj.FullName
	} else {
		Write-Error "Could not find PE project file."
		exit 1
	}

pe-tb: pe-clean
	New-Item -ItemType Directory -Force -Path logs | Out-Null
	catapult -shell -file scripts/run_pe_catapult.tcl 2>&1 | Tee-Object -FilePath logs/catapult_pe.log
	$solDir = Get-ChildItem -Path . -Directory -Filter PE -ErrorAction SilentlyContinue | Select-Object -First 1
	if (-not $solDir) { Write-Error "No PE project directory found."; exit 1 }
	$solPath = Get-ChildItem -Path $solDir.FullName -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -like 'PEParallelBlock.v*' -or $_.Name -like 'solution.v*' } | Sort-Object Name | Select-Object -Last 1
	if (-not $solPath) { Write-Error "No PE solution directory found."; exit 1 }
	$scvMk = Get-ChildItem -Path $solPath.FullName -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.FullName -like '*scverify*Makefile' -or $_.Name -like 'Verify_*.mk' } | Sort-Object FullName | Select-Object -Last 1
	if (-not $scvMk) { Write-Error "SCVerify makefile not found under $($solPath.FullName)"; exit 1 }
	make -C $scvMk.DirectoryName -f $scvMk.Name sim 2>&1 | Tee-Object -FilePath logs/scverify_pe_sim.log

pe-clean:
	if (Test-Path logs) { Remove-Item -Recurse -Force logs -ErrorAction SilentlyContinue }
	if (Test-Path PE) { Remove-Item -Recurse -Force PE -ErrorAction SilentlyContinue }
	if (Test-Path PE.ccs) { Remove-Item -Force PE.ccs -ErrorAction SilentlyContinue }
	if (Test-Path catapult.log) { Remove-Item -Force catapult.log -ErrorAction SilentlyContinue }
	Get-ChildItem -Path . -Filter catapult_pid* -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
	Get-ChildItem -Path . -Filter .Catapult* -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

clean: pe-clean
