.PHONY: pe pe-log pe-compile pe-compile-log pe-gui pe-tb pe-clean clean
.SILENT:

pe: pe-clean
	mkdir -p logs
	catapult -shell -file scripts/run_pe_catapult.tcl 2>&1 | tee logs/catapult_pe.log

pe-log: pe

pe-compile: pe-clean
	mkdir -p logs
	PE_COMPILE_ONLY=1 catapult -shell -file scripts/run_pe_catapult.tcl 2>&1 | tee logs/catapult_pe_compile.log

pe-compile-log: pe-compile

pe-gui:
	PRJ=$$(ls -t PE.ccs PE*.ccs 2>/dev/null | head -n 1); \
	if [ -z "$$PRJ" ]; then \
		echo "No PE.ccs found. Running batch flow once to create project..."; \
		catapult -shell -file scripts/run_pe_catapult.tcl; \
		PRJ=$$(ls -t PE.ccs PE*.ccs 2>/dev/null | head -n 1); \
	fi; \
	if [ -n "$$PRJ" ]; then \
		catapult "$$PRJ" & \
	else \
		echo "Could not find PE project file."; exit 1; \
	fi

pe-tb: pe-clean
	mkdir -p logs
	catapult -shell -file scripts/run_pe_catapult.tcl 2>&1 | tee logs/catapult_pe.log
	SOL_DIR=$$(ls -d PE/PEParallelBlock.v* PE/solution.v* 2>/dev/null | sort -V | tail -n 1); \
	if [ -z "$$SOL_DIR" ]; then echo "No PE solution directory found."; exit 1; fi; \
	SCV_MK=$$(find "$$SOL_DIR" -type f -path "*/scverify/Makefile" 2>/dev/null | sort -V | tail -n 1); \
	if [ -z "$$SCV_MK" ]; then \
		SCV_MK=$$(find "$$SOL_DIR" -type f -path "*/scverify/Verify_*.mk" 2>/dev/null | sort -V | tail -n 1); \
	fi; \
	if [ -z "$$SCV_MK" ]; then echo "SCVerify makefile not found under $$SOL_DIR"; exit 1; fi; \
	SCV_DIR=$$(dirname "$$SCV_MK"); \
	SCV_BASENAME=$$(basename "$$SCV_MK"); \
	$(MAKE) -C "$$SCV_DIR" -f "$$SCV_BASENAME" sim 2>&1 | tee logs/scverify_pe_sim.log

pe-clean:
	rm -rf PE PE.ccs logs catapult.log catapult_pid* .Catapult*

clean: pe-clean
