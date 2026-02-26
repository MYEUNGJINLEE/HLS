.PHONY: clean compile gui tb
.SILENT:

compile: clean
	mkdir -p logs
	PE_COMPILE_ONLY=1 catapult -shell -file scripts/run_pe_catapult.tcl 2>&1 | tee logs/catapult_pe_compile.log

gui:
	PRJ=$$(ls -t pe.ccs PE.ccs pe*.ccs PE*.ccs 2>/dev/null | head -n 1); \
	if [ -z "$$PRJ" ]; then \
		echo "No .ccs found. Running compile first..."; \
		rm -rf PE PE.ccs pe.ccs logs catapult.log catapult_pid* .Catapult*; \
		mkdir -p logs; \
		PE_COMPILE_ONLY=1 catapult -shell -file scripts/run_pe_catapult.tcl 2>&1 | tee logs/catapult_pe_compile.log; \
		PRJ=$$(ls -t pe.ccs PE.ccs pe*.ccs PE*.ccs 2>/dev/null | head -n 1); \
	fi; \
	if [ -n "$$PRJ" ]; then \
		if [ "$$PRJ" != "pe.ccs" ]; then cp "$$PRJ" pe.ccs; fi; \
		catapult -f pe.ccs & \
	else \
		echo "Could not find PE project file."; exit 1; \
	fi

tb: compile
	SOL_DIR=$$(ls -d PE/PEParallelBlock.v* PE/solution.v* 2>/dev/null | sort -V | tail -n 1); \
	if [ -z "$$SOL_DIR" ]; then echo "No PE solution directory found."; exit 1; fi; \
	SCV_MK=$$(find "$$SOL_DIR" -type f -path "*/scverify/Verify_orig_cxx_osci.mk" 2>/dev/null | sort -V | tail -n 1); \
	if [ -z "$$SCV_MK" ]; then \
		SCV_MK=$$(find "$$SOL_DIR" -type f -path "*/scverify/verify_orig_cxx_osci.mk" 2>/dev/null | sort -V | tail -n 1); \
	fi; \
	if [ -z "$$SCV_MK" ]; then \
		SCV_MK=$$(find "$$SOL_DIR" -type f -path "*/scverify/Makefile" 2>/dev/null | sort -V | tail -n 1); \
	fi; \
	if [ -z "$$SCV_MK" ]; then \
		SCV_MK=$$(find "$$SOL_DIR" -type f -path "*/scverify/Verify_*.mk" 2>/dev/null | sort -V | tail -n 1); \
	fi; \
	if [ -z "$$SCV_MK" ]; then echo "SCVerify makefile not found under $$SOL_DIR"; exit 1; fi; \
	mkdir -p logs; \
	ROOT_DIR=$$(pwd); \
	PROJ_DIR=$$(dirname "$$SOL_DIR"); \
	PROJ_DIR_ABS="$$ROOT_DIR/$$PROJ_DIR"; \
	SCV_MK_ABS="$$ROOT_DIR/$$SCV_MK"; \
	LAUNCH_TCL="$$ROOT_DIR/logs/pe_scverify_launch.tcl"; \
	printf "if {![file isdirectory {%s}]} { error {missing PE project directory} }\nproject load {%s} 2025.2\nflow package require /SCVerify\nflow run /SCVerify/launch_make %s {} SIMTOOL=osci build\nflow run /SCVerify/launch_make %s {} SIMTOOL=osci sim\nexit\n" "$$PROJ_DIR_ABS" "$$PROJ_DIR_ABS" "$$SCV_MK_ABS" "$$SCV_MK_ABS" > "$$LAUNCH_TCL"; \
	catapult -shell -file "$$LAUNCH_TCL" > logs/scverify_pe.log 2>&1; \
	SCV_RC=$$?; \
	cat logs/scverify_pe.log; \
	rm -f "$$LAUNCH_TCL"; \
	if [ "$$SCV_RC" -ne 0 ]; then exit "$$SCV_RC"; fi

clean:
	rm -rf PE PE.ccs pe.ccs logs catapult.log catapult_pid* .Catapult*
