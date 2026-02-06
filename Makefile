.PHONY: update push push-compile-results weight-dir stream stream-log stream-tb stream-gui stream-clean block block-log block-gui block-tb block-clean stem stem-log stem-gui stem-tb stem-tb-weight stem-tb-no-weight stem-clean clean

# Update local repository to latest origin/dev
update:
	git fetch origin
	git checkout dev
	git pull origin dev

# Create local weight directory for stem verification files
weight-dir:
	mkdir -p Binary_cnn/weights

# Force-add compile logs, commit, and push to origin/dev
# Usage:
#   make push-compile-results
push: push-compile-results

push-compile-results:
	@LOGS=$$(ls -1 Binary_cnn/logs/*.log Binary_cnn/catapult.log 2>/dev/null); \
	if [ -z "$$LOGS" ]; then \
		echo "No compile logs found (Binary_cnn/logs/*.log, Binary_cnn/catapult.log)."; \
		exit 1; \
	fi; \
	git add -f $$LOGS; \
	if git diff --cached --quiet; then \
		echo "No log changes to commit."; \
		exit 0; \
	fi; \
	git commit -m "logs: catpult log upload"; \
	git push origin dev

# Run Catapult in batch mode (no GUI)
stream:
	cd Binary_cnn && catapult -shell -file scripts/run_streaming_catapult.tcl

# Run Catapult and save full console log
stream-log:
	cd Binary_cnn && mkdir -p logs && catapult -shell -file scripts/run_streaming_catapult.tcl 2>&1 | tee logs/catapult_streaming.log

# Run Catapult batch flow and execute SCVerify testbench simulation
stream-tb:
	cd Binary_cnn && mkdir -p logs && catapult -shell -file scripts/run_streaming_catapult.tcl 2>&1 | tee logs/catapult_streaming.log
	cd Binary_cnn && SOL_DIR=$$(ls -d bin_cnn_streaming/BW_CNN_Streaming.v* BW_CNN_Streaming.v* 2>/dev/null | sort -V | tail -n 1); \
	if [ -z "$$SOL_DIR" ]; then echo "No BW_CNN_Streaming.v* solution directory found."; exit 1; fi; \
	SCV_MK=$$(find "$$SOL_DIR" -type f -path "*/scverify/Makefile" 2>/dev/null | sort -V | tail -n 1); \
	if [ -z "$$SCV_MK" ]; then \
		SCV_MK=$$(find "$$SOL_DIR" -type f -path "*/scverify/Verify_*.mk" 2>/dev/null | sort -V | tail -n 1); \
	fi; \
	if [ -z "$$SCV_MK" ]; then echo "SCVerify makefile not found under $$SOL_DIR"; exit 1; fi; \
	SCV_DIR=$$(dirname "$$SCV_MK"); \
	SCV_BASENAME=$$(basename "$$SCV_MK"); \
	$(MAKE) -C "$$SCV_DIR" -f "$$SCV_BASENAME" sim 2>&1 | tee logs/scverify_sim.log

# Clean streaming projects (including numbered variants _1, _2, ...)
stream-clean:
	cd Binary_cnn && rm -rf bin_cnn_streaming* bin_cnn_streaming.ccs

# Clean block processor projects (including numbered variants _1, _2, ...)
block-clean:
	cd Binary_cnn && rm -rf block_processor* block_processor.ccs

# Clean stem processor projects
stem-clean:
	cd Binary_cnn && rm -rf stem_processor* stem_processor.ccs

# Clean all Catapult-generated artifacts
clean: stream-clean block-clean stem-clean
	cd Binary_cnn && rm -rf bin_cnn bin_cnn.ccs Catapult Catapult.ccs logs catapult_pid* .Catapult*

# Run block processor Catapult in batch mode with log (auto-cleans old project)
block: block-clean
	cd Binary_cnn && mkdir -p logs && catapult -shell -file scripts/run_block_processor_catapult.tcl 2>&1 | tee logs/catapult_block_processor.log

# Alias for block
block-log: block

# Run block processor batch flow and execute SCVerify testbench simulation
block-tb: block-clean
	cd Binary_cnn && mkdir -p logs && catapult -shell -file scripts/run_block_processor_catapult.tcl 2>&1 | tee logs/catapult_block_processor.log
	cd Binary_cnn && SOL_DIR=$$(ls -d block_processor/FusedBlockProcessor.v* 2>/dev/null | sort -V | tail -n 1); \
	if [ -z "$$SOL_DIR" ]; then echo "No FusedBlockProcessor.v* solution directory found."; exit 1; fi; \
	SCV_MK=$$(find "$$SOL_DIR" -type f -path "*/scverify/Makefile" 2>/dev/null | sort -V | tail -n 1); \
	if [ -z "$$SCV_MK" ]; then \
		SCV_MK=$$(find "$$SOL_DIR" -type f -path "*/scverify/Verify_*.mk" 2>/dev/null | sort -V | tail -n 1); \
	fi; \
	if [ -z "$$SCV_MK" ]; then echo "SCVerify Makefile not found under $$SOL_DIR"; exit 1; fi; \
	SCV_DIR=$$(dirname "$$SCV_MK"); \
	SCV_BASENAME=$$(basename "$$SCV_MK"); \
	$(MAKE) -C "$$SCV_DIR" -f "$$SCV_BASENAME" sim 2>&1 | tee logs/scverify_block_processor_sim.log

# Run Catapult with GUI and keep window open
stream-gui:
	cd Binary_cnn && PRJ=$$(ls -t bin_cnn_streaming*.ccs 2>/dev/null | head -n 1); \
	if [ -z "$$PRJ" ]; then \
		echo "No bin_cnn_streaming*.ccs found. Running batch flow once to create project..."; \
		catapult -shell -file scripts/run_streaming_catapult.tcl; \
		PRJ=$$(ls -t bin_cnn_streaming*.ccs 2>/dev/null | head -n 1); \
	fi; \
	if [ -n "$$PRJ" ]; then \
		catapult "$$PRJ" & \
	else \
		PRJ_XML=$$(ls -t bin_cnn_streaming*/SIF/project.xml 2>/dev/null | head -n 1); \
		if [ -z "$$PRJ_XML" ]; then echo "Could not find a streaming project (.ccs or SIF/project.xml)."; exit 1; fi; \
		catapult "$$PRJ_XML" & \
	fi

# Run block processor with GUI and keep window open
block-gui:
	cd Binary_cnn && PRJ=$$(ls -t block_processor*.ccs 2>/dev/null | head -n 1); \
	if [ -z "$$PRJ" ]; then \
		echo "No block_processor*.ccs found. Running batch flow once to create project..."; \
		catapult -shell -file scripts/run_block_processor_catapult.tcl; \
		PRJ=$$(ls -t block_processor*.ccs 2>/dev/null | head -n 1); \
	fi; \
	if [ -n "$$PRJ" ]; then \
		catapult "$$PRJ" & \
	else \
		echo "Could not find block_processor*.ccs project file."; exit 1; \
	fi

# ============================================================================
# Stem Processor Targets
# ============================================================================

# Run stem processor Catapult in batch mode with log (auto-cleans old project)
stem: stem-clean
	cd Binary_cnn && mkdir -p logs && catapult -shell -file scripts/run_stem_catapult.tcl 2>&1 | tee logs/catapult_stem.log

# Alias for stem
stem-log: stem

# Run stem processor testbench in auto mode
# - all 3 files exist: stem-tb-weight
# - none exist: stem-tb-no-weight
# - partial files: error
stem-tb:
	@HAS_W=0; HAS_S=0; HAS_B=0; \
	if [ -f "Binary_cnn/weights/stem_weights.txt" ]; then HAS_W=1; fi; \
	if [ -f "Binary_cnn/weights/stem_bn_scale.txt" ]; then HAS_S=1; fi; \
	if [ -f "Binary_cnn/weights/stem_bn_bias.txt" ]; then HAS_B=1; fi; \
	COUNT=$$((HAS_W + HAS_S + HAS_B)); \
	if [ "$$COUNT" -eq 3 ]; then \
		echo "[stem-tb] Found all weight/BN files. Running stem-tb-weight."; \
		$(MAKE) stem-tb-weight; \
	elif [ "$$COUNT" -eq 0 ]; then \
		echo "[stem-tb] No weight/BN files found. Running stem-tb-no-weight."; \
		$(MAKE) stem-tb-no-weight; \
	else \
		echo "[stem-tb] Partial weight/BN files detected. Provide all 3 files or none."; \
		if [ "$$HAS_W" -eq 0 ]; then echo "Missing: Binary_cnn/weights/stem_weights.txt"; fi; \
		if [ "$$HAS_S" -eq 0 ]; then echo "Missing: Binary_cnn/weights/stem_bn_scale.txt"; fi; \
		if [ "$$HAS_B" -eq 0 ]; then echo "Missing: Binary_cnn/weights/stem_bn_bias.txt"; fi; \
		exit 1; \
	fi

# Run stem processor testbench with real weight/BN files
# Required files:
#   Binary_cnn/weights/stem_weights.txt
#   Binary_cnn/weights/stem_bn_scale.txt
#   Binary_cnn/weights/stem_bn_bias.txt
stem-tb-weight: stem-clean weight-dir
	@if [ ! -f "Binary_cnn/weights/stem_weights.txt" ]; then echo "Missing Binary_cnn/weights/stem_weights.txt"; exit 1; fi
	@if [ ! -f "Binary_cnn/weights/stem_bn_scale.txt" ]; then echo "Missing Binary_cnn/weights/stem_bn_scale.txt"; exit 1; fi
	@if [ ! -f "Binary_cnn/weights/stem_bn_bias.txt" ]; then echo "Missing Binary_cnn/weights/stem_bn_bias.txt"; exit 1; fi
	cd Binary_cnn && mkdir -p logs && catapult -shell -file scripts/run_stem_catapult.tcl 2>&1 | tee logs/catapult_stem.log
	cd Binary_cnn && ROOT_DIR=$$(pwd); \
	SOL_DIR=$$(ls -d stem_processor/StemProcessor.v* stem_processor/solution.v* 2>/dev/null | sort -V | tail -n 1); \
	if [ -z "$$SOL_DIR" ]; then echo "No stem solution directory found (StemProcessor.v* or solution.v*)."; exit 1; fi; \
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
	if [ -z "$$SCV_MK" ]; then echo "SCVerify Makefile not found under $$SOL_DIR"; exit 1; fi; \
	SCV_DIR=$$(dirname "$$SCV_MK"); \
	SCV_BASENAME=$$(basename "$$SCV_MK"); \
	if [ ! -f "$$SCV_DIR/ccs_env.mk" ]; then touch "$$SCV_DIR/ccs_env.mk"; fi; \
	CXX_HOME_RUN="$$CXX_HOME"; \
	if [ -z "$$CXX_HOME_RUN" ]; then \
		CXX_BIN=$$(command -v g++ 2>/dev/null || command -v c++ 2>/dev/null); \
		if [ -n "$$CXX_BIN" ]; then CXX_HOME_RUN=$$(dirname $$(dirname "$$CXX_BIN")); fi; \
	fi; \
	if [ -z "$$CXX_HOME_RUN" ]; then \
		echo "CXX_HOME is not set and g++/c++ was not found in PATH."; \
		exit 1; \
	fi; \
	SYSTEMC_INCDIR_RUN="$$SYSTEMC_INCDIR"; \
	if [ -z "$$SYSTEMC_INCDIR_RUN" ] && [ -n "$$MGC_HOME" ] && [ -f "$$MGC_HOME/shared/include/systemc.h" ]; then \
		SYSTEMC_INCDIR_RUN="$$MGC_HOME/shared/include"; \
	fi; \
	if [ -z "$$SYSTEMC_INCDIR_RUN" ]; then \
		CATAPULT_BIN=$$(command -v catapult 2>/dev/null); \
		if [ -n "$$CATAPULT_BIN" ]; then \
			CATAPULT_BIN_REAL="$$CATAPULT_BIN"; \
			if command -v readlink >/dev/null 2>&1; then \
				CATAPULT_BIN_REAL=$$(readlink -f "$$CATAPULT_BIN" 2>/dev/null || echo "$$CATAPULT_BIN"); \
			fi; \
			CATAPULT_ROOT=$$(cd $$(dirname "$$CATAPULT_BIN_REAL")/.. 2>/dev/null && pwd); \
			if [ -f "$$CATAPULT_ROOT/Mgc_home/shared/include/systemc.h" ]; then \
				SYSTEMC_INCDIR_RUN="$$CATAPULT_ROOT/Mgc_home/shared/include"; \
			fi; \
		fi; \
	fi; \
	if [ -z "$$SYSTEMC_INCDIR_RUN" ]; then \
		for d in /usr/local/systemc/include /usr/include/systemc /usr/include; do \
			if [ -f "$$d/systemc.h" ]; then SYSTEMC_INCDIR_RUN="$$d"; break; fi; \
		done; \
	fi; \
	if [ -z "$$SYSTEMC_INCDIR_RUN" ]; then \
		echo "SYSTEMC_INCDIR is not set and systemc.h was not found."; \
		exit 1; \
	fi; \
	SYSTEMC_LIBDIR_RUN="$$SYSTEMC_LIBDIR"; \
	if [ -z "$$SYSTEMC_LIBDIR_RUN" ]; then \
		for d in "$$(dirname "$$SYSTEMC_INCDIR_RUN")/lib-linux64" "$$(dirname "$$SYSTEMC_INCDIR_RUN")/lib64" "$$(dirname "$$SYSTEMC_INCDIR_RUN")/lib"; do \
			if ls "$$d"/libsystemc* >/dev/null 2>&1; then SYSTEMC_LIBDIR_RUN="$$d"; break; fi; \
		done; \
	fi; \
	printf "CXX_HOME := %s\nSYSTEMC_INCDIR := %s\n" "$$CXX_HOME_RUN" "$$SYSTEMC_INCDIR_RUN" > "$$SCV_DIR/ccs_env.mk"; \
	if [ -n "$$SYSTEMC_LIBDIR_RUN" ]; then printf "SYSTEMC_LIBDIR := %s\n" "$$SYSTEMC_LIBDIR_RUN" >> "$$SCV_DIR/ccs_env.mk"; fi; \
	SCV_LOG="logs/scverify_stem_sim.log"; \
	if [ "$$SCV_BASENAME" = "Makefile" ]; then \
		STEM_WEIGHT_FILE="$$ROOT_DIR/weights/stem_weights.txt" \
		STEM_BN_SCALE_FILE="$$ROOT_DIR/weights/stem_bn_scale.txt" \
		STEM_BN_BIAS_FILE="$$ROOT_DIR/weights/stem_bn_bias.txt" \
		CXX_HOME="$$CXX_HOME_RUN" \
		SYSTEMC_INCDIR="$$SYSTEMC_INCDIR_RUN" \
		$(MAKE) -C "$$SCV_DIR" \
		CXX_HOME="$$CXX_HOME_RUN" \
		SYSTEMC_INCDIR="$$SYSTEMC_INCDIR_RUN" \
		SYSTEMC_LIBDIR="$$SYSTEMC_LIBDIR_RUN" \
		sim > "$$SCV_LOG" 2>&1; \
		SIM_RC=$$?; \
	else \
		STEM_WEIGHT_FILE="$$ROOT_DIR/weights/stem_weights.txt" \
		STEM_BN_SCALE_FILE="$$ROOT_DIR/weights/stem_bn_scale.txt" \
		STEM_BN_BIAS_FILE="$$ROOT_DIR/weights/stem_bn_bias.txt" \
		CXX_HOME="$$CXX_HOME_RUN" \
		SYSTEMC_INCDIR="$$SYSTEMC_INCDIR_RUN" \
		$(MAKE) -C "$$SCV_DIR" -f "$$SCV_BASENAME" \
		CXX_HOME="$$CXX_HOME_RUN" \
		SYSTEMC_INCDIR="$$SYSTEMC_INCDIR_RUN" \
		SYSTEMC_LIBDIR="$$SYSTEMC_LIBDIR_RUN" \
		> "$$SCV_LOG" 2>&1; \
		SIM_RC=$$?; \
	fi; \
	cat "$$SCV_LOG"; \
	if [ "$$SIM_RC" -ne 0 ]; then \
		echo "[stem-tb] FAIL: SCVerify returned $$SIM_RC"; \
		exit "$$SIM_RC"; \
	fi; \
	if grep -q "\\*\\*\\* TEST PASSED \\*\\*\\*" "$$SCV_LOG"; then \
		echo "[stem-tb] PASS"; \
	elif grep -q "\\*\\*\\* TEST FAILED \\*\\*\\*" "$$SCV_LOG"; then \
		echo "[stem-tb] FAIL: TB reported TEST FAILED"; \
		exit 1; \
	else \
		echo "[stem-tb] FAIL: PASS marker not found in $$SCV_LOG"; \
		exit 1; \
	fi

# Run stem processor testbench without external weight/BN files
# (forces TB fallback: pseudo-random weights + identity BN)
stem-tb-no-weight: stem-clean
	cd Binary_cnn && mkdir -p logs && catapult -shell -file scripts/run_stem_catapult.tcl 2>&1 | tee logs/catapult_stem.log
	cd Binary_cnn && SOL_DIR=$$(ls -d stem_processor/StemProcessor.v* stem_processor/solution.v* 2>/dev/null | sort -V | tail -n 1); \
	if [ -z "$$SOL_DIR" ]; then echo "No stem solution directory found (StemProcessor.v* or solution.v*)."; exit 1; fi; \
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
	if [ -z "$$SCV_MK" ]; then echo "SCVerify Makefile not found under $$SOL_DIR"; exit 1; fi; \
	SCV_DIR=$$(dirname "$$SCV_MK"); \
	SCV_BASENAME=$$(basename "$$SCV_MK"); \
	if [ ! -f "$$SCV_DIR/ccs_env.mk" ]; then touch "$$SCV_DIR/ccs_env.mk"; fi; \
	CXX_HOME_RUN="$$CXX_HOME"; \
	if [ -z "$$CXX_HOME_RUN" ]; then \
		CXX_BIN=$$(command -v g++ 2>/dev/null || command -v c++ 2>/dev/null); \
		if [ -n "$$CXX_BIN" ]; then CXX_HOME_RUN=$$(dirname $$(dirname "$$CXX_BIN")); fi; \
	fi; \
	if [ -z "$$CXX_HOME_RUN" ]; then \
		echo "CXX_HOME is not set and g++/c++ was not found in PATH."; \
		exit 1; \
	fi; \
	SYSTEMC_INCDIR_RUN="$$SYSTEMC_INCDIR"; \
	if [ -z "$$SYSTEMC_INCDIR_RUN" ] && [ -n "$$MGC_HOME" ] && [ -f "$$MGC_HOME/shared/include/systemc.h" ]; then \
		SYSTEMC_INCDIR_RUN="$$MGC_HOME/shared/include"; \
	fi; \
	if [ -z "$$SYSTEMC_INCDIR_RUN" ]; then \
		CATAPULT_BIN=$$(command -v catapult 2>/dev/null); \
		if [ -n "$$CATAPULT_BIN" ]; then \
			CATAPULT_BIN_REAL="$$CATAPULT_BIN"; \
			if command -v readlink >/dev/null 2>&1; then \
				CATAPULT_BIN_REAL=$$(readlink -f "$$CATAPULT_BIN" 2>/dev/null || echo "$$CATAPULT_BIN"); \
			fi; \
			CATAPULT_ROOT=$$(cd $$(dirname "$$CATAPULT_BIN_REAL")/.. 2>/dev/null && pwd); \
			if [ -f "$$CATAPULT_ROOT/Mgc_home/shared/include/systemc.h" ]; then \
				SYSTEMC_INCDIR_RUN="$$CATAPULT_ROOT/Mgc_home/shared/include"; \
			fi; \
		fi; \
	fi; \
	if [ -z "$$SYSTEMC_INCDIR_RUN" ]; then \
		for d in /usr/local/systemc/include /usr/include/systemc /usr/include; do \
			if [ -f "$$d/systemc.h" ]; then SYSTEMC_INCDIR_RUN="$$d"; break; fi; \
		done; \
	fi; \
	if [ -z "$$SYSTEMC_INCDIR_RUN" ]; then \
		echo "SYSTEMC_INCDIR is not set and systemc.h was not found."; \
		exit 1; \
	fi; \
	SYSTEMC_LIBDIR_RUN="$$SYSTEMC_LIBDIR"; \
	if [ -z "$$SYSTEMC_LIBDIR_RUN" ]; then \
		for d in "$$(dirname "$$SYSTEMC_INCDIR_RUN")/lib-linux64" "$$(dirname "$$SYSTEMC_INCDIR_RUN")/lib64" "$$(dirname "$$SYSTEMC_INCDIR_RUN")/lib"; do \
			if ls "$$d"/libsystemc* >/dev/null 2>&1; then SYSTEMC_LIBDIR_RUN="$$d"; break; fi; \
		done; \
	fi; \
	printf "CXX_HOME := %s\nSYSTEMC_INCDIR := %s\n" "$$CXX_HOME_RUN" "$$SYSTEMC_INCDIR_RUN" > "$$SCV_DIR/ccs_env.mk"; \
	if [ -n "$$SYSTEMC_LIBDIR_RUN" ]; then printf "SYSTEMC_LIBDIR := %s\n" "$$SYSTEMC_LIBDIR_RUN" >> "$$SCV_DIR/ccs_env.mk"; fi; \
	SCV_LOG="logs/scverify_stem_sim.log"; \
	if [ "$$SCV_BASENAME" = "Makefile" ]; then \
		STEM_WEIGHT_FILE= STEM_BN_SCALE_FILE= STEM_BN_BIAS_FILE= \
		CXX_HOME="$$CXX_HOME_RUN" \
		SYSTEMC_INCDIR="$$SYSTEMC_INCDIR_RUN" \
		$(MAKE) -C "$$SCV_DIR" \
		CXX_HOME="$$CXX_HOME_RUN" \
		SYSTEMC_INCDIR="$$SYSTEMC_INCDIR_RUN" \
		SYSTEMC_LIBDIR="$$SYSTEMC_LIBDIR_RUN" \
		sim > "$$SCV_LOG" 2>&1; \
		SIM_RC=$$?; \
	else \
		STEM_WEIGHT_FILE= STEM_BN_SCALE_FILE= STEM_BN_BIAS_FILE= \
		CXX_HOME="$$CXX_HOME_RUN" \
		SYSTEMC_INCDIR="$$SYSTEMC_INCDIR_RUN" \
		$(MAKE) -C "$$SCV_DIR" -f "$$SCV_BASENAME" \
		CXX_HOME="$$CXX_HOME_RUN" \
		SYSTEMC_INCDIR="$$SYSTEMC_INCDIR_RUN" \
		SYSTEMC_LIBDIR="$$SYSTEMC_LIBDIR_RUN" \
		> "$$SCV_LOG" 2>&1; \
		SIM_RC=$$?; \
	fi; \
	cat "$$SCV_LOG"; \
	if [ "$$SIM_RC" -ne 0 ]; then \
		echo "[stem-tb] FAIL: SCVerify returned $$SIM_RC"; \
		exit "$$SIM_RC"; \
	fi; \
	if grep -q "\\*\\*\\* TEST PASSED \\*\\*\\*" "$$SCV_LOG"; then \
		echo "[stem-tb] PASS"; \
	elif grep -q "\\*\\*\\* TEST FAILED \\*\\*\\*" "$$SCV_LOG"; then \
		echo "[stem-tb] FAIL: TB reported TEST FAILED"; \
		exit 1; \
	else \
		echo "[stem-tb] FAIL: PASS marker not found in $$SCV_LOG"; \
		exit 1; \
	fi

# Run stem processor with GUI
stem-gui:
	cd Binary_cnn && PRJ=$$(ls -t stem_processor*.ccs 2>/dev/null | head -n 1); \
	if [ -z "$$PRJ" ]; then \
		echo "No stem_processor*.ccs found. Running batch flow once to create project..."; \
		catapult -shell -file scripts/run_stem_catapult.tcl; \
		PRJ=$$(ls -t stem_processor*.ccs 2>/dev/null | head -n 1); \
	fi; \
	if [ -n "$$PRJ" ]; then \
		catapult "$$PRJ" & \
	else \
		echo "Could not find stem_processor*.ccs project file."; exit 1; \
	fi
