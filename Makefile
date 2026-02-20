.PHONY: update push push-compile-results weight-dir stream stream-log stream-tb stream-gui stream-clean block block-log block-gui block-tb block-clean sched sched-log sched-gui sched-clean gpt gpt-log gpt-gui gpt-clean stem stem-log stem-gui stem-gui-build stem-tb stem-tb-weight stem-tb-no-weight stem-clean stem2 stem2-log stem2-tb stem2-gui stem2-clean three-pe three-pe-log three-pe-gui three-pe-tb three-pe-clean backbone backbone-log backbone-gui backbone-tb backbone-clean clean-artifacts clean
.SILENT:

# Update local repository to latest origin/dev (stash handles unstaged changes)
update: ; @git stash push -u -m "auto-stash: make update" >/dev/null 2>&1 || git stash save -u "auto-stash: make update" >/dev/null 2>&1 || true; git fetch origin; git checkout dev; git pull --no-rebase origin dev; git stash pop >/dev/null 2>&1 || true

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

# Clean three PE scheduler projects (including numbered variants _1, _2, ...)
sched-clean:
	cd Binary_cnn && rm -rf three_pe_scheduler* three_pe_scheduler.ccs

# Clean GPT backbone phase3 projects (including numbered variants _1, _2, ...)
gpt-clean:
	cd Binary_cnn && rm -rf gpt_backbone_phase3* gpt_backbone_phase3.ccs

# Clean stem processor projects
stem-clean:
	cd Binary_cnn && rm -rf stem_processor* stem_processor.ccs

# Clean stem_v2 processor projects
stem2-clean:
	cd Binary_cnn && rm -rf stem_v2_processor* stem_v2_processor.ccs

# Clean all Catapult-generated artifacts
clean-artifacts:
	rm -rf \
		Binary_cnn/bin_cnn \
		Binary_cnn/bin_cnn.ccs \
		Binary_cnn/Catapult \
		Binary_cnn/Catapult.ccs \
		Binary_cnn/logs \
		Binary_cnn/catapult.log \
		Binary_cnn/catapult_pid* \
		Binary_cnn/.Catapult* \
		catapult.log \
		catapult_pid* \
		.Catapult*
	find Binary_cnn -type d \( -name CDesignChecker -o -name scverify \) -prune -exec rm -rf {} +
	find Binary_cnn -type f \( -name '*.ccs' -o -name '*.vcd' -o -name '*.wlf' -o -name 'transcript' \) -delete

three-pe-clean:
	cd Binary_cnn && rm -rf three_pe_block* three_pe_block.ccs

# Run ThreePEBlock Catapult in batch mode with log (auto-cleans old project)
three-pe: three-pe-clean
	cd Binary_cnn && mkdir -p logs && catapult -shell -file scripts/run_three_pe_catapult.tcl 2>&1 | tee logs/catapult_three_pe_block.log

# Alias for three-pe
three-pe-log: three-pe

# Run ThreePEBlock with GUI (reuses existing project or runs batch first)
three-pe-gui:
	cd Binary_cnn && PRJ=$$(ls -t three_pe_block*.ccs 2>/dev/null | head -n 1); \
	if [ -z "$$PRJ" ]; then \
		echo "No three_pe_block*.ccs found. Running batch flow once to create project..."; \
		catapult -shell -file scripts/run_three_pe_catapult.tcl; \
		PRJ=$$(ls -t three_pe_block*.ccs 2>/dev/null | head -n 1); \
	fi; \
	if [ -n "$$PRJ" ]; then \
		catapult "$$PRJ" & \
	else \
		echo "Could not find three_pe_block*.ccs project file."; exit 1; \
	fi

# Run ThreePEBlock batch flow then SCVerify testbench simulation
three-pe-tb: three-pe-clean
	cd Binary_cnn && mkdir -p logs && catapult -shell -file scripts/run_three_pe_catapult.tcl 2>&1 | tee logs/catapult_three_pe_block.log
	cd Binary_cnn && SOL_DIR=$$(ls -d three_pe_block/ThreePEBlock.v* 2>/dev/null | sort -V | tail -n 1); \
	if [ -z "$$SOL_DIR" ]; then echo "No ThreePEBlock.v* solution directory found."; exit 1; fi; \
	SCV_MK=$$(find "$$SOL_DIR" -type f -path "*/scverify/Makefile" 2>/dev/null | sort -V | tail -n 1); \
	if [ -z "$$SCV_MK" ]; then \
		SCV_MK=$$(find "$$SOL_DIR" -type f -path "*/scverify/Verify_*.mk" 2>/dev/null | sort -V | tail -n 1); \
	fi; \
	if [ -z "$$SCV_MK" ]; then echo "SCVerify Makefile not found under $$SOL_DIR"; exit 1; fi; \
	SCV_DIR=$$(dirname "$$SCV_MK"); \
	SCV_BASENAME=$$(basename "$$SCV_MK"); \
	$(MAKE) -C "$$SCV_DIR" -f "$$SCV_BASENAME" sim 2>&1 | tee logs/scverify_three_pe_block_sim.log

backbone-clean:
	cd Binary_cnn && rm -rf backbone_block1* backbone_block1.ccs

# Run BackboneBlock1 Catapult in batch mode with log (auto-cleans old project)
backbone: backbone-clean
	cd Binary_cnn && mkdir -p logs && catapult -shell -file scripts/run_backbone_catapult.tcl 2>&1 | tee logs/catapult_backbone_block1.log

# Alias for backbone
backbone-log: backbone

# Run BackboneBlock1 with GUI (reuses existing project or runs batch first)
backbone-gui:
	cd Binary_cnn && PRJ=$$(ls -t backbone_block1*.ccs 2>/dev/null | head -n 1); \
	if [ -z "$$PRJ" ]; then \
		echo "No backbone_block1*.ccs found. Running batch flow once to create project..."; \
		catapult -shell -file scripts/run_backbone_catapult.tcl; \
		PRJ=$$(ls -t backbone_block1*.ccs 2>/dev/null | head -n 1); \
	fi; \
	if [ -n "$$PRJ" ]; then \
		catapult "$$PRJ" & \
	else \
		echo "Could not find backbone_block1*.ccs project file."; exit 1; \
	fi

# Run BackboneBlock1 batch flow then SCVerify testbench simulation
backbone-tb: backbone-clean
	cd Binary_cnn && mkdir -p logs && catapult -shell -file scripts/run_backbone_catapult.tcl 2>&1 | tee logs/catapult_backbone_block1.log
	cd Binary_cnn && SOL_DIR=$$(ls -d backbone_block1/BackboneBlock1.v* 2>/dev/null | sort -V | tail -n 1); \
	if [ -z "$$SOL_DIR" ]; then echo "No BackboneBlock1.v* solution directory found."; exit 1; fi; \
	SCV_MK=$$(find "$$SOL_DIR" -type f -path "*/scverify/Makefile" 2>/dev/null | sort -V | tail -n 1); \
	if [ -z "$$SCV_MK" ]; then \
		SCV_MK=$$(find "$$SOL_DIR" -type f -path "*/scverify/Verify_*.mk" 2>/dev/null | sort -V | tail -n 1); \
	fi; \
	if [ -z "$$SCV_MK" ]; then echo "SCVerify Makefile not found under $$SOL_DIR"; exit 1; fi; \
	SCV_DIR=$$(dirname "$$SCV_MK"); \
	SCV_BASENAME=$$(basename "$$SCV_MK"); \
	$(MAKE) -C "$$SCV_DIR" -f "$$SCV_BASENAME" sim 2>&1 | tee logs/scverify_backbone_block1_sim.log

clean: stream-clean block-clean sched-clean gpt-clean stem-clean stem2-clean three-pe-clean backbone-clean clean-artifacts

# Run block processor Catapult in batch mode with log (auto-cleans old project)
block: block-clean
	cd Binary_cnn && mkdir -p logs && catapult -shell -file scripts/run_block_processor_catapult.tcl 2>&1 | tee logs/catapult_block_processor.log

# Alias for block
block-log: block

# Run three PE scheduler Catapult in batch mode with log (auto-cleans old project)
sched: sched-clean
	cd Binary_cnn && mkdir -p logs && catapult -shell -file scripts/run_three_pe_scheduler_catapult.tcl 2>&1 | tee logs/catapult_three_pe_scheduler.log

# Alias for sched
sched-log: sched

# Run GPT backbone phase3 Catapult in batch mode with log (auto-cleans old project)
gpt: gpt-clean
	cd Binary_cnn && mkdir -p logs && catapult -shell -file scripts/run_gpt_backbone_phase3_catapult.tcl 2>&1 | tee logs/catapult_gpt_backbone_phase3.log

# Alias for gpt
gpt-log: gpt

# Run three PE scheduler with GUI and keep window open
sched-gui:
	cd Binary_cnn && PRJ=$$(ls -t three_pe_scheduler*.ccs 2>/dev/null | head -n 1); \
	if [ -z "$$PRJ" ]; then \
		echo "No three_pe_scheduler*.ccs found. Running batch flow once to create project..."; \
		catapult -shell -file scripts/run_three_pe_scheduler_catapult.tcl; \
		PRJ=$$(ls -t three_pe_scheduler*.ccs 2>/dev/null | head -n 1); \
	fi; \
	if [ -n "$$PRJ" ]; then \
		catapult "$$PRJ" & \
	else \
		echo "Could not find three_pe_scheduler*.ccs project file."; exit 1; \
	fi

# Run GPT backbone phase3 with GUI and keep window open
gpt-gui:
	cd Binary_cnn && PRJ=$$(ls -t gpt_backbone_phase3*.ccs 2>/dev/null | head -n 1); \
	if [ -z "$$PRJ" ]; then \
		echo "No gpt_backbone_phase3*.ccs found. Running batch flow once to create project..."; \
		catapult -shell -file scripts/run_gpt_backbone_phase3_catapult.tcl; \
		PRJ=$$(ls -t gpt_backbone_phase3*.ccs 2>/dev/null | head -n 1); \
	fi; \
	if [ -n "$$PRJ" ]; then \
		catapult "$$PRJ" & \
	else \
		echo "Could not find gpt_backbone_phase3*.ccs project file."; exit 1; \
	fi

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

# Run stem processor compile flow, then open GUI project file
stem: stem-clean
	cd Binary_cnn && catapult -shell -file scripts/run_stem_catapult.tcl
	cd Binary_cnn && \
	if [ -z "$$DISPLAY" ]; then \
		echo "[stem] Compile finished. DISPLAY is empty, skipping GUI open."; \
		echo "[stem] Open later with: catapult stem_processor.ccs"; \
		exit 0; \
	fi; \
	PRJ=$$(ls -t stem_processor*.ccs 2>/dev/null | head -n 1); \
	if [ -n "$$PRJ" ]; then \
		echo "[stem] Opening GUI project: $$PRJ"; \
		catapult "$$PRJ" & \
	else \
		echo "[stem] Could not find stem_processor*.ccs project file."; \
		exit 1; \
	fi

# Run stem processor in batch mode
stem-log: stem-clean
	cd Binary_cnn && catapult -shell -file scripts/run_stem_catapult.tcl

# Run stem processor testbench in auto mode
# - packed weight file exists: stem-tb-weight
# - missing packed weight file: stem-tb-no-weight
stem-tb:
	@HAS_W=0; \
	if [ -f "Binary_cnn/weights/stem_weights.txt" ]; then HAS_W=1; fi; \
	if [ "$$HAS_W" -eq 1 ]; then \
		echo "[stem-tb] Found packed weight file. Running stem-tb-weight."; \
		$(MAKE) stem-tb-weight; \
	else \
		echo "[stem-tb] Packed weight file not found. Running stem-tb-no-weight."; \
		$(MAKE) stem-tb-no-weight; \
	fi

# Run stem processor testbench with packed weight stream file
# Required file:
#   Binary_cnn/weights/stem_weights.txt
stem-tb-weight: stem-clean weight-dir
	@if [ ! -f "Binary_cnn/weights/stem_weights.txt" ]; then echo "Missing Binary_cnn/weights/stem_weights.txt"; exit 1; fi
	cd Binary_cnn && catapult -shell -file scripts/run_stem_catapult.tcl
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
	PROJ_DIR=$$(dirname "$$SOL_DIR"); \
	PROJ_DIR_ABS="$$ROOT_DIR/$$PROJ_DIR"; \
	SCV_MK_ABS="$$ROOT_DIR/$$SCV_MK"; \
	SCV_LOG=$$(mktemp /tmp/scverify_stem_sim.XXXXXX.log); \
	LAUNCH_TCL=$$(mktemp /tmp/stem_scverify_launch.XXXXXX.tcl); \
	printf "if {![file isdirectory {%s}]} { error {missing stem project directory} }\nproject load {%s} 2025.2\nflow package require /SCVerify\nflow run /SCVerify/launch_make %s {} SIMTOOL=osci sim\nexit\n" "$$PROJ_DIR_ABS" "$$PROJ_DIR_ABS" "$$SCV_MK_ABS" > "$$LAUNCH_TCL"; \
	(cd "$$ROOT_DIR" && \
		STEM_WEIGHT_FILE="$$ROOT_DIR/weights/stem_weights.txt" \
		catapult -shell -file "$$LAUNCH_TCL" > "$$SCV_LOG" 2>&1); \
	SIM_RC=$$?; \
	rm -f "$$LAUNCH_TCL"; \
	cat "$$SCV_LOG"; \
	if [ "$$SIM_RC" -ne 0 ]; then \
		rm -f "$$SCV_LOG"; \
		echo "[stem-tb] FAIL: SCVerify returned $$SIM_RC"; \
		exit "$$SIM_RC"; \
	fi; \
	if grep -q "\\*\\*\\* TEST PASSED \\*\\*\\*" "$$SCV_LOG"; then \
		rm -f "$$SCV_LOG"; \
		echo "[stem-tb] PASS"; \
	elif grep -q "\\*\\*\\* TEST FAILED \\*\\*\\*" "$$SCV_LOG"; then \
		rm -f "$$SCV_LOG"; \
		echo "[stem-tb] FAIL: TB reported TEST FAILED"; \
		exit 1; \
	else \
		rm -f "$$SCV_LOG"; \
		echo "[stem-tb] FAIL: PASS marker not found in $$SCV_LOG"; \
		exit 1; \
	fi

# Run stem processor testbench without external packed weight file
# (forces TB fallback: pseudo-random packed weights/params)
stem-tb-no-weight: stem-clean
	cd Binary_cnn && catapult -shell -file scripts/run_stem_catapult.tcl
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
	PROJ_DIR=$$(dirname "$$SOL_DIR"); \
	PROJ_DIR_ABS="$$ROOT_DIR/$$PROJ_DIR"; \
	SCV_MK_ABS="$$ROOT_DIR/$$SCV_MK"; \
	SCV_LOG=$$(mktemp /tmp/scverify_stem_sim.XXXXXX.log); \
	LAUNCH_TCL=$$(mktemp /tmp/stem_scverify_launch.XXXXXX.tcl); \
	printf "if {![file isdirectory {%s}]} { error {missing stem project directory} }\nproject load {%s} 2025.2\nflow package require /SCVerify\nflow run /SCVerify/launch_make %s {} SIMTOOL=osci sim\nexit\n" "$$PROJ_DIR_ABS" "$$PROJ_DIR_ABS" "$$SCV_MK_ABS" > "$$LAUNCH_TCL"; \
	(cd "$$ROOT_DIR" && \
		STEM_WEIGHT_FILE= \
		catapult -shell -file "$$LAUNCH_TCL" > "$$SCV_LOG" 2>&1); \
	SIM_RC=$$?; \
	rm -f "$$LAUNCH_TCL"; \
	cat "$$SCV_LOG"; \
	if [ "$$SIM_RC" -ne 0 ]; then \
		rm -f "$$SCV_LOG"; \
		echo "[stem-tb] FAIL: SCVerify returned $$SIM_RC"; \
		exit "$$SIM_RC"; \
	fi; \
	if grep -q "\\*\\*\\* TEST PASSED \\*\\*\\*" "$$SCV_LOG"; then \
		rm -f "$$SCV_LOG"; \
		echo "[stem-tb] PASS"; \
	elif grep -q "\\*\\*\\* TEST FAILED \\*\\*\\*" "$$SCV_LOG"; then \
		rm -f "$$SCV_LOG"; \
		echo "[stem-tb] FAIL: TB reported TEST FAILED"; \
		exit 1; \
	else \
		rm -f "$$SCV_LOG"; \
		echo "[stem-tb] FAIL: PASS marker not found in $$SCV_LOG"; \
		exit 1; \
	fi

# ============================================================================
# Stem V2 Processor Targets (banking directives + reduced line buffer rows)
# ============================================================================

# Run stem_v2 processor Catapult in batch mode with log
stem2: stem2-clean
	cd Binary_cnn && mkdir -p logs && catapult -shell -file scripts/run_stem_v2_catapult.tcl 2>&1 | tee logs/catapult_stem_v2.log

# Alias
stem2-log: stem2

# Run stem_v2 processor testbench (pseudo-random weights + identity BN)
stem2-tb: stem2-clean
	cd Binary_cnn && mkdir -p logs && catapult -shell -file scripts/run_stem_v2_catapult.tcl 2>&1 | tee logs/catapult_stem_v2.log
	cd Binary_cnn && ROOT_DIR=$$(pwd); \
	SOL_DIR=$$(ls -d stem_v2_processor/StemProcessor.v* stem_v2_processor/solution.v* 2>/dev/null | sort -V | tail -n 1); \
	if [ -z "$$SOL_DIR" ]; then echo "No stem_v2 solution directory found."; exit 1; fi; \
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
	PROJ_DIR=$$(dirname "$$SOL_DIR"); \
	PROJ_DIR_ABS="$$ROOT_DIR/$$PROJ_DIR"; \
	SCV_MK_ABS="$$ROOT_DIR/$$SCV_MK"; \
	SCV_LOG="$$ROOT_DIR/logs/scverify_stem_v2_sim.log"; \
	LAUNCH_TCL="$$ROOT_DIR/logs/stem_v2_scverify_launch.tcl"; \
	printf "if {![file isdirectory {%s}]} { error {missing stem_v2 project directory} }\nproject load {%s} 2025.2\nflow package require /SCVerify\nflow run /SCVerify/launch_make %s {} SIMTOOL=osci sim\nexit\n" "$$PROJ_DIR_ABS" "$$PROJ_DIR_ABS" "$$SCV_MK_ABS" > "$$LAUNCH_TCL"; \
	(cd "$$ROOT_DIR" && \
		catapult -shell -file "$$LAUNCH_TCL" > "$$SCV_LOG" 2>&1); \
	SIM_RC=$$?; \
	rm -f "$$LAUNCH_TCL"; \
	cat "$$SCV_LOG"; \
	if [ "$$SIM_RC" -ne 0 ]; then \
		echo "[stem2-tb] FAIL: SCVerify returned $$SIM_RC"; \
		exit "$$SIM_RC"; \
	fi; \
	if grep -q "\\*\\*\\* TEST PASSED \\*\\*\\*" "$$SCV_LOG"; then \
		echo "[stem2-tb] PASS"; \
	elif grep -q "\\*\\*\\* TEST FAILED \\*\\*\\*" "$$SCV_LOG"; then \
		echo "[stem2-tb] FAIL: TB reported TEST FAILED"; \
		exit 1; \
	else \
		echo "[stem2-tb] FAIL: PASS marker not found in $$SCV_LOG"; \
		exit 1; \
	fi

# Run stem_v2 processor with GUI
stem2-gui:
	cd Binary_cnn && PRJ=$$(ls -t stem_v2_processor*.ccs 2>/dev/null | head -n 1); \
	if [ -z "$$PRJ" ]; then \
		echo "No stem_v2_processor*.ccs found. Running batch flow once..."; \
		catapult -shell -file scripts/run_stem_v2_catapult.tcl; \
		PRJ=$$(ls -t stem_v2_processor*.ccs 2>/dev/null | head -n 1); \
	fi; \
	if [ -n "$$PRJ" ]; then \
		catapult "$$PRJ" & \
	else \
		echo "Could not find stem_v2_processor*.ccs project file."; exit 1; \
	fi

# Run stem processor with GUI
stem-gui:
	cd Binary_cnn && PRJ="$$FILE"; \
	if [ -z "$$PRJ" ]; then PRJ=$$(ls -t stem_processor*.ccs 2>/dev/null | head -n 1); fi; \
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

# Alias (same as make stem)
stem-gui-build: stem
