.PHONY: update stream stream-log stream-tb stream-gui stream-clean block block-log block-gui block-tb block-clean stem stem-log stem-gui stem-tb stem-clean clean

# Update local repository to latest origin/dev
update:
	git fetch origin
	git checkout dev
	git pull origin dev

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
	if [ ! -f "$$SOL_DIR/scverify/Makefile" ]; then echo "SCVerify Makefile not found: $$SOL_DIR/scverify/Makefile"; exit 1; fi; \
	$$(MAKE) -C "$$SOL_DIR/scverify" sim 2>&1 | tee logs/scverify_sim.log

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

# Run block processor Catapult in batch mode (auto-cleans old project)
block: block-clean
	cd Binary_cnn && catapult -shell -file scripts/run_block_processor_catapult.tcl

# Run block processor with log (visible in terminal + saved to file)
block-log: block-clean
	cd Binary_cnn && mkdir -p logs && catapult -shell -file scripts/run_block_processor_catapult.tcl 2>&1 | tee logs/catapult_block_processor.log

# Run block processor batch flow and execute SCVerify testbench simulation
block-tb: block-clean
	cd Binary_cnn && mkdir -p logs && catapult -shell -file scripts/run_block_processor_catapult.tcl 2>&1 | tee logs/catapult_block_processor.log
	cd Binary_cnn && SOL_DIR=$$(ls -d block_processor/FusedBlockProcessor.v* 2>/dev/null | sort -V | tail -n 1); \
	if [ -z "$$SOL_DIR" ]; then echo "No FusedBlockProcessor.v* solution directory found."; exit 1; fi; \
	SCV_MK=$$(find "$$SOL_DIR" -type f -path "*/scverify/Makefile" 2>/dev/null | sort -V | tail -n 1); \
	if [ -z "$$SCV_MK" ]; then echo "SCVerify Makefile not found under $$SOL_DIR"; exit 1; fi; \
	SCV_DIR=$$(dirname "$$SCV_MK"); \
	$$(MAKE) -C "$$SCV_DIR" sim 2>&1 | tee logs/scverify_block_processor_sim.log

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

# Run stem processor Catapult in batch mode (auto-cleans old project)
stem: stem-clean
	cd Binary_cnn && catapult -shell -file scripts/run_stem_catapult.tcl

# Run stem processor with log
stem-log: stem-clean
	cd Binary_cnn && mkdir -p logs && catapult -shell -file scripts/run_stem_catapult.tcl 2>&1 | tee logs/catapult_stem.log

# Run stem processor batch flow and execute SCVerify testbench simulation
stem-tb: stem-clean
	cd Binary_cnn && mkdir -p logs && catapult -shell -file scripts/run_stem_catapult.tcl 2>&1 | tee logs/catapult_stem.log
	cd Binary_cnn && SOL_DIR=$$(ls -d stem_processor/StemProcessor.v* 2>/dev/null | sort -V | tail -n 1); \
	if [ -z "$$SOL_DIR" ]; then echo "No StemProcessor.v* solution directory found."; exit 1; fi; \
	SCV_MK=$$(find "$$SOL_DIR" -type f -path "*/scverify/Makefile" 2>/dev/null | sort -V | tail -n 1); \
	if [ -z "$$SCV_MK" ]; then echo "SCVerify Makefile not found under $$SOL_DIR"; exit 1; fi; \
	SCV_DIR=$$(dirname "$$SCV_MK"); \
	$$(MAKE) -C "$$SCV_DIR" sim 2>&1 | tee logs/scverify_stem_sim.log

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
