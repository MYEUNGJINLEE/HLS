.PHONY: update stream stream-log stream-tb stream-gui block block-log block-gui block-tb

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
	cd Binary_cnn && mkdir -p logs && catapult -shell -file scripts/run_streaming_catapult.tcl > logs/catapult_streaming.log 2>&1

# Run Catapult batch flow and execute SCVerify testbench simulation
stream-tb:
	cd Binary_cnn && mkdir -p logs && catapult -shell -file scripts/run_streaming_catapult.tcl > logs/catapult_streaming.log 2>&1
	cd Binary_cnn && SOL_DIR=$$(ls -d bin_cnn_streaming/BW_CNN_Streaming.v* BW_CNN_Streaming.v* 2>/dev/null | sort -V | tail -n 1); \
	if [ -z "$$SOL_DIR" ]; then echo "No BW_CNN_Streaming.v* solution directory found."; exit 1; fi; \
	if [ ! -f "$$SOL_DIR/scverify/Makefile" ]; then echo "SCVerify Makefile not found: $$SOL_DIR/scverify/Makefile"; exit 1; fi; \
	$$(MAKE) -C "$$SOL_DIR/scverify" sim | tee logs/scverify_sim.log

# Run block processor Catapult in batch mode
block:
	cd Binary_cnn && catapult -shell -file scripts/run_block_processor_catapult.tcl

# Run block processor with log
block-log:
	cd Binary_cnn && mkdir -p logs && catapult -shell -file scripts/run_block_processor_catapult.tcl > logs/catapult_block_processor.log 2>&1


# Run block processor batch flow and execute SCVerify testbench simulation
block-tb:
	cd Binary_cnn && mkdir -p logs && catapult -shell -file scripts/run_block_processor_catapult.tcl > logs/catapult_block_processor.log 2>&1
	cd Binary_cnn && SOL_DIR=$$(ls -d block_processor* 2>/dev/null | sort -V | tail -n 1); \
	if [ -z "$$SOL_DIR" ]; then echo "No block_processor* project directory found."; exit 1; fi; \
	SCV_MK=$$(find "$$SOL_DIR" -type f -path "*/scverify/Makefile" 2>/dev/null | sort -V | tail -n 1); \
	if [ -z "$$SCV_MK" ]; then echo "SCVerify Makefile not found under $$SOL_DIR"; exit 1; fi; \
	SCV_DIR=$$(dirname "$$SCV_MK"); \
	$$(MAKE) -C "$$SCV_DIR" sim | tee logs/scverify_block_processor_sim.log

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
	  echo "No block_processor*.ccs found. Running GUI flow once to create project..."; \
	  catapult -gui -file scripts/run_block_processor_catapult_gui.tcl & \
	else \
	  catapult "$$PRJ" & \
	fi