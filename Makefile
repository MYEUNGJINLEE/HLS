.PHONY: update stream stream-log stream-tb stream-gui

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

# Run Catapult with GUI and keep window open
stream-gui:
	cd Binary_cnn && catapult -gui -file scripts/run_streaming_catapult_gui.tcl
