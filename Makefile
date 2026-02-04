.PHONY: update stream stream-log

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
