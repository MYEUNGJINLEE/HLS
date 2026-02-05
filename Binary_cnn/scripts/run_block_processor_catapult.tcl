# Catapult batch script for FusedBlockProcessor
# Run example:
#   cd Binary_cnn
#   catapult -shell -file scripts/run_block_processor_catapult.tcl

project new -name block_processor

# Start from a clean solution state
solution new -state initial
solution options defaults

# Useful default flow options
solution options set /Flows/Enable-SCVerify yes
solution options set /Output/GenerateCycleNetlist false

# Specify top design (override BW_CNN_Streaming pragma from include chain)
solution options set /Input/TopDesignName FusedBlockProcessor

# Design + testbench files
solution file add ./src/streaming/block_processor.cpp -type C++
solution file add ./src/streaming/block_processor_tb.cpp -type C++ -exclude true

# Analyze/compile
go analyze
go compile

# Optional: generate SCVerify build/run scripts
flow package require /SCVerify

exit
