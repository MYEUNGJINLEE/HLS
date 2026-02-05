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

# Design + testbench files
solution file add ./src/streaming/bw_cnn_streaming.h -type C++
solution file add ./src/streaming/block_config.h -type C++
solution file add ./src/streaming/tile_manager.h -type C++
solution file add ./src/streaming/inter_layer_buffer.h -type C++
solution file add ./src/streaming/conv_compute.h -type C++
solution file add ./src/streaming/block_processor.h -type C++
solution file add ./src/streaming/block_processor.cpp -type C++
solution file add ./src/streaming/block_processor_tb.cpp -type C++ -exclude true

# Analyze/compile
go analyze
go compile

# Optional: generate SCVerify build/run scripts
flow package require /SCVerify

exit
