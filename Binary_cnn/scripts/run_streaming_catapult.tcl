# Catapult batch script for BW_CNN_Streaming
# Run example:
#   cd Binary_cnn
#   catapult -shell -file scripts/run_streaming_catapult.tcl

project new -name bin_cnn_streaming

# Start from a clean solution state
solution new -state initial
solution options defaults

# Useful default flow options
solution options set /Flows/Enable-SCVerify yes
solution options set /Output/GenerateCycleNetlist false

# Design + testbench files (TB excluded from synthesis)
solution file add ./src/streaming/bw_cnn_streaming.h -type C++
solution file add ./src/streaming/bw_cnn_streaming.cpp -type C++
solution file add ./src/streaming/bw_cnn_streaming_tb.cpp -type C++ -exclude true

# Analyze/compile
go analyze
go compile

# Optional: generate SCVerify build/run scripts
flow package require /SCVerify

exit
