# Catapult GUI script for BW_CNN_Streaming
# Run example:
#   cd Binary_cnn
#   catapult -gui -file scripts/run_streaming_catapult_gui.tcl

project new -name bin_cnn_streaming

solution new -state initial
solution options defaults
solution options set /Flows/Enable-SCVerify yes
solution options set /Output/GenerateCycleNetlist false

solution file add ./src/streaming/bw_cnn_streaming.h -type C++
solution file add ./src/streaming/bw_cnn_streaming.cpp -type C++
solution file add ./src/streaming/bw_cnn_streaming_tb.cpp -type C++ -exclude true

go analyze
go compile
flow package require /SCVerify

# NOTE: no 'exit' so GUI stays open
