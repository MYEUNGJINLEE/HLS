# written for flow package Vivado 
set sdc_version 1.7 

create_clock -name clk -period 3.33 -waveform { 0.0 1.665 } [get_ports {clk}]
set_clock_uncertainty 0.0 [get_clocks {clk}]

create_clock -name virtual_io_clk -period 3.33
## IO TIMING CONSTRAINTS
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {rst}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {input_stream_rsc_dat[*]}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {input_stream_rsc_vld}]
set_output_delay -clock [get_clocks {clk}] 0.0 [get_ports {input_stream_rsc_rdy}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {weight_stream_rsc_dat[*]}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {weight_stream_rsc_vld}]
set_output_delay -clock [get_clocks {clk}] 0.0 [get_ports {weight_stream_rsc_rdy}]
set_output_delay -clock [get_clocks {clk}] 0.0 [get_ports {p3_stream_rsc_dat[*]}]
set_output_delay -clock [get_clocks {clk}] 0.0 [get_ports {p3_stream_rsc_vld}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {p3_stream_rsc_rdy}]
set_output_delay -clock [get_clocks {clk}] 0.0 [get_ports {p4_stream_rsc_dat[*]}]
set_output_delay -clock [get_clocks {clk}] 0.0 [get_ports {p4_stream_rsc_vld}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {p4_stream_rsc_rdy}]
set_output_delay -clock [get_clocks {clk}] 0.0 [get_ports {p5_stream_rsc_dat[*]}]
set_output_delay -clock [get_clocks {clk}] 0.0 [get_ports {p5_stream_rsc_vld}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {p5_stream_rsc_rdy}]

