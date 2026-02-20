// SCVerify DUT wrapper used for SystemC > HDL interface bindings


module ccs_wrapper (
  clk, rst, input_stream_rsc_dat, input_stream_rsc_vld, input_stream_rsc_rdy, weight_stream_rsc_dat, weight_stream_rsc_vld,
      weight_stream_rsc_rdy, p3_stream_rsc_dat, p3_stream_rsc_vld, p3_stream_rsc_rdy, p4_stream_rsc_dat, p4_stream_rsc_vld,
      p4_stream_rsc_rdy, p5_stream_rsc_dat, p5_stream_rsc_vld, p5_stream_rsc_rdy
);
  input clk;
  input rst;
  input [511:0] input_stream_rsc_dat;
  input input_stream_rsc_vld;
  output input_stream_rsc_rdy;
  input [63:0] weight_stream_rsc_dat;
  input weight_stream_rsc_vld;
  output weight_stream_rsc_rdy;
  output [511:0] p3_stream_rsc_dat;
  output p3_stream_rsc_vld;
  input p3_stream_rsc_rdy;
  output [511:0] p4_stream_rsc_dat;
  output p4_stream_rsc_vld;
  input p4_stream_rsc_rdy;
  output [511:0] p5_stream_rsc_dat;
  output p5_stream_rsc_vld;
  input p5_stream_rsc_rdy;


  GPTBackbonePhase3 dut_inst (
    .clk(clk),
    .rst(rst),
    .input_stream_rsc_dat(input_stream_rsc_dat),
    .input_stream_rsc_vld(input_stream_rsc_vld),
    .input_stream_rsc_rdy(input_stream_rsc_rdy),
    .weight_stream_rsc_dat(weight_stream_rsc_dat),
    .weight_stream_rsc_vld(weight_stream_rsc_vld),
    .weight_stream_rsc_rdy(weight_stream_rsc_rdy),
    .p3_stream_rsc_dat(p3_stream_rsc_dat),
    .p3_stream_rsc_vld(p3_stream_rsc_vld),
    .p3_stream_rsc_rdy(p3_stream_rsc_rdy),
    .p4_stream_rsc_dat(p4_stream_rsc_dat),
    .p4_stream_rsc_vld(p4_stream_rsc_vld),
    .p4_stream_rsc_rdy(p4_stream_rsc_rdy),
    .p5_stream_rsc_dat(p5_stream_rsc_dat),
    .p5_stream_rsc_vld(p5_stream_rsc_vld),
    .p5_stream_rsc_rdy(p5_stream_rsc_rdy)
  );

endmodule

