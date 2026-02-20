// mc_dut_wrapper.h
#ifndef INCLUDED_CCS_DUT_WRAPPER_H
#define INCLUDED_CCS_DUT_WRAPPER_H

#ifndef SC_USE_STD_STRING
#define SC_USE_STD_STRING
#endif

#include <systemc.h>
#include <mc_simulator_extensions.h>

#ifdef CCS_SYSC
namespace HDL {
#endif

// Create a foreign module wrapper around the HDL
#ifdef VCS_SYSTEMC
// VCS support - ccs_DUT_wrapper is derived from VCS-generated SystemC wrapper around HDL code
class ccs_DUT_wrapper : public ccs_wrapper
{
public:
  ccs_DUT_wrapper(const sc_module_name& nm, const char *hdl_name)
  : ccs_wrapper(nm)
  {}
  ~ccs_DUT_wrapper() {}
};

#else
// non VCS simulators - ccs_DUT_wrapper is derived from mc_foreign_module (adding 2nd ctor arg)
class ccs_DUT_wrapper: public mc_foreign_module
{
public:
  // Interface Ports
  sc_in<bool> clk;
  sc_in<sc_logic> rst;
  sc_in<sc_lv<512> > input_stream_rsc_dat;
  sc_in<sc_logic> input_stream_rsc_vld;
  sc_out<sc_logic> input_stream_rsc_rdy;
  sc_in<sc_lv<64> > weight_stream_rsc_dat;
  sc_in<sc_logic> weight_stream_rsc_vld;
  sc_out<sc_logic> weight_stream_rsc_rdy;
  sc_out<sc_lv<512> > p3_stream_rsc_dat;
  sc_out<sc_logic> p3_stream_rsc_vld;
  sc_in<sc_logic> p3_stream_rsc_rdy;
  sc_out<sc_lv<512> > p4_stream_rsc_dat;
  sc_out<sc_logic> p4_stream_rsc_vld;
  sc_in<sc_logic> p4_stream_rsc_rdy;
  sc_out<sc_lv<512> > p5_stream_rsc_dat;
  sc_out<sc_logic> p5_stream_rsc_vld;
  sc_in<sc_logic> p5_stream_rsc_rdy;
public:
  ccs_DUT_wrapper(const sc_module_name& nm, const char *hdl_name)
  :
    mc_foreign_module(nm, hdl_name), 
    clk("clk"), 
    rst("rst"), 
    input_stream_rsc_dat("input_stream_rsc_dat"), 
    input_stream_rsc_vld("input_stream_rsc_vld"), 
    input_stream_rsc_rdy("input_stream_rsc_rdy"), 
    weight_stream_rsc_dat("weight_stream_rsc_dat"), 
    weight_stream_rsc_vld("weight_stream_rsc_vld"), 
    weight_stream_rsc_rdy("weight_stream_rsc_rdy"), 
    p3_stream_rsc_dat("p3_stream_rsc_dat"), 
    p3_stream_rsc_vld("p3_stream_rsc_vld"), 
    p3_stream_rsc_rdy("p3_stream_rsc_rdy"), 
    p4_stream_rsc_dat("p4_stream_rsc_dat"), 
    p4_stream_rsc_vld("p4_stream_rsc_vld"), 
    p4_stream_rsc_rdy("p4_stream_rsc_rdy"), 
    p5_stream_rsc_dat("p5_stream_rsc_dat"), 
    p5_stream_rsc_vld("p5_stream_rsc_vld"), 
    p5_stream_rsc_rdy("p5_stream_rsc_rdy")
  {
    elaborate_foreign_module(hdl_name);
  }

  ~ccs_DUT_wrapper() {}
};
#endif // VCS_SYSTEMC

#ifdef CCS_SYSC
} // namespace HDL
#endif
#endif // INCLUDED_CCS_DUT_WRAPPER_H


