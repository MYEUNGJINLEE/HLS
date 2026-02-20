#include <string>
#include <fstream>
#include <iostream>
#include "mc_testbench.h"
#include <mc_reset.h>
#include <mc_transactors.h>
#include <mc_scverify.h>
#include <mc_stall_ctrl.h>
#include <ac_read_env.h>
#include "ccs_ioport_trans_rsc_v1.h"
#include <mc_monitor.h>
#include <mc_simulator_extensions.h>
#include "mc_dut_wrapper.h"
#include "ccs_probes.h"
#include <mt19937ar.c>
#ifndef TO_QUOTED_STRING
#define TO_QUOTED_STRING(x) TO_QUOTED_STRING1(x)
#define TO_QUOTED_STRING1(x) #x
#endif
// Hold time for the SCVerify testbench to account for the gate delay after downstream synthesis in pico second(s)
// Hold time value is obtained from 'top_gate_constraints.cpp', which is generated at the end of RTL synthesis
#ifdef CCS_DUT_GATE
extern double __scv_hold_time;
extern double __scv_hold_time_RSCID_1;
extern double __scv_hold_time_RSCID_2;
extern double __scv_hold_time_RSCID_3;
extern double __scv_hold_time_RSCID_4;
extern double __scv_hold_time_RSCID_5;
#else
double __scv_hold_time = 0.0; // default for non-gate simulation is zero
double __scv_hold_time_RSCID_1 = 0;
double __scv_hold_time_RSCID_2 = 0;
double __scv_hold_time_RSCID_3 = 0;
double __scv_hold_time_RSCID_4 = 0;
double __scv_hold_time_RSCID_5 = 0;
#endif

class scverify_top : public sc_module
{
public:
  sc_signal<sc_logic>                                         rst;
  sc_signal<sc_logic>                                         rst_n;
  sc_signal<sc_logic>                                         SIG_SC_LOGIC_0;
  sc_signal<sc_logic>                                         SIG_SC_LOGIC_1;
  sc_signal<sc_logic>                                         TLS_design_is_idle;
  sc_signal<bool>                                             TLS_design_is_idle_reg;
  unsigned long                                               d_max_sim_time;
  unsigned long                                               d_deadlock_time;
  bool                                                        env_SCVerify_DEADLOCK_DETECTION;
  bool                                                        env_SCVerify_DISABLE_EMPTY_INPUTS;
  bool                                                        env_SCVerify_IDLE_SYNCHRONIZATION_MODE;
  bool                                                        env_SCVerify_ENABLE_RESET_TOGGLE;
  float                                                       env_SCVerify_RESET_CYCLES;
  unsigned long                                               env_SCVerify_MAX_SIM_TIME;
  bool                                                        env_SCVerify_RAND_SEED_USES_TIME;
  int                                                         env_SCVerify_AUTOWAIT;
  int                                                         env_SCVerify_AUTOWAIT_INPUT_CYCLES;
  int                                                         env_SCVerify_AUTOWAIT_OUTPUT_CYCLES;
  bool                                                        env_SCVerify_ENABLE_STALL_TOGGLE;
  int                                                         env_SCVerify_STALL_HOLD_COUNT;
  int                                                         d_idle_sync_auto_wait;
  bool                                                        d_iosync_pause_on_stall;
  bool                                                        d_idle_sync_enabled;
  bool                                                        d_disable_on_empty;
  sc_clock                                                    clk;
  mc_programmable_reset                                       rst_driver;
  sc_signal<sc_logic>                                         TLS_rst;
  sc_signal<sc_lv<512> >                                      TLS_input_stream_rsc_dat;
  sc_signal<sc_logic>                                         TLS_input_stream_rsc_vld;
  sc_signal<sc_logic>                                         TLS_input_stream_rsc_rdy;
  sc_signal<sc_lv<64> >                                       TLS_weight_stream_rsc_dat;
  sc_signal<sc_logic>                                         TLS_weight_stream_rsc_vld;
  sc_signal<sc_logic>                                         TLS_weight_stream_rsc_rdy;
  sc_signal<sc_lv<512> >                                      TLS_p3_stream_rsc_dat;
  sc_signal<sc_logic>                                         TLS_p3_stream_rsc_vld;
  sc_signal<sc_logic>                                         TLS_p3_stream_rsc_rdy;
  sc_signal<sc_lv<512> >                                      TLS_p4_stream_rsc_dat;
  sc_signal<sc_logic>                                         TLS_p4_stream_rsc_vld;
  sc_signal<sc_logic>                                         TLS_p4_stream_rsc_rdy;
  sc_signal<sc_lv<512> >                                      TLS_p5_stream_rsc_dat;
  sc_signal<sc_logic>                                         TLS_p5_stream_rsc_vld;
  sc_signal<sc_logic>                                         TLS_p5_stream_rsc_rdy;
  ccs_DUT_wrapper                                             GPTBackbonePhase3_INST;
  ccs_ctrl_in_buf_wait_trans_rsc_v1<1,1,512,1,0,0,1 >         input_stream_rsc_INST;
  ccs_ctrl_in_buf_wait_trans_rsc_v1<1,2,64,1,0,0,1 >          weight_stream_rsc_INST;
  ccs_out_buf_wait_trans_rsc_v1<1,3,512,1,0,0,1,0 >           p3_stream_rsc_INST;
  ccs_out_buf_wait_trans_rsc_v1<1,4,512,1,0,0,1,0 >           p4_stream_rsc_INST;
  ccs_out_buf_wait_trans_rsc_v1<1,5,512,1,0,0,1,0 >           p5_stream_rsc_INST;
  tlm::tlm_fifo<ac_int<512, false > >                         TLS_in_fifo_input_stream;
  tlm::tlm_fifo<mc_wait_ctrl>                                 TLS_in_wait_ctrl_fifo_input_stream;
  tlm::tlm_fifo<int>                                          TLS_in_fifo_input_stream_sizecount;
  sc_signal<sc_logic>                                         TLS_input_stream_rsc_trdone;
  mc_channel_input_transactor<ac_int<512, false >,512,false>  transactor_input_stream;
  tlm::tlm_fifo<ac_int<64, false > >                          TLS_in_fifo_weight_stream;
  tlm::tlm_fifo<mc_wait_ctrl>                                 TLS_in_wait_ctrl_fifo_weight_stream;
  tlm::tlm_fifo<int>                                          TLS_in_fifo_weight_stream_sizecount;
  sc_signal<sc_logic>                                         TLS_weight_stream_rsc_trdone;
  mc_channel_input_transactor<ac_int<64, false >,64,false>    transactor_weight_stream;
  tlm::tlm_fifo<ac_int<512, false > >                         TLS_out_fifo_p3_stream;
  tlm::tlm_fifo<mc_wait_ctrl>                                 TLS_out_wait_ctrl_fifo_p3_stream;
  sc_signal<sc_logic>                                         TLS_p3_stream_rsc_trdone;
  mc_output_transactor<ac_int<512, false >,512,false>         transactor_p3_stream;
  tlm::tlm_fifo<ac_int<512, false > >                         TLS_out_fifo_p4_stream;
  tlm::tlm_fifo<mc_wait_ctrl>                                 TLS_out_wait_ctrl_fifo_p4_stream;
  sc_signal<sc_logic>                                         TLS_p4_stream_rsc_trdone;
  mc_output_transactor<ac_int<512, false >,512,false>         transactor_p4_stream;
  tlm::tlm_fifo<ac_int<512, false > >                         TLS_out_fifo_p5_stream;
  tlm::tlm_fifo<mc_wait_ctrl>                                 TLS_out_wait_ctrl_fifo_p5_stream;
  sc_signal<sc_logic>                                         TLS_p5_stream_rsc_trdone;
  mc_output_transactor<ac_int<512, false >,512,false>         transactor_p5_stream;
  mc_testbench                                                testbench_INST;
  sc_signal<sc_logic>                                         catapult_start;
  sc_signal<sc_logic>                                         catapult_done;
  sc_signal<sc_logic>                                         catapult_ready;
  sc_signal<sc_logic>                                         in_sync;
  sc_signal<sc_logic>                                         out_sync;
  sc_signal<sc_logic>                                         inout_sync;
  sc_signal<unsigned>                                         wait_for_init;
  sync_generator                                              sync_generator_INST;
  catapult_monitor                                            catapult_monitor_INST;
  mc_wait_ctrl                                               *autowait_input_cfg;
  mc_wait_ctrl                                               *autowait_output_cfg;
  sc_event                                                    generate_reset_event;
  sc_event                                                    deadlock_event;
  sc_signal<sc_logic>                                         deadlocked;
  sc_event                                                    max_sim_time_event;
  sc_signal<sc_logic>                                         OFS_GPTBackbonePhase3_run_inst_GPTBackbonePhase3_run_staller_inst_run_wen;
  sc_signal<sc_logic>                                         OFS_input_stream_rsc_vld;
  sc_signal<sc_logic>                                         OFS_weight_stream_rsc_vld;
  sc_signal<sc_logic>                                         OFS_p3_stream_rsc_rdy;
  sc_signal<sc_logic>                                         OFS_p4_stream_rsc_rdy;
  sc_signal<sc_logic>                                         OFS_p5_stream_rsc_rdy;
  sc_signal<sc_logic>                                         OFS_GPTBackbonePhase3_run_inst_GPTBackbonePhase3_run_input_stream_rsci_inst_GPTBackbonePhase3_run_input_stream_rsci_input_stream_wait_ctrl_inst_input_stream_rsci_ivld_oreg;
  sc_signal<sc_logic>                                         OFS_GPTBackbonePhase3_run_inst_GPTBackbonePhase3_run_weight_stream_rsci_inst_GPTBackbonePhase3_run_weight_stream_rsci_weight_stream_wait_ctrl_inst_weight_stream_rsci_ivld_oreg;
  sc_signal<sc_logic>                                         OFS_GPTBackbonePhase3_run_inst_GPTBackbonePhase3_run_p3_stream_rsci_inst_GPTBackbonePhase3_run_p3_stream_rsci_p3_stream_wait_ctrl_inst_p3_stream_rsci_irdy_oreg;
  sc_signal<sc_logic>                                         OFS_GPTBackbonePhase3_run_inst_GPTBackbonePhase3_run_p4_stream_rsci_inst_GPTBackbonePhase3_run_p4_stream_rsci_p4_stream_wait_ctrl_inst_p4_stream_rsci_irdy_oreg;
  sc_signal<sc_logic>                                         OFS_GPTBackbonePhase3_run_inst_GPTBackbonePhase3_run_p5_stream_rsci_inst_GPTBackbonePhase3_run_p5_stream_rsci_p5_stream_wait_ctrl_inst_p5_stream_rsci_irdy_oreg;
  sc_signal<sc_logic>                                         TLS_enable_stalls;
  sc_signal<unsigned short>                                   TLS_stall_coverage;
  bool                                                        var_trdone;

  void read_env();
  void TLS_rst_method();
  void drive_TLS_input_stream_rsc_trdone();
  void drive_TLS_weight_stream_rsc_trdone();
  void drive_TLS_p3_stream_rsc_trdone();
  void drive_TLS_p4_stream_rsc_trdone();
  void drive_TLS_p5_stream_rsc_trdone();
  void max_sim_time_notify();
  void start_of_simulation();
  void setup_autowait();
  void inform_autowait();
  void manual_flush();
  void setup_debug();
  void debug(const char* varname, int flags, int count);
  void generate_reset();
  void install_observe_foreign_signals();
  void deadlock_watch();
  void deadlock_notify();
  void drive_idle_reg();
  void idle_watch();

  // Constructor
  SC_HAS_PROCESS(scverify_top);
  scverify_top(const sc_module_name& name)
    : rst("rst")
    , rst_n("rst_n")
    , SIG_SC_LOGIC_0("SIG_SC_LOGIC_0")
    , SIG_SC_LOGIC_1("SIG_SC_LOGIC_1")
    , TLS_design_is_idle("TLS_design_is_idle")
    , TLS_design_is_idle_reg("TLS_design_is_idle_reg")
    , CCS_CLK_CTOR(clk, "clk", 3.33, SC_NS, 0.5, 0, SC_NS, false)
    , rst_driver("rst_driver", ac_env::read_float("SCVerify_RESET_CYCLES",2.0)*3.330000, false)
    , TLS_rst("TLS_rst")
    , TLS_input_stream_rsc_dat("TLS_input_stream_rsc_dat")
    , TLS_input_stream_rsc_vld("TLS_input_stream_rsc_vld")
    , TLS_input_stream_rsc_rdy("TLS_input_stream_rsc_rdy")
    , TLS_weight_stream_rsc_dat("TLS_weight_stream_rsc_dat")
    , TLS_weight_stream_rsc_vld("TLS_weight_stream_rsc_vld")
    , TLS_weight_stream_rsc_rdy("TLS_weight_stream_rsc_rdy")
    , TLS_p3_stream_rsc_dat("TLS_p3_stream_rsc_dat")
    , TLS_p3_stream_rsc_vld("TLS_p3_stream_rsc_vld")
    , TLS_p3_stream_rsc_rdy("TLS_p3_stream_rsc_rdy")
    , TLS_p4_stream_rsc_dat("TLS_p4_stream_rsc_dat")
    , TLS_p4_stream_rsc_vld("TLS_p4_stream_rsc_vld")
    , TLS_p4_stream_rsc_rdy("TLS_p4_stream_rsc_rdy")
    , TLS_p5_stream_rsc_dat("TLS_p5_stream_rsc_dat")
    , TLS_p5_stream_rsc_vld("TLS_p5_stream_rsc_vld")
    , TLS_p5_stream_rsc_rdy("TLS_p5_stream_rsc_rdy")
    , GPTBackbonePhase3_INST("rtl", "ccs_wrapper")
    , input_stream_rsc_INST("input_stream_rsc", true)
    , weight_stream_rsc_INST("weight_stream_rsc", true)
    , p3_stream_rsc_INST("p3_stream_rsc", true)
    , p4_stream_rsc_INST("p4_stream_rsc", true)
    , p5_stream_rsc_INST("p5_stream_rsc", true)
    , TLS_in_fifo_input_stream("TLS_in_fifo_input_stream", -1)
    , TLS_in_wait_ctrl_fifo_input_stream("TLS_in_wait_ctrl_fifo_input_stream", -1)
    , TLS_in_fifo_input_stream_sizecount("TLS_in_fifo_input_stream_sizecount", 1)
    , TLS_input_stream_rsc_trdone("TLS_input_stream_rsc_trdone")
    , transactor_input_stream("transactor_input_stream", 0, 512, 0, 2, d_disable_on_empty)
    , TLS_in_fifo_weight_stream("TLS_in_fifo_weight_stream", -1)
    , TLS_in_wait_ctrl_fifo_weight_stream("TLS_in_wait_ctrl_fifo_weight_stream", -1)
    , TLS_in_fifo_weight_stream_sizecount("TLS_in_fifo_weight_stream_sizecount", 1)
    , TLS_weight_stream_rsc_trdone("TLS_weight_stream_rsc_trdone")
    , transactor_weight_stream("transactor_weight_stream", 0, 64, 0, 2, d_disable_on_empty)
    , TLS_out_fifo_p3_stream("TLS_out_fifo_p3_stream", -1)
    , TLS_out_wait_ctrl_fifo_p3_stream("TLS_out_wait_ctrl_fifo_p3_stream", -1)
    , TLS_p3_stream_rsc_trdone("TLS_p3_stream_rsc_trdone")
    , transactor_p3_stream("transactor_p3_stream", 0, 512, 0)
    , TLS_out_fifo_p4_stream("TLS_out_fifo_p4_stream", -1)
    , TLS_out_wait_ctrl_fifo_p4_stream("TLS_out_wait_ctrl_fifo_p4_stream", -1)
    , TLS_p4_stream_rsc_trdone("TLS_p4_stream_rsc_trdone")
    , transactor_p4_stream("transactor_p4_stream", 0, 512, 0)
    , TLS_out_fifo_p5_stream("TLS_out_fifo_p5_stream", -1)
    , TLS_out_wait_ctrl_fifo_p5_stream("TLS_out_wait_ctrl_fifo_p5_stream", -1)
    , TLS_p5_stream_rsc_trdone("TLS_p5_stream_rsc_trdone")
    , transactor_p5_stream("transactor_p5_stream", 0, 512, 0)
    , testbench_INST("user_tb")
    , catapult_start("catapult_start")
    , catapult_done("catapult_done")
    , catapult_ready("catapult_ready")
    , in_sync("in_sync")
    , out_sync("out_sync")
    , inout_sync("inout_sync")
    , wait_for_init("wait_for_init")
    , sync_generator_INST("sync_generator", true, false, false, false, 850075974, 850075974, 0)
    , catapult_monitor_INST("Monitor", clk, true, 850075974LL, 848724293LL)
    , autowait_input_cfg(NULL)
    , autowait_output_cfg(NULL)
    , deadlocked("deadlocked")
    , var_trdone(false)
  {
    read_env();

    rst_driver.reset_out(TLS_rst);

    GPTBackbonePhase3_INST.clk(clk);
    GPTBackbonePhase3_INST.rst(TLS_rst);
    GPTBackbonePhase3_INST.input_stream_rsc_dat(TLS_input_stream_rsc_dat);
    GPTBackbonePhase3_INST.input_stream_rsc_vld(TLS_input_stream_rsc_vld);
    GPTBackbonePhase3_INST.input_stream_rsc_rdy(TLS_input_stream_rsc_rdy);
    GPTBackbonePhase3_INST.weight_stream_rsc_dat(TLS_weight_stream_rsc_dat);
    GPTBackbonePhase3_INST.weight_stream_rsc_vld(TLS_weight_stream_rsc_vld);
    GPTBackbonePhase3_INST.weight_stream_rsc_rdy(TLS_weight_stream_rsc_rdy);
    GPTBackbonePhase3_INST.p3_stream_rsc_dat(TLS_p3_stream_rsc_dat);
    GPTBackbonePhase3_INST.p3_stream_rsc_vld(TLS_p3_stream_rsc_vld);
    GPTBackbonePhase3_INST.p3_stream_rsc_rdy(TLS_p3_stream_rsc_rdy);
    GPTBackbonePhase3_INST.p4_stream_rsc_dat(TLS_p4_stream_rsc_dat);
    GPTBackbonePhase3_INST.p4_stream_rsc_vld(TLS_p4_stream_rsc_vld);
    GPTBackbonePhase3_INST.p4_stream_rsc_rdy(TLS_p4_stream_rsc_rdy);
    GPTBackbonePhase3_INST.p5_stream_rsc_dat(TLS_p5_stream_rsc_dat);
    GPTBackbonePhase3_INST.p5_stream_rsc_vld(TLS_p5_stream_rsc_vld);
    GPTBackbonePhase3_INST.p5_stream_rsc_rdy(TLS_p5_stream_rsc_rdy);

    input_stream_rsc_INST.clk(clk);
    input_stream_rsc_INST.en(SIG_SC_LOGIC_0);
    input_stream_rsc_INST.arst(SIG_SC_LOGIC_1);
    input_stream_rsc_INST.srst(TLS_rst);
    input_stream_rsc_INST.rdy(TLS_input_stream_rsc_rdy);
    input_stream_rsc_INST.vld(TLS_input_stream_rsc_vld);
    input_stream_rsc_INST.dat(TLS_input_stream_rsc_dat);
    input_stream_rsc_INST.add_attribute(*(new sc_attribute<double>("CLK_SKEW_DELAY", __scv_hold_time_RSCID_1)));
    input_stream_rsc_INST.add_attribute(*(new sc_attribute<int>("TRANS_LATENCY", 510468)));

    weight_stream_rsc_INST.clk(clk);
    weight_stream_rsc_INST.en(SIG_SC_LOGIC_0);
    weight_stream_rsc_INST.arst(SIG_SC_LOGIC_1);
    weight_stream_rsc_INST.srst(TLS_rst);
    weight_stream_rsc_INST.rdy(TLS_weight_stream_rsc_rdy);
    weight_stream_rsc_INST.vld(TLS_weight_stream_rsc_vld);
    weight_stream_rsc_INST.dat(TLS_weight_stream_rsc_dat);
    weight_stream_rsc_INST.add_attribute(*(new sc_attribute<double>("CLK_SKEW_DELAY", __scv_hold_time_RSCID_2)));
    weight_stream_rsc_INST.add_attribute(*(new sc_attribute<int>("TRANS_LATENCY", 2)));

    p3_stream_rsc_INST.clk(clk);
    p3_stream_rsc_INST.en(SIG_SC_LOGIC_0);
    p3_stream_rsc_INST.arst(SIG_SC_LOGIC_1);
    p3_stream_rsc_INST.srst(TLS_rst);
    p3_stream_rsc_INST.rdy(TLS_p3_stream_rsc_rdy);
    p3_stream_rsc_INST.vld(TLS_p3_stream_rsc_vld);
    p3_stream_rsc_INST.dat(TLS_p3_stream_rsc_dat);
    p3_stream_rsc_INST.add_attribute(*(new sc_attribute<double>("CLK_SKEW_DELAY", __scv_hold_time_RSCID_3)));

    p4_stream_rsc_INST.clk(clk);
    p4_stream_rsc_INST.en(SIG_SC_LOGIC_0);
    p4_stream_rsc_INST.arst(SIG_SC_LOGIC_1);
    p4_stream_rsc_INST.srst(TLS_rst);
    p4_stream_rsc_INST.rdy(TLS_p4_stream_rsc_rdy);
    p4_stream_rsc_INST.vld(TLS_p4_stream_rsc_vld);
    p4_stream_rsc_INST.dat(TLS_p4_stream_rsc_dat);
    p4_stream_rsc_INST.add_attribute(*(new sc_attribute<double>("CLK_SKEW_DELAY", __scv_hold_time_RSCID_4)));

    p5_stream_rsc_INST.clk(clk);
    p5_stream_rsc_INST.en(SIG_SC_LOGIC_0);
    p5_stream_rsc_INST.arst(SIG_SC_LOGIC_1);
    p5_stream_rsc_INST.srst(TLS_rst);
    p5_stream_rsc_INST.rdy(TLS_p5_stream_rsc_rdy);
    p5_stream_rsc_INST.vld(TLS_p5_stream_rsc_vld);
    p5_stream_rsc_INST.dat(TLS_p5_stream_rsc_dat);
    p5_stream_rsc_INST.add_attribute(*(new sc_attribute<double>("CLK_SKEW_DELAY", __scv_hold_time_RSCID_5)));

    transactor_input_stream.in_fifo(TLS_in_fifo_input_stream);
    transactor_input_stream.in_wait_ctrl_fifo(TLS_in_wait_ctrl_fifo_input_stream);
    transactor_input_stream.sizecount_fifo(TLS_in_fifo_input_stream_sizecount);
    transactor_input_stream.set_disable_on_empty(d_disable_on_empty);
    transactor_input_stream.bind_clk(clk, true, rst);
    transactor_input_stream.add_attribute(*(new sc_attribute<int>("MC_TRANSACTOR_EVENT", 0 )));
    transactor_input_stream.register_block(&input_stream_rsc_INST, input_stream_rsc_INST.basename(), TLS_input_stream_rsc_trdone, 0, 0, 1);

    transactor_weight_stream.in_fifo(TLS_in_fifo_weight_stream);
    transactor_weight_stream.in_wait_ctrl_fifo(TLS_in_wait_ctrl_fifo_weight_stream);
    transactor_weight_stream.sizecount_fifo(TLS_in_fifo_weight_stream_sizecount);
    transactor_weight_stream.set_disable_on_empty(d_disable_on_empty);
    transactor_weight_stream.bind_clk(clk, true, rst);
    transactor_weight_stream.add_attribute(*(new sc_attribute<int>("MC_TRANSACTOR_EVENT", 0 )));
    transactor_weight_stream.register_block(&weight_stream_rsc_INST, weight_stream_rsc_INST.basename(), TLS_weight_stream_rsc_trdone, 0, 0, 1);

    transactor_p3_stream.out_fifo(TLS_out_fifo_p3_stream);
    transactor_p3_stream.out_wait_ctrl_fifo(TLS_out_wait_ctrl_fifo_p3_stream);
    transactor_p3_stream.bind_clk(clk, true, rst);
    transactor_p3_stream.add_attribute(*(new sc_attribute<int>("MC_TRANSACTOR_EVENT", 0 )));
    transactor_p3_stream.register_block(&p3_stream_rsc_INST, p3_stream_rsc_INST.basename(), TLS_p3_stream_rsc_trdone, 0, 0, 1);

    transactor_p4_stream.out_fifo(TLS_out_fifo_p4_stream);
    transactor_p4_stream.out_wait_ctrl_fifo(TLS_out_wait_ctrl_fifo_p4_stream);
    transactor_p4_stream.bind_clk(clk, true, rst);
    transactor_p4_stream.add_attribute(*(new sc_attribute<int>("MC_TRANSACTOR_EVENT", 0 )));
    transactor_p4_stream.register_block(&p4_stream_rsc_INST, p4_stream_rsc_INST.basename(), TLS_p4_stream_rsc_trdone, 0, 0, 1);

    transactor_p5_stream.out_fifo(TLS_out_fifo_p5_stream);
    transactor_p5_stream.out_wait_ctrl_fifo(TLS_out_wait_ctrl_fifo_p5_stream);
    transactor_p5_stream.bind_clk(clk, true, rst);
    transactor_p5_stream.add_attribute(*(new sc_attribute<int>("MC_TRANSACTOR_EVENT", 0 )));
    transactor_p5_stream.register_block(&p5_stream_rsc_INST, p5_stream_rsc_INST.basename(), TLS_p5_stream_rsc_trdone, 0, 0, 1);

    testbench_INST.clk(clk);
    testbench_INST.ccs_input_stream(TLS_in_fifo_input_stream);
    testbench_INST.ccs_wait_ctrl_input_stream(TLS_in_wait_ctrl_fifo_input_stream);
    testbench_INST.ccs_sizecount_input_stream(TLS_in_fifo_input_stream_sizecount);
    testbench_INST.ccs_weight_stream(TLS_in_fifo_weight_stream);
    testbench_INST.ccs_wait_ctrl_weight_stream(TLS_in_wait_ctrl_fifo_weight_stream);
    testbench_INST.ccs_sizecount_weight_stream(TLS_in_fifo_weight_stream_sizecount);
    testbench_INST.ccs_p3_stream(TLS_out_fifo_p3_stream);
    testbench_INST.ccs_wait_ctrl_p3_stream(TLS_out_wait_ctrl_fifo_p3_stream);
    testbench_INST.ccs_p4_stream(TLS_out_fifo_p4_stream);
    testbench_INST.ccs_wait_ctrl_p4_stream(TLS_out_wait_ctrl_fifo_p4_stream);
    testbench_INST.ccs_p5_stream(TLS_out_fifo_p5_stream);
    testbench_INST.ccs_wait_ctrl_p5_stream(TLS_out_wait_ctrl_fifo_p5_stream);
    testbench_INST.design_is_idle(TLS_design_is_idle_reg);
    testbench_INST.enable_stalls(TLS_enable_stalls);
    testbench_INST.stall_coverage(TLS_stall_coverage);

    sync_generator_INST.clk(clk);
    sync_generator_INST.rst(rst);
    sync_generator_INST.in_sync(in_sync);
    sync_generator_INST.out_sync(out_sync);
    sync_generator_INST.inout_sync(inout_sync);
    sync_generator_INST.wait_for_init(wait_for_init);
    sync_generator_INST.catapult_start(catapult_start);
    sync_generator_INST.catapult_ready(catapult_ready);
    sync_generator_INST.catapult_done(catapult_done);

    catapult_monitor_INST.rst(rst);


    SC_METHOD(TLS_rst_method);
      sensitive_pos << TLS_rst;
      dont_initialize();

    SC_METHOD(drive_TLS_input_stream_rsc_trdone);
      sensitive << TLS_input_stream_rsc_rdy;
      sensitive << TLS_input_stream_rsc_vld;
      sensitive << rst;

    SC_METHOD(drive_TLS_weight_stream_rsc_trdone);
      sensitive << TLS_weight_stream_rsc_rdy;
      sensitive << TLS_weight_stream_rsc_vld;
      sensitive << rst;

    SC_METHOD(drive_TLS_p3_stream_rsc_trdone);
      sensitive << TLS_p3_stream_rsc_vld;
      sensitive << TLS_p3_stream_rsc_rdy;

    SC_METHOD(drive_TLS_p4_stream_rsc_trdone);
      sensitive << TLS_p4_stream_rsc_vld;
      sensitive << TLS_p4_stream_rsc_rdy;

    SC_METHOD(drive_TLS_p5_stream_rsc_trdone);
      sensitive << TLS_p5_stream_rsc_vld;
      sensitive << TLS_p5_stream_rsc_rdy;

    SC_METHOD(max_sim_time_notify);
      sensitive << max_sim_time_event;
      dont_initialize();

    SC_METHOD(inform_autowait);
      sensitive << testbench_INST.testbench_aw_event;
      dont_initialize();

    SC_METHOD(manual_flush);
      sensitive << testbench_INST.manual_flush_event;
      dont_initialize();

    SC_METHOD(generate_reset);
      sensitive << generate_reset_event;
      sensitive << testbench_INST.reset_request_event;

    SC_METHOD(deadlock_watch);
      sensitive << clk;
      sensitive << OFS_GPTBackbonePhase3_run_inst_GPTBackbonePhase3_run_staller_inst_run_wen;
      dont_initialize();

    SC_METHOD(deadlock_notify);
      sensitive << deadlock_event;
      dont_initialize();

    SC_METHOD(drive_idle_reg);
      sensitive << clk;

    SC_METHOD(idle_watch);
      sensitive << OFS_GPTBackbonePhase3_run_inst_GPTBackbonePhase3_run_staller_inst_run_wen;
      dont_initialize();


    SIG_SC_LOGIC_0.write(SC_LOGIC_0);
    SIG_SC_LOGIC_1.write(SC_LOGIC_1);
    if (env_SCVerify_RAND_SEED_USES_TIME) {;
    std::ostringstream msg;;
    msg << "SCVerify option RAND_SEED_USES_TIME in use. Random stall intervals will not be consistent from run to run.";
    SC_REPORT_WARNING("System", msg.str().c_str());;
    mt19937_init_genrand((int)time(NULL));
    } else {;
    mt19937_init_genrand(19650218UL);
    };
    install_observe_foreign_signals();
    deadlocked.write(SC_LOGIC_0);
    TLS_design_is_idle.write(SC_LOGIC_0);
  }
  ~scverify_top() {
  }
};
void scverify_top::read_env() {
  env_SCVerify_DEADLOCK_DETECTION = ac_env::read_bool("SCVerify_DEADLOCK_DETECTION",false);
  env_SCVerify_DISABLE_EMPTY_INPUTS = ac_env::read_bool("SCVerify_DISABLE_EMPTY_INPUTS",false);
  env_SCVerify_IDLE_SYNCHRONIZATION_MODE = ac_env::read_bool("SCVerify_IDLE_SYNCHRONIZATION_MODE",false);
  env_SCVerify_ENABLE_RESET_TOGGLE = ac_env::read_bool("SCVerify_ENABLE_RESET_TOGGLE",false);
  env_SCVerify_RESET_CYCLES = ac_env::read_float("SCVerify_RESET_CYCLES",2.0);
  env_SCVerify_MAX_SIM_TIME = ac_env::read_int("SCVerify_MAX_SIM_TIME",0);
  env_SCVerify_RAND_SEED_USES_TIME = ac_env::read_bool("SCVerify_RAND_SEED_USES_TIME", false);
  env_SCVerify_AUTOWAIT = ac_env::read_int("SCVerify_AUTOWAIT",/*default to 0 (no automatic I/O stalling)*/ 0);
  env_SCVerify_AUTOWAIT_INPUT_CYCLES = ac_env::read_int("SCVerify_AUTOWAIT_INPUT_CYCLES",/*original hardcoded default*/ 5);
  env_SCVerify_AUTOWAIT_OUTPUT_CYCLES = ac_env::read_int("SCVerify_AUTOWAIT_OUTPUT_CYCLES",/*original hardcoded default*/ 5);
  env_SCVerify_ENABLE_STALL_TOGGLE = ac_env::read_bool("SCVerify_ENABLE_STALL_TOGGLE", true);
  env_SCVerify_STALL_HOLD_COUNT = ac_env::read_int("SCVerify_STALL_HOLD_COUNT", 5);
  if (env_SCVerify_AUTOWAIT != 0 && env_SCVerify_AUTOWAIT_INPUT_CYCLES > d_idle_sync_auto_wait)  {d_idle_sync_auto_wait = env_SCVerify_AUTOWAIT_INPUT_CYCLES;}
  if (env_SCVerify_AUTOWAIT != 0 && env_SCVerify_AUTOWAIT_OUTPUT_CYCLES > d_idle_sync_auto_wait) {d_idle_sync_auto_wait = env_SCVerify_AUTOWAIT_OUTPUT_CYCLES;}
  d_idle_sync_auto_wait = d_idle_sync_auto_wait * 2;
  d_iosync_pause_on_stall = ac_env::read_bool("SCVerify_IOSYNC_PAUSE_ON_STALL",false);
  d_idle_sync_enabled = env_SCVerify_IDLE_SYNCHRONIZATION_MODE;
  d_disable_on_empty = env_SCVerify_DISABLE_EMPTY_INPUTS || env_SCVerify_IDLE_SYNCHRONIZATION_MODE;
  float longest_clk_per = 3.330000;
  unsigned long latency_est = 848724290;
  unsigned long trcnt = 1;
  if (env_SCVerify_ENABLE_RESET_TOGGLE) { trcnt = ceil(env_SCVerify_RESET_CYCLES+0.5)*3; }
  unsigned long wait_for_cycles = latency_est * 9 + trcnt;
  wait_for_cycles += 100; //extended for Xilinx
  unsigned long wait_time = ceil(longest_clk_per) * wait_for_cycles;
  d_deadlock_time = wait_time;
  d_max_sim_time = env_SCVerify_MAX_SIM_TIME;
  if ((d_max_sim_time > 0) && (wait_time > d_max_sim_time)) {
    d_max_sim_time = wait_time;
    std::ostringstream msg;
    msg << "Maximum simulation time extended to meet calculated value of " << d_max_sim_time << " ns";
    SC_REPORT_WARNING("System", msg.str().c_str());
  }
}

void scverify_top::TLS_rst_method() {
  std::ostringstream msg;
  msg << "TLS_rst active @ " << sc_time_stamp();
  SC_REPORT_INFO("HW reset", msg.str().c_str());
  input_stream_rsc_INST.clear();
  weight_stream_rsc_INST.clear();
  p3_stream_rsc_INST.clear();
  p4_stream_rsc_INST.clear();
  p5_stream_rsc_INST.clear();
}

void scverify_top::drive_TLS_input_stream_rsc_trdone() {
  if (!env_SCVerify_ENABLE_RESET_TOGGLE && rst.read() == SC_LOGIC_1) {
    assert(TLS_input_stream_rsc_rdy.read() != SC_LOGIC_1);
  }
  TLS_input_stream_rsc_trdone.write(TLS_input_stream_rsc_rdy.read() & TLS_input_stream_rsc_vld.read() & ~rst.read());
}

void scverify_top::drive_TLS_weight_stream_rsc_trdone() {
  if (!env_SCVerify_ENABLE_RESET_TOGGLE && rst.read() == SC_LOGIC_1) {
    assert(TLS_weight_stream_rsc_rdy.read() != SC_LOGIC_1);
  }
  TLS_weight_stream_rsc_trdone.write(TLS_weight_stream_rsc_rdy.read() & TLS_weight_stream_rsc_vld.read() & ~rst.read());
}

void scverify_top::drive_TLS_p3_stream_rsc_trdone() {
  TLS_p3_stream_rsc_trdone.write(TLS_p3_stream_rsc_vld.read() & TLS_p3_stream_rsc_rdy.read());
}

void scverify_top::drive_TLS_p4_stream_rsc_trdone() {
  TLS_p4_stream_rsc_trdone.write(TLS_p4_stream_rsc_vld.read() & TLS_p4_stream_rsc_rdy.read());
}

void scverify_top::drive_TLS_p5_stream_rsc_trdone() {
  TLS_p5_stream_rsc_trdone.write(TLS_p5_stream_rsc_vld.read() & TLS_p5_stream_rsc_rdy.read());
}

void scverify_top::max_sim_time_notify() {
  testbench_INST.set_failed(true);
  testbench_INST.check_results();
  SC_REPORT_ERROR("System", "Specified maximum simulation time reached");
  sc_stop();
}

void scverify_top::start_of_simulation() {
  setup_autowait();
  if (d_max_sim_time>0) max_sim_time_event.notify(d_max_sim_time,SC_NS);
}

void scverify_top::setup_autowait() {
  autowait_input_cfg = new mc_wait_ctrl(0, 1, env_SCVerify_AUTOWAIT_INPUT_CYCLES, mc_wait_ctrl::RANDOM, mc_wait_ctrl::INITIAL, 0, true);
  autowait_output_cfg = new mc_wait_ctrl(0, 1, env_SCVerify_AUTOWAIT_OUTPUT_CYCLES, mc_wait_ctrl::RANDOM, mc_wait_ctrl::INITIAL, 0, true);
  transactor_input_stream.set_auto_wait_limit(env_SCVerify_AUTOWAIT);
  transactor_input_stream.configure_autowait(autowait_input_cfg);
  transactor_weight_stream.set_auto_wait_limit(env_SCVerify_AUTOWAIT);
  transactor_weight_stream.configure_autowait(autowait_input_cfg);
  transactor_p3_stream.set_auto_wait_limit(env_SCVerify_AUTOWAIT);
  transactor_p3_stream.configure_autowait(autowait_output_cfg);
  transactor_p4_stream.set_auto_wait_limit(env_SCVerify_AUTOWAIT);
  transactor_p4_stream.configure_autowait(autowait_output_cfg);
  transactor_p5_stream.set_auto_wait_limit(env_SCVerify_AUTOWAIT);
  transactor_p5_stream.configure_autowait(autowait_output_cfg);
}

void scverify_top::inform_autowait() {
  bool waited = false;
  waited |= (transactor_input_stream.atleast_one_autowait(0) > 0);
  waited |= (transactor_weight_stream.atleast_one_autowait(0) > 0);
  waited |= (transactor_p3_stream.atleast_one_autowait(1) > 0);
  waited |= (transactor_p4_stream.atleast_one_autowait(1) > 0);
  waited |= (transactor_p5_stream.atleast_one_autowait(1) > 0);
  if (waited)
    SC_REPORT_INFO(name(), "At least one AUTOWAIT was applied during simulation.");
}

void scverify_top::manual_flush() {
  static int flush_wait_events = 0;
  static bool message_once = true;
  static bool done_waiting = false;
  if ( !done_waiting && (0
    || transactor_input_stream.is_waiting(0)
    || transactor_weight_stream.is_waiting(0)
    || transactor_p3_stream.is_waiting(1)
    || transactor_p4_stream.is_waiting(1)
    || transactor_p5_stream.is_waiting(1)
  )) return;
  done_waiting = true;
  if ( testbench_INST.manual_flush_output_active )
    flush_wait_events = 0;
  if ( ++flush_wait_events > testbench::manual_flush_wait_cycles ) {
    bool input_reenabled = false;
    if ( transactor_input_stream.flush_disabled_input() ) {
      std::cout << "Re-enabled input 'input_stream' for manual flush mode @ " << sc_time_stamp() << std::endl;
      input_reenabled = true;
    }
    if ( transactor_weight_stream.flush_disabled_input() ) {
      std::cout << "Re-enabled input 'weight_stream' for manual flush mode @ " << sc_time_stamp() << std::endl;
      input_reenabled = true;
    }
    if ( input_reenabled && message_once ) {
      SC_REPORT_WARNING(name(), "Activating manual flush mode for remaining outputs due to disabled empty inputs.");
      testbench_INST.manual_flush_enabled = true;
      message_once = false;
    }
  }
}

void scverify_top::setup_debug() {
#ifdef MC_DEFAULT_TRANSACTOR_LOG
  static int transactor_input_stream_flags = MC_DEFAULT_TRANSACTOR_LOG;
  static int transactor_weight_stream_flags = MC_DEFAULT_TRANSACTOR_LOG;
  static int transactor_p3_stream_flags = MC_DEFAULT_TRANSACTOR_LOG;
  static int transactor_p4_stream_flags = MC_DEFAULT_TRANSACTOR_LOG;
  static int transactor_p5_stream_flags = MC_DEFAULT_TRANSACTOR_LOG;
#else
  static int transactor_input_stream_flags = (d_disable_on_empty ? 0 : MC_TRANSACTOR_UNDERFLOW) | MC_TRANSACTOR_WAIT;
  static int transactor_weight_stream_flags = (d_disable_on_empty ? 0 : MC_TRANSACTOR_UNDERFLOW) | MC_TRANSACTOR_WAIT;
  static int transactor_p3_stream_flags = MC_TRANSACTOR_UNDERFLOW | MC_TRANSACTOR_WAIT;
  static int transactor_p4_stream_flags = MC_TRANSACTOR_UNDERFLOW | MC_TRANSACTOR_WAIT;
  static int transactor_p5_stream_flags = MC_TRANSACTOR_UNDERFLOW | MC_TRANSACTOR_WAIT;
#endif
  static int transactor_input_stream_count = -1;
  static int transactor_weight_stream_count = -1;
  static int transactor_p3_stream_count = -1;
  static int transactor_p4_stream_count = -1;
  static int transactor_p5_stream_count = -1;

  // At the breakpoint, modify the local variables
  // above to turn on/off different levels of transaction
  // logging for each variable. Available flags are:
  //   MC_TRANSACTOR_EMPTY       - log empty FIFOs (on by default)
  //   MC_TRANSACTOR_UNDERFLOW   - log FIFOs that run empty and then are loaded again (off)
  //   MC_TRANSACTOR_READ        - log all read events
  //   MC_TRANSACTOR_WRITE       - log all write events
  //   MC_TRANSACTOR_LOAD        - log all FIFO load events
  //   MC_TRANSACTOR_DUMP        - log all FIFO dump events
  //   MC_TRANSACTOR_STREAMCNT   - log all streamed port index counter events
  //   MC_TRANSACTOR_WAIT        - log user specified handshake waits
  //   MC_TRANSACTOR_SIZE        - log input FIFO size updates

  std::ifstream debug_cmds;
  debug_cmds.open("scverify.cmd",std::fstream::in);
  if (debug_cmds.is_open()) {
    std::cout << "Reading SCVerify debug commands from file 'scverify.cmd'" << std::endl;
    std::string line;
    while (getline(debug_cmds,line)) {
      std::size_t pos1 = line.find(" ");
      if (pos1 == std::string::npos) continue;
      std::size_t pos2 = line.find(" ", pos1+1);
      std::string varname = line.substr(0,pos1);
      std::string flags = line.substr(pos1+1,pos2-pos1-1);
      std::string count = line.substr(pos2+1);
      debug(varname.c_str(),std::atoi(flags.c_str()),std::atoi(count.c_str()));
    }
    debug_cmds.close();
  } else {
    debug("transactor_input_stream",transactor_input_stream_flags,transactor_input_stream_count);
    debug("transactor_weight_stream",transactor_weight_stream_flags,transactor_weight_stream_count);
    debug("transactor_p3_stream",transactor_p3_stream_flags,transactor_p3_stream_count);
    debug("transactor_p4_stream",transactor_p4_stream_flags,transactor_p4_stream_count);
    debug("transactor_p5_stream",transactor_p5_stream_flags,transactor_p5_stream_count);
  }
}

void scverify_top::debug(const char* varname, int flags, int count) {
  sc_module *xlator_p = 0;
  sc_attr_base *debug_attr_p = 0;
  if (strcmp(varname,"transactor_input_stream") == 0)
    xlator_p = &transactor_input_stream;
  if (strcmp(varname,"transactor_weight_stream") == 0)
    xlator_p = &transactor_weight_stream;
  if (strcmp(varname,"transactor_p3_stream") == 0)
    xlator_p = &transactor_p3_stream;
  if (strcmp(varname,"transactor_p4_stream") == 0)
    xlator_p = &transactor_p4_stream;
  if (strcmp(varname,"transactor_p5_stream") == 0)
    xlator_p = &transactor_p5_stream;
  if (xlator_p) {
    debug_attr_p = xlator_p->get_attribute("MC_TRANSACTOR_EVENT");
    if (!debug_attr_p) {
      debug_attr_p = new sc_attribute<int>("MC_TRANSACTOR_EVENT",flags);
      xlator_p->add_attribute(*debug_attr_p);
    }
    ((sc_attribute<int>*)debug_attr_p)->value = flags;
  }

  if (count>=0) {
    debug_attr_p = xlator_p->get_attribute("MC_TRANSACTOR_COUNT");
    if (!debug_attr_p) {
      debug_attr_p = new sc_attribute<int>("MC_TRANSACTOR_COUNT",count);
      xlator_p->add_attribute(*debug_attr_p);
    }
    ((sc_attribute<int>*)debug_attr_p)->value = count;
  }
}

// Process: SC_METHOD generate_reset
void scverify_top::generate_reset() {
  static bool activate_reset = true;
  static int toggle_hw_reset = env_SCVerify_ENABLE_RESET_TOGGLE;
  static int toggle_rst = toggle_hw_reset << 1;
  if (activate_reset || sc_time_stamp() == SC_ZERO_TIME) {
    setup_debug();
    activate_reset = false;
    rst.write(SC_LOGIC_1);
    if (toggle_rst) {
      rst_driver.reset_driver();
      toggle_rst--;
    }
    else {
      rst_driver.reset_driver();
    }
    toggle_hw_reset = toggle_rst;
    generate_reset_event.notify(env_SCVerify_RESET_CYCLES*3.330000, SC_NS);
  } else {
    if (toggle_hw_reset) {
      generate_reset_event.notify(env_SCVerify_RESET_CYCLES*3.330000, SC_NS);
    } else {
      transactor_input_stream.reset_streams();
      transactor_weight_stream.reset_streams();
      transactor_p3_stream.reset_streams();
      transactor_p4_stream.reset_streams();
      transactor_p5_stream.reset_streams();
      rst.write(SC_LOGIC_0);
    }
    activate_reset = true;
  }
}

void scverify_top::install_observe_foreign_signals() {
#if !defined(CCS_DUT_SYSC)
#if defined(CCS_DUT_CYCLE) || defined(CCS_DUT_RTL)
  OBSERVE_FOREIGN_SIGNAL(OFS_GPTBackbonePhase3_run_inst_GPTBackbonePhase3_run_staller_inst_run_wen, /scverify_top/rtl/dut_inst/GPTBackbonePhase3_run_inst/GPTBackbonePhase3_run_staller_inst/run_wen);
  OBSERVE_FOREIGN_SIGNAL(OFS_input_stream_rsc_vld, /scverify_top/rtl/dut_inst/input_stream_rsc_vld);
  OBSERVE_FOREIGN_SIGNAL(OFS_weight_stream_rsc_vld, /scverify_top/rtl/dut_inst/weight_stream_rsc_vld);
  OBSERVE_FOREIGN_SIGNAL(OFS_p3_stream_rsc_rdy, /scverify_top/rtl/dut_inst/p3_stream_rsc_rdy);
  OBSERVE_FOREIGN_SIGNAL(OFS_p4_stream_rsc_rdy, /scverify_top/rtl/dut_inst/p4_stream_rsc_rdy);
  OBSERVE_FOREIGN_SIGNAL(OFS_p5_stream_rsc_rdy, /scverify_top/rtl/dut_inst/p5_stream_rsc_rdy);
  OBSERVE_FOREIGN_SIGNAL(OFS_GPTBackbonePhase3_run_inst_GPTBackbonePhase3_run_input_stream_rsci_inst_GPTBackbonePhase3_run_input_stream_rsci_input_stream_wait_ctrl_inst_input_stream_rsci_ivld_oreg,
      /scverify_top/rtl/dut_inst/GPTBackbonePhase3_run_inst/GPTBackbonePhase3_run_input_stream_rsci_inst/GPTBackbonePhase3_run_input_stream_rsci_input_stream_wait_ctrl_inst/input_stream_rsci_ivld_oreg);
  OBSERVE_FOREIGN_SIGNAL(OFS_GPTBackbonePhase3_run_inst_GPTBackbonePhase3_run_weight_stream_rsci_inst_GPTBackbonePhase3_run_weight_stream_rsci_weight_stream_wait_ctrl_inst_weight_stream_rsci_ivld_oreg,
      /scverify_top/rtl/dut_inst/GPTBackbonePhase3_run_inst/GPTBackbonePhase3_run_weight_stream_rsci_inst/GPTBackbonePhase3_run_weight_stream_rsci_weight_stream_wait_ctrl_inst/weight_stream_rsci_ivld_oreg);
  OBSERVE_FOREIGN_SIGNAL(OFS_GPTBackbonePhase3_run_inst_GPTBackbonePhase3_run_p3_stream_rsci_inst_GPTBackbonePhase3_run_p3_stream_rsci_p3_stream_wait_ctrl_inst_p3_stream_rsci_irdy_oreg,
      /scverify_top/rtl/dut_inst/GPTBackbonePhase3_run_inst/GPTBackbonePhase3_run_p3_stream_rsci_inst/GPTBackbonePhase3_run_p3_stream_rsci_p3_stream_wait_ctrl_inst/p3_stream_rsci_irdy_oreg);
  OBSERVE_FOREIGN_SIGNAL(OFS_GPTBackbonePhase3_run_inst_GPTBackbonePhase3_run_p4_stream_rsci_inst_GPTBackbonePhase3_run_p4_stream_rsci_p4_stream_wait_ctrl_inst_p4_stream_rsci_irdy_oreg,
      /scverify_top/rtl/dut_inst/GPTBackbonePhase3_run_inst/GPTBackbonePhase3_run_p4_stream_rsci_inst/GPTBackbonePhase3_run_p4_stream_rsci_p4_stream_wait_ctrl_inst/p4_stream_rsci_irdy_oreg);
  OBSERVE_FOREIGN_SIGNAL(OFS_GPTBackbonePhase3_run_inst_GPTBackbonePhase3_run_p5_stream_rsci_inst_GPTBackbonePhase3_run_p5_stream_rsci_p5_stream_wait_ctrl_inst_p5_stream_rsci_irdy_oreg,
      /scverify_top/rtl/dut_inst/GPTBackbonePhase3_run_inst/GPTBackbonePhase3_run_p5_stream_rsci_inst/GPTBackbonePhase3_run_p5_stream_rsci_p5_stream_wait_ctrl_inst/p5_stream_rsci_irdy_oreg);
#endif
#endif
}

void scverify_top::deadlock_watch() {
#if !defined(CCS_DUT_SYSC) && defined(DEADLOCK_DETECTION)
#if defined(CCS_DUT_CYCLE) || defined(CCS_DUT_RTL)
#if defined(MTI_SYSTEMC) || defined(NCSC) || defined(VCS_SYSTEMC)
  if (!clk) {
    if (rst == SC_LOGIC_0 &&
      (OFS_GPTBackbonePhase3_run_inst_GPTBackbonePhase3_run_staller_inst_run_wen.read() == SC_LOGIC_0)
      && (OFS_input_stream_rsc_vld.read() == SC_LOGIC_1)
      && (OFS_weight_stream_rsc_vld.read() == SC_LOGIC_1)
      && (OFS_p3_stream_rsc_rdy.read() == SC_LOGIC_1)
      && (OFS_p4_stream_rsc_rdy.read() == SC_LOGIC_1)
      && (OFS_p5_stream_rsc_rdy.read() == SC_LOGIC_1)
    ) {
      deadlocked.write(SC_LOGIC_1);
      deadlock_event.notify(d_deadlock_time, SC_NS);
    } else {
      if (deadlocked.read() == SC_LOGIC_1)
        deadlock_event.cancel();
      deadlocked.write(SC_LOGIC_0);
    }
  }
#endif
#endif
#endif
}

void scverify_top::deadlock_notify() {
  if (deadlocked.read() == SC_LOGIC_1) {
    testbench_INST.check_results();
    SC_REPORT_ERROR("System", "Simulation deadlock detected");
    sc_stop();
  }
}

void scverify_top::drive_idle_reg() {
#if defined(MTI_SYSTEMC) || defined(NCSC) || defined(VCS_SYSTEMC)
  static unsigned short wait_for_idle_buf;
  if (clk.read() == SC_LOGIC_1 && rst.read() == SC_LOGIC_0) {
    if (TLS_design_is_idle.read() == SC_LOGIC_1) {
      wait_for_idle_buf++;
    } else {
      wait_for_idle_buf = 0;
    }
    int stable_cycles = testbench::idle_sync_stable_cycles;
    if (stable_cycles < d_idle_sync_auto_wait) { stable_cycles = d_idle_sync_auto_wait; }
    if (wait_for_idle_buf >= stable_cycles) {
      TLS_design_is_idle_reg.write(true);
      if (var_trdone) var_trdone = false;
      wait_for_idle_buf = 0;
    } else {
      TLS_design_is_idle_reg.write(false);
    }
  }
#endif
}

void scverify_top::idle_watch() {
#if defined(MTI_SYSTEMC) || defined(NCSC) || defined(VCS_SYSTEMC)
  bool X = (OFS_GPTBackbonePhase3_run_inst_GPTBackbonePhase3_run_staller_inst_run_wen.read()==0);
  bool Y = false;
  var_trdone = X || Y;
  TLS_design_is_idle.write(var_trdone ? SC_LOGIC_1 : SC_LOGIC_0);
#endif
}

#if defined(MC_SIMULATOR_OSCI) || defined(MC_SIMULATOR_VCS)
int sc_main(int argc, char *argv[]) {
  sc_report_handler::set_actions("/IEEE_Std_1666/deprecated", SC_DO_NOTHING);
  scverify_top scverify_top("scverify_top");
  sc_start();
  return scverify_top.testbench_INST.failed();
}
#else
MC_MODULE_EXPORT(scverify_top);
#endif
