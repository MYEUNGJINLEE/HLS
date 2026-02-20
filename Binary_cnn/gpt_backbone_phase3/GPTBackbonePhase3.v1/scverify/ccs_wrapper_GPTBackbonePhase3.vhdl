-- SCVerify DUT wrapper used for SystemC > HDL interface bindings

LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;

ENTITY ccs_wrapper IS
  PORT(
    clk : IN STD_LOGIC;
    rst : IN STD_LOGIC;
    input_stream_rsc_dat : IN STD_LOGIC_VECTOR(511 DOWNTO 0);
    input_stream_rsc_vld : IN STD_LOGIC;
    input_stream_rsc_rdy : OUT STD_LOGIC;
    weight_stream_rsc_dat : IN STD_LOGIC_VECTOR(63 DOWNTO 0);
    weight_stream_rsc_vld : IN STD_LOGIC;
    weight_stream_rsc_rdy : OUT STD_LOGIC;
    p3_stream_rsc_dat : OUT STD_LOGIC_VECTOR(511 DOWNTO 0);
    p3_stream_rsc_vld : OUT STD_LOGIC;
    p3_stream_rsc_rdy : IN STD_LOGIC;
    p4_stream_rsc_dat : OUT STD_LOGIC_VECTOR(511 DOWNTO 0);
    p4_stream_rsc_vld : OUT STD_LOGIC;
    p4_stream_rsc_rdy : IN STD_LOGIC;
    p5_stream_rsc_dat : OUT STD_LOGIC_VECTOR(511 DOWNTO 0);
    p5_stream_rsc_vld : OUT STD_LOGIC;
    p5_stream_rsc_rdy : IN STD_LOGIC
  );
END ccs_wrapper;

ARCHITECTURE wrap OF ccs_wrapper IS
  COMPONENT GPTBackbonePhase3
    PORT (
      clk : IN STD_LOGIC;
      rst : IN STD_LOGIC;
      input_stream_rsc_dat : IN STD_LOGIC_VECTOR(511 DOWNTO 0);
      input_stream_rsc_vld : IN STD_LOGIC;
      input_stream_rsc_rdy : OUT STD_LOGIC;
      weight_stream_rsc_dat : IN STD_LOGIC_VECTOR(63 DOWNTO 0);
      weight_stream_rsc_vld : IN STD_LOGIC;
      weight_stream_rsc_rdy : OUT STD_LOGIC;
      p3_stream_rsc_dat : OUT STD_LOGIC_VECTOR(511 DOWNTO 0);
      p3_stream_rsc_vld : OUT STD_LOGIC;
      p3_stream_rsc_rdy : IN STD_LOGIC;
      p4_stream_rsc_dat : OUT STD_LOGIC_VECTOR(511 DOWNTO 0);
      p4_stream_rsc_vld : OUT STD_LOGIC;
      p4_stream_rsc_rdy : IN STD_LOGIC;
      p5_stream_rsc_dat : OUT STD_LOGIC_VECTOR(511 DOWNTO 0);
      p5_stream_rsc_vld : OUT STD_LOGIC;
      p5_stream_rsc_rdy : IN STD_LOGIC
    );
  END COMPONENT;

BEGIN

  dut_inst : GPTBackbonePhase3
    PORT MAP (
      clk => clk,
      rst => rst,
      input_stream_rsc_dat => input_stream_rsc_dat,
      input_stream_rsc_vld => input_stream_rsc_vld,
      input_stream_rsc_rdy => input_stream_rsc_rdy,
      weight_stream_rsc_dat => weight_stream_rsc_dat,
      weight_stream_rsc_vld => weight_stream_rsc_vld,
      weight_stream_rsc_rdy => weight_stream_rsc_rdy,
      p3_stream_rsc_dat => p3_stream_rsc_dat,
      p3_stream_rsc_vld => p3_stream_rsc_vld,
      p3_stream_rsc_rdy => p3_stream_rsc_rdy,
      p4_stream_rsc_dat => p4_stream_rsc_dat,
      p4_stream_rsc_vld => p4_stream_rsc_vld,
      p4_stream_rsc_rdy => p4_stream_rsc_rdy,
      p5_stream_rsc_dat => p5_stream_rsc_dat,
      p5_stream_rsc_vld => p5_stream_rsc_vld,
      p5_stream_rsc_rdy => p5_stream_rsc_rdy
    );

END wrap;

