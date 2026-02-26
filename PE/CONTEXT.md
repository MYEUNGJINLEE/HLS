# PE Context Handoff

This file is a quick handoff context for other AI agents working on `PE/`.

## 1) Project Purpose

- Standalone Catapult HLS project for a 4-PE parallel block.
- Initial focus is PE functionality (not full YOLO pipeline).
- Top module is `PEParallelBlock`.

Top interface:

```cpp
bool PEParallelBlock::run(
  const PEBlockCfg &cfg,
  ac_channel<pe_packed_act_t> &input_stream,
  ac_channel<pe_weight_pkt_t> &weight_stream,
  ac_channel<pe_packed_act_t> &output_stream);
```

## 2) Core Data Types and Formats

- Activation lane: `pe_act_t = ac_int<8, true>`
- Accumulator: `pe_acc_t = ac_int<20, true>`
- Packed activation: `pe_packed_act_t = ac_int<512, false>` (`64 x int8`)
- Weight packet: `pe_weight_pkt_t = ac_int<128, false>`

Packet contract:
- Header: `[127:124] kind`, `[123:120] op`, `[119:112] oc_idx`, `[111:104] ic_group`, `[103:96] aux`
- Payload: `[95:0]`

Kinds:
- `0 = WEIGHT`, `1 = BN`, `2 = END`

Ops:
- `0 = conv1x1`, `1 = conv3x3`, `2 = dw3x3`

## 3) Config Model

Primary config is `PEBlockCfg` in `include/pe_types.h`.

Main fields:
- `topo`: `PE_TOPO_STRAIGHT` or `PE_TOPO_SPLITCAT`
- `pe0..pe3`: per-kernel config
- split controls: `split_num`, `split_den`, `use_input_split`
- post-store route:
  - `post_route`: `PE_POST_DIRECT`, `PE_POST_BRANCH`, `PE_POST_SPLIT`, `PE_POST_CONCAT`
  - `post_split_a_ch`: valid only for `PE_POST_SPLIT`

Validation entry:
- `pe_validate_block_cfg(...)` in `include/pe_config.h`

## 4) Runtime Flow Contract

Current flow in `PEParallelBlock::run`:

1. Validate cfg
2. Fetch all required weight packets from `weight_stream`
3. Validate packet layout
4. Compute
5. Store output by `post_route`

### Topology behavior

- `PE_TOPO_STRAIGHT`: run only `pe0`
- `PE_TOPO_SPLITCAT`:
  - input fork or split
  - path A: `pe0 -> pe1`
  - path B: `pe2`
  - concat A/B
  - default: feed to `pe3`
  - if `post_route == PE_POST_CONCAT`: bypass `pe3`, store concat result directly

### Post-store route behavior

- `PE_POST_DIRECT`: output base tensor as-is
- `PE_POST_BRANCH`: duplicate output stream twice (A then B)
- `PE_POST_SPLIT`: split by channel and serialize split-A then split-B
- `PE_POST_CONCAT`: only valid in SplitCat; output concat tensor before `pe3`

Important: when `post_route == PE_POST_CONCAT`, `pe3` weights are not fetched.

## 5) Key Files

- `include/pe_types.h`: core constants/types/enums/helpers
- `include/pe_packets.h`: 128b packet encode/decode
- `include/pe_config.h`: cfg constructors + validators
- `src/pe_unit.h/.cpp`: single PE compute kernels
- `src/pe_parallel_block.h/.cpp`: 4-PE orchestration, routing
- `tb/pe_reference_model.h`: scalar reference model
- `tb/pe_parallel_block_tb.cpp`: deterministic + random + fail tests
- `scripts/run_pe_catapult.tcl`: Catapult flow script
- `Makefile`: local build/compile/gui aliases

## 6) Build and Run Commands

Inside `PE/`:

- Full flow: `make pe`
- Analyze only: `make pe-analyze`
- Compile only: `make pe-compile`
- Compile aliases: `make compile`, `make comple`, `make COMPILE`, `make COMPLE`
- GUI: `make pe-gui` or `make gui`
- TB: `make pe-tb`
- Clean: `make clean` (also `make pe-clean`)

GUI behavior:
- `make gui` resolves `.ccs`, normalizes to `pe.ccs`, then runs:
  - `catapult -f pe.ccs &`

From repo root (proxy targets exist):
- `make pe`, `make pe-log`, `make pe-tb`, `make pe-clean`, `make pe-gui`

## 7) Catapult Notes / Previous Error Context

Past failures seen in this project included:
- `HIER-6`: local non-static `ac_channel` usage
- `HIER-7`: same channel read by both parent block and child hierarchical block
- `HIER-10`: channels that do not cross hierarchy boundary
- `HIER-11`: treating channel as both input and output

Current code structure was adjusted to avoid those patterns:
- Separate channel for pe3 input path (`g_cat_for_pe3`)
- Post-split output path uses memory buffer (`g_split_b_buf`) instead of internal temp channels

## 8) Testbench Status

TB includes:
- deterministic cases
- random stress (200 fixed-seed cases)
- fail-path cases

Extra deterministic post-route checks included:
- straight + post-branch
- straight + post-split
- splitcat + post-concat

## 9) Recent PE Commit Trail

- `216d176` - gui/clean alias updates in `PE/Makefile`
- `9261651` - hierarchy channel fixes in post-route flow
- `f39ef0a` - post-route store flow + compile aliases

## 10) Practical Handoff Note

If compile errors reappear, inspect first:
- `src/pe_parallel_block.cpp` (channel topology)
- `include/pe_types.h` and `include/pe_config.h` (weight count + cfg validity)
- `tb/pe_reference_model.h` (model parity with RTL/HLS behavior)
