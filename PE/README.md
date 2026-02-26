# PE Project (Standalone)

This project is a standalone 4-PE parallel block for hardware-first YOLO bring-up.

## Scope (v1)

- Topology:
  - `PE_TOPO_STRAIGHT`
  - `PE_TOPO_SPLITCAT`
- Post-store route:
  - `PE_POST_DIRECT`
  - `PE_POST_BRANCH` (duplicate output stream A/B)
  - `PE_POST_SPLIT` (serialize split-A then split-B)
  - `PE_POST_CONCAT` (SplitCat only, bypass `pe3` and store concat tensor)
- Ops:
  - `PE_OP_CONV1X1`
  - `PE_OP_CONV3X3`
  - `PE_OP_DW3X3`
- Activation format:
  - `pe_packed_act_t` = `ac_int<512,false>` (`64ch x int8`)
- Weight packet format:
  - `pe_weight_pkt_t` = `ac_int<128,false>` (unified header + payload)

## Directory

- `include/pe_types.h`: core types/constants/cfg structs
- `include/pe_packets.h`: 128b packet encode/decode helpers
- `include/pe_config.h`: cfg constructors + validators
- `src/pe_unit.*`: single PE implementation
- `src/pe_parallel_block.*`: 4-PE block top (`run(cfg, in, w, out)`)
- `tb/pe_reference_model.h`: scalar reference model
- `tb/pe_parallel_block_tb.cpp`: deterministic + random + failure tests
- `scripts/run_pe_catapult.tcl`: Catapult batch script

## 128b Packet Contract

Header bits:

- `[127:124] kind` (`0=WEIGHT`, `1=BN`, `2=END`)
- `[123:120] op` (`0=conv1x1`, `1=conv3x3`, `2=dw3x3`)
- `[119:112] oc_idx`
- `[111:104] ic_group` (conv1x1/conv3x3)
- `[103:96] aux` (v1 must be `0`)
- `[95:0] payload`

Payload:

- `WEIGHT+conv1x1`: `payload[63:0]` = 64 binary weights
- `WEIGHT+conv3x3`: `payload[71:0]` = `8ch x 9 kernel bits`
- `WEIGHT+dw3x3`: `payload[8:0]` = 9 kernel bits
- `BN`: `payload[7:0]=shift`, `payload[23:8]=bias`

## Layer Execution Order

- `cfg validate`
- `weight fetch`
- `compute`
- `store(route by cfg.post_route)`

## Build / Run

From `PE` directory:

```bash
make pe
make pe-tb
make pe-gui
make pe-clean
```

From repo root:

```bash
make pe
make pe-tb
```

## Test Coverage

`tb/pe_parallel_block_tb.cpp` includes:

- Deterministic tests:
  - Straight + conv1x1
  - Straight + post-branch
  - Straight + post-split
  - Straight + conv3x3
  - Straight + dw3x3
  - SplitCat chain (A1/A2/B1/CCAT)
  - SplitCat + post-concat (bypass pe3)
- Random stress:
  - 200 fixed-seed cases
  - includes both Straight and SplitCat
  - channels in `{64,128,192,256}`
- Failure paths:
  - invalid split ratio
  - invalid conv1x1 pad
  - weight shortage
  - weight overflow

## Notes

- `Shortcut/Upsample/MaxPool` are intentionally out of scope for v1.
- Channel counts are validated as multiples of 64.
- BN path is `Shift + Bias + ReLU + int8 clamp`.
- `PE_POST_CONCAT` mode consumes only `pe0/pe1/pe2` weights (no `pe3` weight fetch).
