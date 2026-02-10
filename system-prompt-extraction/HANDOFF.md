# Stem SCHD Stabilization Handoff

## 1) Goal (what must be finished)
- Keep stem logic **bit-accurate** while making Catapult schedule pass under aggressive timing.
- Remove/avoid recurring `SCHD-3`, `SCHD-22`, and `could not schedule partition /StemProcessor/run`.
- Preserve current public interface (`StemProcessor::run` signature, stream formats, test input/output shape).
- Keep terminal test flow usable (`make stem-log`, `make stem-tb-*`) and verify TB PASS.

## 2) Current baseline (already pushed)
- Branch: `dev`
- Latest pushed commit: `503955b`
- Commit message: `stem: scheduling stabilization via pragma fix, pipeline split, and Tcl mapping update`
- Files changed in that commit:
  - `Binary_cnn/src/stem/stem_config.h`
  - `Binary_cnn/src/stem/stem_buffer.h`
  - `Binary_cnn/src/stem/stem_processor.cpp`
  - `Binary_cnn/scripts/run_stem_catapult.tcl`

## 3) What was implemented

### A. Pragma validity fix (Catapult parsing)
- Added literal `_Pragma` expansion in `Binary_cnn/src/stem/stem_config.h`:
  - `STEM_UNROLL_PRAGMA` for factors `1/2/4/8`.
- Replaced old macro-form pragmas in:
  - `Binary_cnn/src/stem/stem_buffer.h`
  - `Binary_cnn/src/stem/stem_processor.cpp`
- Purpose:
  - Avoid historical warning: `Invalid pragma 'hls_unroll<factor=STEM_UNROLL_FACTOR>'`.

### B. `run()` dependency weakening (stage decoupling)
- In `Binary_cnn/src/stem/stem_processor.cpp`, introduced row-token staging:
  - `conv2_row_stage[STEM_MAX_WIDTH][32]`
  - `mp_row_stage[STEM_MAX_WIDTH][32]`
  - validity/index flags to sync producer/consumer rows.
- Stage policy:
  - Stage4 writes Conv2 row into staging buffer.
  - Stage5 writes MP row into staging buffer.
  - Stage6 runs Conv3 only when both staged rows for the same output row are ready.
- Intent:
  - Reduce same-iteration producer/consumer feedback pressure.

### C. Arithmetic split on hot loops
- Split per-OC loops into two phases for Conv0/1/2/3:
  - `*_ACC_OC` (accumulate only)
  - `*_POST_OC` (BN + ReLU + clamp + cast)
- Intent:
  - Shorten critical feedback path seen by scheduler.

### D. Tcl flow robustness
- In `Binary_cnn/scripts/run_stem_catapult.tcl`:
  - Added `map_buffer_resource` procedure.
  - Dynamic root-based mapping attempts for:
    - `line_buf_a.buffer`, `line_buf_b.buffer`
    - `conv1_buf.buffer`, `mp_buf.buffer`
    - `concat_buf.path_a`, `concat_buf.path_b`
  - Added mapping summary logs (`MAP OK/MAP FAIL`).
  - Added explicit `directive set -CLOCK_OVERHEAD 0`.
  - Added post-compile report archiving to `Binary_cnn/logs/reports`.

## 4) What worked vs. what failed historically

### Worked
- Manual SCVerify run via Catapult flow command can pass in terminal when environment/project path is valid:
  - PASS marker: `*** TEST PASSED ***`
  - Example observed output included `Input: 640x640x3`, `Output: 160x160x32`, and matching output count.
- `stem_processor_tb.cpp` casting issues were corrected to `.to_double()` style (no raw cast of `ac_fixed`).

### Failed / fragile points seen before
- Direct SCVerify `make` in generated dir can fail due missing env (`CXX_HOME`, `SYSTEMC_INCDIR`, `ccs_env.mk`).
- Running Catapult from wrong working context caused `flow working directory is not valid`.
- GUI invocation confusion:
  - `catapult -shell -file ...` is batch mode.
  - GUI project open should be `catapult <project.ccs>` (not `-file`).
- Server-side git frequently got blocked by merge conflicts in `Makefile` (`needs merge`, `missing separator`) when pull/merge interrupted.

## 5) Validation status right now
- Local repo status at handoff creation: clean after push of `503955b`.
- Catapult runtime validation was **not** run in this local Codex environment (no direct server GUI/session control here).
- The next agent must run server validation commands below.

## 6) Next-agent runbook (do this first)

1. Sync server working tree cleanly.
```bash
cd /mnt/HDD/soc_lab_LMJ/catapultwork/git/HLS
git checkout dev
git fetch origin
git reset --hard origin/dev
```

2. Run compile/schedule log flow.
```bash
make stem-log
```

3. Check schedule blockers quickly.
```bash
rg -n "SCHD-3|SCHD-22|could not schedule|Feedback path is too long|Invalid pragma" Binary_cnn/logs/catapult_stem.log
```

4. Run TB without files.
```bash
make stem-tb-no-weight
```

5. Run TB with files (if available).
```bash
make stem-tb-weight
```

6. Confirm pass markers.
```bash
rg -n "\*\*\* TEST PASSED \*\*\*|Value mismatches" Binary_cnn/logs/scverify_stem_sim.log
```

## 7) If SCHD still fails: prioritized follow-up

1. Identify exact failing loop names/lines in latest `catapult_stem.log`.
2. Apply loop-local relief only where failing:
   - Slightly relax `hls_pipeline_init_interval` on specific `*_ACC_OC` loop(s) only.
   - Keep post loops separate.
3. If failures tie to concat/mp extraction paths:
   - Add one-cycle decoupling registers between extraction and accumulation for that stage.
4. Keep bit-accuracy guard:
   - After each change, run `make stem-tb-no-weight` and confirm mismatches remain 0.

## 8) Acceptance criteria to close task
- `make stem-log` completes without `SCHD-3/22` and without invalid pragma warnings.
- `make stem-tb-no-weight` and `make stem-tb-weight` both show `*** TEST PASSED ***`.
- `Value mismatches: 0` for golden comparison.
- No interface changes to stem public I/O.
