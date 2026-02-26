# Catapult batch script for BackboneBlock1 (DS + C3 Bottleneck)
# Run example:
#   cd Binary_cnn
#   catapult -shell -file scripts/run_backbone_catapult.tcl

project new -name backbone_block1

solution new -state initial
solution options defaults
solution options set /Output/GenerateCycleNetlist false

# Design + testbench registration
solution file add ./src/backbone/backbone_block.cpp -type C++
solution file add ./src/backbone/backbone_block_tb.cpp -type C++ -exclude true

go analyze

# ============================================================================
# Hierarchy debug
# ============================================================================
puts "======== DESIGN HIERARCHY DEBUG ========"
catch {puts "solution design get: [solution design get]"}
catch {puts "directive get -rec: [directive get -rec]"}
puts "========================================"

# ============================================================================
# Helper procs (same as stem TCL)
# ============================================================================

proc map_buffer_resource {label keys roots} {
    foreach root $roots {
        foreach key $keys {
            set path "${root}/${key}:rsc"
            if {![catch {directive set $path -MAP_TO_MODULE {BLOCK_1R1W_RBW}} err]} {
                puts "MAP OK  : ${label} -> ${path}"
                return 1
            }
        }
    }
    puts "MAP FAIL: ${label}"
    return 0
}

proc apply_array_partition {label keys roots dim factor} {
    foreach root $roots {
        foreach key $keys {
            set path "${root}/${key}:rsc"
            if {![catch {directive set $path -ARRAY_PARTITION cyclic -dim $dim -factor $factor} err]} {
                puts "PARTITION OK  : ${label} -> ${path}"
                return 1
            }
        }
    }
    puts "PARTITION FAIL: ${label}"
    return 0
}

proc apply_complete_partition {label keys roots dim} {
    foreach root $roots {
        foreach key $keys {
            set path "${root}/${key}:rsc"
            if {![catch {directive set $path -ARRAY_PARTITION complete -dim $dim} err]} {
                puts "COMPLETE PARTITION OK: ${label} -> ${path}"
                return 1
            }
        }
    }
    puts "COMPLETE PARTITION SKIP: ${label}"
    return 0
}

proc map_to_register {label keys roots} {
    foreach root $roots {
        foreach key $keys {
            set path "${root}/${key}:rsc"
            if {![catch {directive set $path -MAP_TO_MODULE {[Register]}} err]} {
                puts "REG OK  : ${label} -> ${path}"
                return 1
            }
        }
    }
    puts "REG SKIP: ${label}"
    return 0
}

proc set_loop_dependence {label keys roots} {
    foreach root $roots {
        foreach key $keys {
            set path "${root}/${key}"
            if {![catch {directive set $path -DEPENDENCE_TYPE inter -DEPENDENCE false} err]} {
                puts "DEPENDENCE OK  : ${label} -> ${path}"
                return 1
            }
        }
    }
    puts "DEPENDENCE SKIP: ${label}"
    return 0
}

# ============================================================================
# Build root paths for BackboneBlock1 hierarchy
# ============================================================================

set roots {
    /BackboneBlock1
    /BackboneBlock1/run
    /backbone_block1/BackboneBlock1
    /backbone_block1/BackboneBlock1/run
}

# Collect from solution design
set design_names {}
catch {set design_names [solution design get]}
foreach d $design_names {
    if {[string first "::" $d] >= 0} { continue }
    lappend roots "/$d"
    lappend roots "/$d/run"
}
set roots [lsort -unique $roots]
puts "BackboneBlock1 roots: $roots"

# ============================================================================
# Global Directives
# ============================================================================

# Clock: 300 MHz = 3.333 ns
if {[catch {directive set -CLOCK_PERIOD 3.333} err]} {
    puts "CLOCK_PERIOD set failed: $err"
} else {
    puts "CLOCK_PERIOD set to 3.333 ns (300 MHz)"
}

if {[catch {directive set -CLOCK_OVERHEAD 0} err]} {
    puts "CLOCK_OVERHEAD set failed: $err"
} else {
    puts "CLOCK_OVERHEAD set to 0"
}

# MEM_MAP_THRESHOLD: arrays <= N bits → registers; > N bits → BRAM
# Backbone weight arrays (w_ds: 64*32*9=18432b, w_c3a2: 32*32*9=9216b) → BRAM
# Small arrays (w_c3a1: 32*64=2048b, shifts, biases) → registers
if {[catch {directive set -MEM_MAP_THRESHOLD 8192} err]} {
    puts "MEM_MAP_THRESHOLD set failed: $err"
} else {
    puts "MEM_MAP_THRESHOLD set to 8192"
}

# ============================================================================
# Memory Mapping: Line Buffers → BLOCK_1R1W_RBW
# ============================================================================
puts "======== MEMORY MAPPING ========"

set map_total 0
set map_ok 0

# ds_input_buf: 8×160×32 = 40KB → BRAM
incr map_total
if {[map_buffer_resource "ds_input_buf" {
    ds_input_buf
    run/ds_input_buf
    MAIN_LOOP_BB1:for/ds_input_buf
} $roots]} { incr map_ok }

# c3a2_input_buf: 8×80×32 = 20KB → BRAM
incr map_total
if {[map_buffer_resource "c3a2_input_buf" {
    c3a2_input_buf
    run/c3a2_input_buf
    MAIN_LOOP_BB1:for/c3a2_input_buf
} $roots]} { incr map_ok }

# ds_save_buf: 2×80×64 = 10KB → BRAM
incr map_total
if {[map_buffer_resource "ds_save_buf" {
    ds_save_buf
    run/ds_save_buf
    MAIN_LOOP_BB1:for/ds_save_buf
} $roots]} { incr map_ok }

puts "Memory mapping: ${map_ok}/${map_total}"

# ============================================================================
# Array Partitioning
# ============================================================================
# Sliding window pattern: ds_win / c3a2_win are class member registers.
# ds_input_buf / c3a2_input_buf are now 2D packed (256-bit per col), no ch dim.
# BRAM reads: 1 sequential read per KR row per column step → no port conflict.
# No dim=3 partition needed (channel dim removed by packing).
# ds_win/c3a2_win: 8×3×32×8=6144 bits < MEM_MAP_THRESHOLD → auto registers.
puts "======== ARRAY PARTITIONING ========"

set part_total 0
set part_ok 0

# ds_save_buf: complete partition on channel dim (dim=3) for parallel preload
incr part_total
if {[apply_complete_partition "ds_save_buf ch-dim" {
    ds_save_buf run/ds_save_buf
} $roots 3]} { incr part_ok }

# ds_window local: complete partition on channel dim for unrolled access
incr part_total
if {[apply_complete_partition "ds_window dim3" {
    MAIN_LOOP_BB1:for:STAGE2_COL:for/ds_window
    run/MAIN_LOOP_BB1:for:STAGE2_COL:for/ds_window
} $roots 3]} { incr part_ok }

# c3a2_window local: complete partition on channel dim
incr part_total
if {[apply_complete_partition "c3a2_window dim3" {
    MAIN_LOOP_BB1:for:STAGE3_COL:for/c3a2_window
    run/MAIN_LOOP_BB1:for:STAGE3_COL:for/c3a2_window
} $roots 3]} { incr part_ok }

puts "Array partition: ${part_ok}/${part_total}"

# ============================================================================
# Register Mapping: Local compute arrays → flip-flops (not BRAM)
# ============================================================================
puts "======== REGISTER MAPPINGS ========"

# Stage 2 (DS + C3A1) local arrays
map_to_register "DS partial acc" {
    MAIN_LOOP_BB1:for:STAGE2_COL:for/acc_ds
    run/MAIN_LOOP_BB1:for:STAGE2_COL:for/acc_ds
    MAIN_LOOP_BB1:for/acc_ds
} $roots

map_to_register "DS output" {
    MAIN_LOOP_BB1:for:STAGE2_COL:for/ds_out
    run/MAIN_LOOP_BB1:for:STAGE2_COL:for/ds_out
} $roots

map_to_register "C3A1 partial acc" {
    MAIN_LOOP_BB1:for:STAGE2_COL:for/acc_c3a1
    run/MAIN_LOOP_BB1:for:STAGE2_COL:for/acc_c3a1
} $roots

# Stage 3 (C3A2 + C3B1 + C3CAT) local arrays
map_to_register "DS local (for C3B1)" {
    MAIN_LOOP_BB1:for:STAGE3_COL:for/ds_local
    run/MAIN_LOOP_BB1:for:STAGE3_COL:for/ds_local
} $roots

map_to_register "C3A2 partial acc" {
    MAIN_LOOP_BB1:for:STAGE3_COL:for/acc_c3a2
    run/MAIN_LOOP_BB1:for:STAGE3_COL:for/acc_c3a2
} $roots

map_to_register "C3A2 output" {
    MAIN_LOOP_BB1:for:STAGE3_COL:for/c3a2_out
    run/MAIN_LOOP_BB1:for:STAGE3_COL:for/c3a2_out
} $roots

map_to_register "C3B1 partial acc" {
    MAIN_LOOP_BB1:for:STAGE3_COL:for/acc_c3b1
    run/MAIN_LOOP_BB1:for:STAGE3_COL:for/acc_c3b1
} $roots

map_to_register "C3B1 output" {
    MAIN_LOOP_BB1:for:STAGE3_COL:for/c3b1_out
    run/MAIN_LOOP_BB1:for:STAGE3_COL:for/c3b1_out
} $roots

map_to_register "C3CAT partial acc" {
    MAIN_LOOP_BB1:for:STAGE3_COL:for/acc_ccat
    run/MAIN_LOOP_BB1:for:STAGE3_COL:for/acc_ccat
} $roots

map_to_register "final output" {
    MAIN_LOOP_BB1:for:STAGE3_COL:for/final_out
    run/MAIN_LOOP_BB1:for:STAGE3_COL:for/final_out
} $roots

# Sliding window register arrays (class members, 6144 bits each → auto-registers)
# Explicit mapping to prevent Catapult from mapping them to BRAM
map_to_register "ds_win" {
    ds_win run/ds_win
} $roots

map_to_register "c3a2_win" {
    c3a2_win run/c3a2_win
} $roots

# Weight arrays (small ones → registers, large → BRAM via MEM_MAP_THRESHOLD)
map_to_register "shift arrays" {
    run/shift_ds shift_ds
    run/shift_c3a1 shift_c3a1
    run/shift_c3a2 shift_c3a2
    run/shift_c3b1 shift_c3b1
    run/shift_ccat shift_ccat
} $roots

map_to_register "bias arrays" {
    run/bias_ds bias_ds
    run/bias_c3a1 bias_c3a1
    run/bias_c3a2 bias_c3a2
    run/bias_c3b1 bias_c3b1
    run/bias_ccat bias_ccat
} $roots

puts "======== REGISTER MAPPINGS COMPLETE ========"

# ============================================================================
# Loop Dependence Directives (partial accumulators across IC groups)
# ============================================================================
puts "======== LOOP DEPENDENCE DIRECTIVES ========"

# DS Conv partial accumulator (acc_ds updated across ic_grp iterations)
set_loop_dependence "DS acc_ds" {
    MAIN_LOOP_BB1:for:STAGE2_COL:for:DS_CONV_IC_GRP:for/acc_ds
    run/MAIN_LOOP_BB1:for:STAGE2_COL:for:DS_CONV_IC_GRP:for/acc_ds
} $roots

# C3A1 partial accumulator
set_loop_dependence "C3A1 acc_c3a1" {
    MAIN_LOOP_BB1:for:STAGE2_COL:for:C3A1_IC_GRP:for/acc_c3a1
    run/MAIN_LOOP_BB1:for:STAGE2_COL:for:C3A1_IC_GRP:for/acc_c3a1
} $roots

# C3A2 partial accumulator
set_loop_dependence "C3A2 acc_c3a2" {
    MAIN_LOOP_BB1:for:STAGE3_COL:for:C3A2_IC_GRP:for/acc_c3a2
    run/MAIN_LOOP_BB1:for:STAGE3_COL:for:C3A2_IC_GRP:for/acc_c3a2
} $roots

# C3B1 partial accumulator
set_loop_dependence "C3B1 acc_c3b1" {
    MAIN_LOOP_BB1:for:STAGE3_COL:for:C3B1_IC_GRP:for/acc_c3b1
    run/MAIN_LOOP_BB1:for:STAGE3_COL:for:C3B1_IC_GRP:for/acc_c3b1
} $roots

# C3CAT partial accumulator
set_loop_dependence "C3CAT acc_ccat" {
    MAIN_LOOP_BB1:for:STAGE3_COL:for:CCAT_IC_GRP:for/acc_ccat
    run/MAIN_LOOP_BB1:for:STAGE3_COL:for:CCAT_IC_GRP:for/acc_ccat
} $roots

puts "======== LOOP DIRECTIVES COMPLETE ========"

go compile

# ============================================================================
# POST-COMPILE: Array partition (resource paths exist after go compile)
# ============================================================================
# Sliding window pattern: BRAM reads are now sequential (1 per KR row per col step).
# No SCHD-4 risk from these buffers.
# Partition dim=1 (rows=8) is optional but helps Catapult allocate independent
# row BRAMs → simpler scheduling.
# ============================================================================
puts "======== POST-COMPILE PARTITION ========"

# Diagnostic: dump resource paths to find correct hierarchy
puts "--- POST-COMPILE RESOURCE PATHS (buf-related) ---"
catch {
    set all_dirs [directive get -rec]
    foreach d $all_dirs {
        if {[string match "*input_buf*" $d] || [string match "*save_buf*" $d]} {
            puts "RSC PATH: $d"
        }
    }
} diag_err
if {$diag_err ne {}} { puts "diag err: $diag_err" }
puts "--- END RESOURCE PATHS ---"

# Refresh roots post-compile (hierarchy is fully resolved)
set post_roots $roots
catch {
    set design_names2 [solution design get]
    foreach d $design_names2 {
        if {[string first "::" $d] < 0} {
            lappend post_roots "/$d"
            lappend post_roots "/$d/run"
        }
    }
}
set post_roots [lsort -unique $post_roots]
puts "Post-compile roots: $post_roots"

# Partition dim=1 (rows=8) only — channels are now packed (dim=3 no longer exists)
# c3a2_input_buf[8][80] × 256-bit: 8 row BRAMs, 3 KC reads/row → 3 instances each
apply_complete_partition "POST c3a2_input_buf dim=1" {
    c3a2_input_buf
    run/c3a2_input_buf
} $post_roots 1

# ds_input_buf[8][160] × 256-bit: 8 row BRAMs, 3 KC reads/row → 3 instances each
apply_complete_partition "POST ds_input_buf dim=1" {
    ds_input_buf
    run/ds_input_buf
} $post_roots 1

puts "======== POST-COMPILE PARTITION DONE ========"

# ============================================================================
# Technology Library (Xilinx Zynq UltraScale+, same as three_pe_block)
# ============================================================================
solution library remove *
solution library add mgc_Xilinx-ZYNQ-uplus-1_beh -- -rtlsyntool Vivado -manufacturer Xilinx -family ZYNQ-uplus -speed -1 -part xczu11eg-ffvb1517-1-e
solution library add Xilinx_RAMS
solution library add ccs_fpga_hic
solution library add Xilinx_FIFO

go architect

go schedule

# Generate .ccs launcher
set ccs_fd [open ./backbone_block1.ccs w]
puts $ccs_fd {// Auto-generated by run_backbone_catapult.tcl}
puts $ccs_fd {if {[info script] != {} && [file isdirectory [file rootname [info script]]]} {
  project load [file rootname [info script]] 2025.2
} else {
  set _dirs [lsort -dictionary [glob -nocomplain backbone_block1*]]
  set _proj {}
  foreach d $_dirs { if {[file isdirectory $d]} { set _proj $d } }
  if {$_proj ne {}} {
    project load $_proj 2025.2
  } else {
    error {unable to locate project directory 'backbone_block1*'}
  }
}}
close $ccs_fd

if {![info exists KEEP_GUI_OPEN] || !$KEEP_GUI_OPEN} {
  exit
}
