# Catapult batch script for StemProcessor (2-PE architecture)
# Run example:
#   cd Binary_cnn
#   catapult -shell -file scripts/run_stem_catapult.tcl

project new -name stem_processor

# Start from a clean solution state
solution new -state initial
solution options defaults

# Useful default flow options
solution options set /Output/GenerateCycleNetlist false

# Design + testbench registration (TB excluded from synthesis flow)
solution file add ./src/stem/stem_processor.cpp -type C++
solution file add ./src/stem/stem_engine_a.cpp -type C++
solution file add ./src/stem/stem_engine_b.cpp -type C++
solution file add ./src/stem/stem_processor_tb.cpp -type C++ -exclude true

# Analyze
go analyze

# ============================================================================
# Discover Design Hierarchy
# ============================================================================
puts "======== DESIGN HIERARCHY DEBUG ========"
catch {puts "solution design get: [solution design get]"}
catch {puts "directive get -rec: [directive get -rec]"}
puts "========================================"

# ============================================================================
# Memory Banking Directives
# ============================================================================
# Force SRAM mapping to prevent 'memories' pass exploration.
# Try multiple path formats since naming depends on Catapult version.
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
    puts "COMPLETE PARTITION SKIP: ${label} (path not found or unsupported)"
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
    puts "DEPENDENCE SKIP: ${label} (path not found, pragma may suffice)"
    return 0
}

# ============================================================================
# Build candidate roots for 2-PE hierarchy
# ============================================================================

# Top-level roots
set roots {
    /StemProcessor
    /StemProcessor/run
    /stem_processor/StemProcessor
    /stem_processor/StemProcessor/run
}

# Engine-A roots
set roots_a {
    /StemProcessor/engine_a
    /StemProcessor/engine_a/run
    /StemProcessor/run/engine_a
    /StemProcessor/run/engine_a/run
    /StemEngineA
    /StemEngineA/run
}

# Engine-B roots
set roots_b {
    /StemProcessor/engine_b
    /StemProcessor/engine_b/run
    /StemProcessor/run/engine_b
    /StemProcessor/run/engine_b/run
    /StemEngineB
    /StemEngineB/run
}

# Add discovered design names
set design_names {}
catch {set design_names [solution design get]}
foreach d $design_names {
    if {[string first "::" $d] >= 0} { continue }
    lappend roots "/$d"
    lappend roots "/$d/run"
    lappend roots_a "/$d/engine_a"
    lappend roots_a "/$d/engine_a/run"
    lappend roots_b "/$d/engine_b"
    lappend roots_b "/$d/engine_b/run"
}
set roots [lsort -unique $roots]
set roots_a [lsort -unique $roots_a]
set roots_b [lsort -unique $roots_b]

# Combined roots for global directives
set all_roots [concat $roots $roots_a $roots_b]
set all_roots [lsort -unique $all_roots]

puts "Top-level roots: $roots"
puts "Engine-A roots: $roots_a"
puts "Engine-B roots: $roots_b"

# ============================================================================
# Memory Mapping: Line Buffers → BLOCK_1R1W_RBW
# ============================================================================

puts "======== MEMORY MAPPING ========"

set map_total 0
set map_ok 0

# Engine-A: line_buf_a, conv1_buf
incr map_total
if {[map_buffer_resource "line_buf_a.buffer" {line_buf_a.buffer line_buf_a/buffer run/line_buf_a.buffer run/line_buf_a/buffer} $roots_a]} { incr map_ok }
incr map_total
if {[map_buffer_resource "conv1_buf.buffer" {conv1_buf.buffer conv1_buf/buffer run/conv1_buf.buffer run/conv1_buf/buffer} $roots_a]} { incr map_ok }

# Engine-B: mp_buf
incr map_total
if {[map_buffer_resource "mp_buf.buffer" {mp_buf.buffer mp_buf/buffer run/mp_buf.buffer run/mp_buf/buffer} $roots_b]} { incr map_ok }

# Engine-B: staging arrays (if above MEM_MAP_THRESHOLD → BRAM mapping)
incr map_total
if {[map_buffer_resource "conv2_staging" {conv2_staging run/conv2_staging} $roots_b]} { incr map_ok }
incr map_total
if {[map_buffer_resource "mp_staging" {mp_staging run/mp_staging} $roots_b]} { incr map_ok }

puts "Memory directive mapping summary: ${map_ok}/${map_total} resources mapped."

# ============================================================================
# Array Partitioning
# ============================================================================

puts "======== ARRAY PARTITIONING ========"

set part_total 0
set part_ok 0

# Engine-A line buffer channel dim partitioning
incr part_total
if {[apply_array_partition "line_buf_a.buffer" {line_buf_a.buffer line_buf_a/buffer run/line_buf_a.buffer run/line_buf_a/buffer} $roots_a 4 8]} { incr part_ok }
incr part_total
if {[apply_array_partition "conv1_buf.buffer" {conv1_buf.buffer conv1_buf/buffer run/conv1_buf.buffer run/conv1_buf/buffer} $roots_a 4 8]} { incr part_ok }

# Engine-B line buffer channel dim partitioning
incr part_total
if {[apply_array_partition "mp_buf.buffer" {mp_buf.buffer mp_buf/buffer run/mp_buf.buffer run/mp_buf/buffer} $roots_b 4 8]} { incr part_ok }

# Engine-B staging: complete partition on channel dim for parallel preload access
incr part_total
if {[apply_complete_partition "conv2_staging dim2" {conv2_staging run/conv2_staging} $roots_b 2]} { incr part_ok }
incr part_total
if {[apply_complete_partition "mp_staging dim2" {mp_staging run/mp_staging} $roots_b 2]} { incr part_ok }

puts "Array partition summary: ${part_ok}/${part_total} resources partitioned."

# ============================================================================
# Global Directives
# ============================================================================

# Keep clock overhead explicit to avoid SCHD-22 style schedule blockers.
if {[catch {directive set -CLOCK_OVERHEAD 0} clk_err]} {
    puts "CLOCK_OVERHEAD set failed: $clk_err"
} else {
    puts "CLOCK_OVERHEAD set to 0"
}

# MEM_MAP_THRESHOLD: arrays with <= N elements are mapped to registers.
if {[catch {directive set -MEM_MAP_THRESHOLD 8192} mem_err]} {
    puts "MEM_MAP_THRESHOLD set failed: $mem_err"
} else {
    puts "MEM_MAP_THRESHOLD set to 8192"
}

# ============================================================================
# Register Mapping (Engine-A)
# ============================================================================

puts "======== ENGINE-A REGISTER/RAM MAPPINGS ========"

# Conv0 accumulator
map_to_register "A: Conv0 partial acc" {
    run/MAIN_LOOP_A:if#1:for:acc_partial_conv0
    MAIN_LOOP_A:if#1:for:acc_partial_conv0
    run/MAIN_LOOP_A:if:for:acc_partial_conv0
    MAIN_LOOP_A:if:for:acc_partial_conv0
} $roots_a

# Conv1 accumulator
map_to_register "A: Conv1 partial acc" {
    run/MAIN_LOOP_A:if#1:for:acc_partial_conv1
    MAIN_LOOP_A:if#1:for:acc_partial_conv1
    run/MAIN_LOOP_A:if:for:acc_partial_conv1
    MAIN_LOOP_A:if:for:acc_partial_conv1
} $roots_a

# Conv2 accumulator
map_to_register "A: Conv2 spatial acc" {
    run/MAIN_LOOP_A:if#2:for:acc_spatial
    MAIN_LOOP_A:if#2:for:acc_spatial
    run/MAIN_LOOP_A:if#2:if:for:acc_spatial
    MAIN_LOOP_A:if#2:if:for:acc_spatial
    run/MAIN_LOOP_A:if:for:acc_spatial
    MAIN_LOOP_A:if:for:acc_spatial
} $roots_a

# Conv0 pixel output
map_to_register "A: conv0_pix" {
    run/MAIN_LOOP_A:if#1:for:conv0_pix
    MAIN_LOOP_A:if#1:for:conv0_pix
    run/MAIN_LOOP_A:if:for:conv0_pix
    MAIN_LOOP_A:if:for:conv0_pix
} $roots_a

# Conv1 output group
map_to_register "A: out_grp" {
    run/MAIN_LOOP_A:if#1:for:out_grp
    MAIN_LOOP_A:if#1:for:out_grp
    run/MAIN_LOOP_A:if:for:out_grp
    MAIN_LOOP_A:if:for:out_grp
} $roots_a

# Conv2 pixel output
map_to_register "A: conv2_pix" {
    run/MAIN_LOOP_A:if#2:for:conv2_pix
    MAIN_LOOP_A:if#2:for:conv2_pix
    run/MAIN_LOOP_A:if#2:if:for:conv2_pix
    MAIN_LOOP_A:if#2:if:for:conv2_pix
} $roots_a

# Window arrays
map_to_register "A: window_conv0" {
    run/MAIN_LOOP_A:if#1:for:window
    MAIN_LOOP_A:if#1:for:window
    run/MAIN_LOOP_A:if:for:window
    MAIN_LOOP_A:if:for:window
} $roots_a

map_to_register "A: window_conv2" {
    run/MAIN_LOOP_A:if#2:for:window
    MAIN_LOOP_A:if#2:for:window
    run/MAIN_LOOP_A:if#2:if:for:window
    MAIN_LOOP_A:if#2:if:for:window
} $roots_a

# Shift arrays
map_to_register "A: shift0" {run/shift0 shift0} $roots_a
map_to_register "A: shift1" {run/shift1 shift1} $roots_a
map_to_register "A: shift2" {run/shift2 shift2} $roots_a

# Bias arrays
map_to_register "A: bias0" {run/bias0 bias0} $roots_a
map_to_register "A: bias1" {run/bias1 bias1} $roots_a
map_to_register "A: bias2" {run/bias2 bias2} $roots_a

puts "======== ENGINE-A MAPPINGS COMPLETE ========"

# ============================================================================
# Register Mapping (Engine-B)
# ============================================================================

puts "======== ENGINE-B REGISTER/RAM MAPPINGS ========"

# Conv3 accumulator
map_to_register "B: Conv3 partial acc" {
    run/MAIN_LOOP_B:for:acc_partial
    MAIN_LOOP_B:for:acc_partial
    run/MAIN_LOOP_B:for:CONV3_ROW:for:acc_partial
    MAIN_LOOP_B:for:CONV3_ROW:for:acc_partial
} $roots_b

# Conv3 preload arrays
map_to_register "B: conv2_local" {
    run/MAIN_LOOP_B:for:conv2_local
    MAIN_LOOP_B:for:conv2_local
    run/MAIN_LOOP_B:for:CONV3_ROW:for:conv2_local
    MAIN_LOOP_B:for:CONV3_ROW:for:conv2_local
} $roots_b

map_to_register "B: mp_local" {
    run/MAIN_LOOP_B:for:mp_local
    MAIN_LOOP_B:for:mp_local
    run/MAIN_LOOP_B:for:CONV3_ROW:for:mp_local
    MAIN_LOOP_B:for:CONV3_ROW:for:mp_local
} $roots_b

# MaxPool result
map_to_register "B: mp_result" {
    run/MAIN_LOOP_B:for:mp_result
    MAIN_LOOP_B:for:mp_result
    run/MAIN_LOOP_B:for:MAXPOOL_ROW:for:mp_result
    MAIN_LOOP_B:for:MAXPOOL_ROW:for:mp_result
} $roots_b

# Output array
map_to_register "B: out_ch" {
    run/MAIN_LOOP_B:for:out_ch
    MAIN_LOOP_B:for:out_ch
    run/MAIN_LOOP_B:for:CONV3_ROW:for:out_ch
    MAIN_LOOP_B:for:CONV3_ROW:for:out_ch
} $roots_b

# MaxPool window
map_to_register "B: window_mp" {
    run/MAIN_LOOP_B:for:window
    MAIN_LOOP_B:for:window
    run/MAIN_LOOP_B:for:MAXPOOL_ROW:for:window
    MAIN_LOOP_B:for:MAXPOOL_ROW:for:window
} $roots_b

# Shift/bias
map_to_register "B: shift3" {run/shift3 shift3} $roots_b
map_to_register "B: bias3" {run/bias3 bias3} $roots_b

puts "======== ENGINE-B MAPPINGS COMPLETE ========"

# ============================================================================
# Loop Dependence Directives
# ============================================================================

puts "======== APPLYING LOOP DEPENDENCE DIRECTIVES ========"

# Engine-A: Conv0 partial accumulator
set_loop_dependence "A: Conv0 partial acc" {
    run/MAIN_LOOP_A:if#1:for:CONV0_IC:for:acc_partial_conv0
    MAIN_LOOP_A:if#1:for:CONV0_IC:for:acc_partial_conv0
    run/MAIN_LOOP_A:if:for:CONV0_IC:for:acc_partial_conv0
    MAIN_LOOP_A:if:for:CONV0_IC:for:acc_partial_conv0
} $roots_a

# Engine-A: Conv1 partial accumulator
set_loop_dependence "A: Conv1 partial acc" {
    run/MAIN_LOOP_A:if#1:for:CONV1_IC_GRP:for:acc_partial_conv1
    MAIN_LOOP_A:if#1:for:CONV1_IC_GRP:for:acc_partial_conv1
    run/MAIN_LOOP_A:if:for:CONV1_IC_GRP:for:acc_partial_conv1
    MAIN_LOOP_A:if:for:CONV1_IC_GRP:for:acc_partial_conv1
} $roots_a

# Engine-A: Conv2 spatial accumulator
set_loop_dependence "A: Conv2 spatial acc" {
    run/MAIN_LOOP_A:if#2:if:for:CONV2_IC_GRP:for:acc_spatial
    MAIN_LOOP_A:if#2:if:for:CONV2_IC_GRP:for:acc_spatial
    run/MAIN_LOOP_A:if:for:CONV2_IC_GRP:for:acc_spatial
    MAIN_LOOP_A:if:for:CONV2_IC_GRP:for:acc_spatial
} $roots_a

# Engine-B: Conv3 partial accumulator
set_loop_dependence "B: Conv3 partial acc" {
    run/MAIN_LOOP_B:for:CONV3_IC_GRP:for:acc_partial
    MAIN_LOOP_B:for:CONV3_IC_GRP:for:acc_partial
    run/MAIN_LOOP_B:for:CONV3_ROW:for:CONV3_IC_GRP:for:acc_partial
    MAIN_LOOP_B:for:CONV3_ROW:for:CONV3_IC_GRP:for:acc_partial
} $roots_b

puts "======== LOOP DIRECTIVES COMPLETE ========"

# ============================================================================
# Inter-PE Channel (FIFO) Depth Hints
# ============================================================================
# conv0_pipe: 1 row of Conv0 output at a time (320 pixels × 256bit)
# conv2_pipe: 1 row of Conv2 output at a time (160 pixels × 256bit)
# w3_relay  : 64 weight/param packets (STEM_PACKS_CONV3)
# ============================================================================

puts "======== INTER-PE FIFO DEPTHS ========"

set fifo_candidates_top {
    /StemProcessor/run/conv0_pipe
    /StemProcessor/conv0_pipe
    /stem_processor/StemProcessor/run/conv0_pipe
}
foreach fp $fifo_candidates_top {
    catch {
        directive set "${fp}:rsc" -FIFO_DEPTH 320
        puts "FIFO conv0_pipe depth=320 @ ${fp}"
    }
    catch {
        directive set "${fp}:rsc" -FIFO_DEPTH 160
        puts "FIFO conv2_pipe depth=160 @ [string map {conv0_pipe conv2_pipe} ${fp}]"
    }
    catch {
        directive set "[string map {conv0_pipe conv2_pipe} ${fp}]:rsc" -FIFO_DEPTH 160
    }
    catch {
        directive set "[string map {conv0_pipe w3_relay} ${fp}]:rsc" -FIFO_DEPTH 64
    }
}

puts "========================================"

go compile

# Generate .ccs launcher so GUI can reopen project
set ccs_fd [open ./stem_processor.ccs w]
puts $ccs_fd {// Auto-generated by run_stem_catapult.tcl}
puts $ccs_fd {if {[info script] != {} && [file isdirectory [file rootname [info script]]]} {
  project load [file rootname [info script]] 2025.2
} else {
  set _dirs [lsort -dictionary [glob -nocomplain stem_processor*]]
  set _proj {}
  foreach d $_dirs { if {[file isdirectory $d]} { set _proj $d } }
  if {$_proj ne {}} {
    project load $_proj 2025.2
  } else {
    error {unable to locate project directory 'stem_processor*'}
  }
}}
close $ccs_fd

if {![info exists KEEP_GUI_OPEN] || !$KEEP_GUI_OPEN} {
  exit
}
