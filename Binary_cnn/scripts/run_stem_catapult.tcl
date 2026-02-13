# Catapult batch script for StemProcessor
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

# Build candidate roots from discovered design names + known fallbacks.
set roots {
    /StemProcessor
    /StemProcessor/run
    /stem_processor/StemProcessor
    /stem_processor/StemProcessor/run
}
set design_names {}
catch {set design_names [solution design get]}
foreach d $design_names {
    if {[string first "::" $d] >= 0} { continue }
    lappend roots "/$d"
    lappend roots "/$d/run"
}
set roots [lsort -unique $roots]
puts "Memory mapping roots: $roots"

set map_total 0
set map_ok 0

incr map_total
if {[map_buffer_resource "line_buf_a.buffer" {line_buf_a.buffer line_buf_a/buffer run/line_buf_a.buffer run/line_buf_a/buffer} $roots]} { incr map_ok }
incr map_total
if {[map_buffer_resource "conv1_buf.buffer" {conv1_buf.buffer conv1_buf/buffer run/conv1_buf.buffer run/conv1_buf/buffer} $roots]} { incr map_ok }
incr map_total
if {[map_buffer_resource "mp_buf.buffer" {mp_buf.buffer mp_buf/buffer run/mp_buf.buffer run/mp_buf/buffer} $roots]} { incr map_ok }

incr map_total
if {[map_buffer_resource "conv2_row_stage" {conv2_row_stage run/conv2_row_stage} $roots]} { incr map_ok }
incr map_total
if {[map_buffer_resource "mp_row_stage" {mp_row_stage run/mp_row_stage} $roots]} { incr map_ok }

puts "Memory directive mapping summary: ${map_ok}/${map_total} resources mapped."

# Array partitioning for parallel channel access (dim=4 is inner channel index).
# Use discovered hierarchy roots; do not hardcode /StemProcessor/run paths.
set part_total 0
set part_ok 0
incr part_total
if {[apply_array_partition "line_buf_a.buffer" {line_buf_a.buffer line_buf_a/buffer run/line_buf_a.buffer run/line_buf_a/buffer} $roots 4 8]} { incr part_ok }
incr part_total
if {[apply_array_partition "conv1_buf.buffer" {conv1_buf.buffer conv1_buf/buffer run/conv1_buf.buffer run/conv1_buf/buffer} $roots 4 8]} { incr part_ok }
incr part_total
if {[apply_array_partition "mp_buf.buffer" {mp_buf.buffer mp_buf/buffer run/mp_buf.buffer run/mp_buf/buffer} $roots 4 8]} { incr part_ok }

# Complete partition on row staging arrays (channel dim) for parallel register preload access.
proc apply_complete_partition {label keys roots dim} {
    foreach root $roots {
        foreach key $keys {
            set path "${root}/${key}:rsc"
            if {![catch {directive set $path -ARRAY_PARTITION complete -dim $dim} err]} {
                puts "COMPLETE PARTITION OK: ${label} -> ${path}"
                incr ::part_ok
                incr ::part_total
                return 1
            }
        }
    }
    puts "COMPLETE PARTITION SKIP: ${label} (path not found or unsupported)"
    incr ::part_total
    return 0
}

apply_complete_partition "conv2_row_stage dim3" {conv2_row_stage run/conv2_row_stage} $roots 3
apply_complete_partition "mp_row_stage dim3" {mp_row_stage run/mp_row_stage} $roots 3

puts "Array partition summary: ${part_ok}/${part_total} resources partitioned."

# Keep clock overhead explicit to avoid SCHD-22 style schedule blockers.
if {[catch {directive set -CLOCK_OVERHEAD 0} clk_err]} {
    puts "CLOCK_OVERHEAD set failed: $clk_err"
} else {
    puts "CLOCK_OVERHEAD set to 0"
}

# ============================================================================
# Loop Scheduling and Dependence Directives
# ============================================================================
# Help scheduler by breaking false dependencies on partial accumulator arrays.
# These arrays are used to flatten nested accumulator loops and eliminate
# feedback paths that cause SCHD-3 errors.
# ============================================================================

puts "======== APPLYING LOOP DEPENDENCE DIRECTIVES ========"

# Helper proc to set loop dependence directives
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

# Try to set false inter-iteration dependence on partial accumulator arrays
# This helps the scheduler understand that loop iterations are independent.
set_loop_dependence "Conv0 partial acc" {
    run/MAIN_LOOP:if#1:for:CONV0_IC:for:acc_partial_conv0
    MAIN_LOOP:if#1:for:CONV0_IC:for:acc_partial_conv0
} $roots

set_loop_dependence "Conv1 partial acc" {
    run/MAIN_LOOP:if#1:for:CONV1_IC_GRP:for:acc_partial_conv1
    MAIN_LOOP:if#1:for:CONV1_IC_GRP:for:acc_partial_conv1
} $roots

set_loop_dependence "Conv2 partial acc" {
    run/MAIN_LOOP:if#3:if:for:CONV2_IC_GRP:for:acc_spatial
    MAIN_LOOP:if#3:if:for:CONV2_IC_GRP:for:acc_spatial
} $roots

set_loop_dependence "Conv3 partial acc" {
    run/MAIN_LOOP:if#5:if:for:CONV3_IC_GRP:for:acc_partial
    MAIN_LOOP:if#5:if:for:CONV3_IC_GRP:for:acc_partial
} $roots

puts "======== LOOP DIRECTIVES COMPLETE ========"

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
