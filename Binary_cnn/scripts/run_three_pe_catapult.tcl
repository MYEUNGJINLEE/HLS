# Catapult batch script for ThreePEBlock (Fused Single-Loop Architecture)
# Run example:
#   cd Binary_cnn
#   catapult -shell -file scripts/run_three_pe_catapult.tcl

project new -name three_pe_block

solution new -state initial
solution options defaults
solution options set /Output/GenerateCycleNetlist false

# ============================================================================
# 소스 파일 등록
# ============================================================================

solution file add ./src/stem/stem_config.h    -type C++
solution file add ./src/pe/pe_config.h        -type C++
solution file add ./src/pe/generic_pe.h       -type C++
solution file add ./src/pe/generic_pe.cpp     -type C++
solution file add ./src/pe/three_pe_block.h   -type C++
solution file add ./src/pe/three_pe_block.cpp -type C++

solution file add ./src/pe/three_pe_block_tb.cpp -type C++ -exclude true

go analyze

# ============================================================================
# 계층 디버그
# ============================================================================
puts "======== DESIGN HIERARCHY DEBUG ========"
catch {puts "solution design get: [solution design get]"}
catch {puts "directive get -rec: [directive get -rec]"}
puts "========================================"

# ============================================================================
# 헬퍼 프로시저
# ============================================================================

proc map_bram {label keys roots} {
    foreach root $roots {
        foreach key $keys {
            set path "${root}/${key}:rsc"
            if {![catch {directive set $path -MAP_TO_MODULE {BLOCK_1R1W_RBW}} err]} {
                puts "BRAM OK  : ${label} -> ${path}"
                return 1
            }
        }
    }
    puts "BRAM SKIP: ${label}"
    return 0
}

proc map_register {label keys roots} {
    foreach root $roots {
        foreach key $keys {
            set path "${root}/${key}:rsc"
            if {![catch {directive set $path -MAP_TO_MODULE {[Register]}} err]} {
                puts "REG OK   : ${label} -> ${path}"
                return 1
            }
        }
    }
    puts "REG SKIP : ${label}"
    return 0
}

proc set_dependence {label keys roots} {
    foreach root $roots {
        foreach key $keys {
            set path "${root}/${key}"
            if {![catch {directive set $path -DEPENDENCE_TYPE inter -DEPENDENCE false} err]} {
                puts "DEP OK   : ${label} -> ${path}"
                return 1
            }
        }
    }
    puts "DEP SKIP : ${label}"
    return 0
}

# ============================================================================
# 루트 경로 구성
# ============================================================================

set roots {
    /ThreePEBlock
    /ThreePEBlock/run
    /ThreePEBlock/run_straight
    /ThreePEBlock/run_branch_cat
    /ThreePEBlock/run_shortcut
    /three_pe_block/ThreePEBlock
    /three_pe_block/ThreePEBlock/run
    /three_pe_block/ThreePEBlock/run_straight
    /three_pe_block/ThreePEBlock/run_branch_cat
    /three_pe_block/ThreePEBlock/run_shortcut
}

set design_names {}
catch {set design_names [solution design get]}
foreach d $design_names {
    if {[string first "::" $d] >= 0} { continue }
    lappend roots "/$d"
    lappend roots "/$d/run"
}
set roots [lsort -unique $roots]
puts "Roots: $roots"

# ============================================================================
# 전역 디렉티브
# ============================================================================

# MEM_MAP_THRESHOLD: 8192b 이하 배열 → 레지스터, 초과 → BRAM
# line_buf (655KB), w3_a_mem (36KB) → BRAM 명시 매핑
# w1_a/b/c_mem (4KB), w_ccat_mem (4KB) → 레지스터 자동 매핑
if {[catch {directive set -MEM_MAP_THRESHOLD 8192} err]} {
    puts "MEM_MAP_THRESHOLD failed: $err"
} else {
    puts "MEM_MAP_THRESHOLD = 8192"
}

if {[catch {directive set -CLOCK_OVERHEAD 0} err]} {
    puts "CLOCK_OVERHEAD failed: $err"
} else {
    puts "CLOCK_OVERHEAD = 0"
}

# ============================================================================
# BRAM 매핑
# ============================================================================
puts "======== BRAM MAPPINGS ========"

# ThreePEBlock 라인버퍼 (8×160×64×8b = 655KB)
map_bram "line_buf" {
    line_buf run/line_buf
    run_straight/line_buf
    run_branch_cat/line_buf
    run_shortcut/line_buf
} $roots

# ThreePEBlock Conv3×3 가중치 타일 메모리 (64×8×72b = 36KB)
map_bram "w3_a_mem" {
    w3_a_mem run/w3_a_mem
    run_straight/w3_a_mem
    run_branch_cat/w3_a_mem
    run_shortcut/w3_a_mem
} $roots

# ThreePEBlock 숏컷 버퍼 (2×160×64×8b = 160KB)
map_bram "shortcut_buf" {
    shortcut_buf run/shortcut_buf
} $roots

puts "======== BRAM MAPPINGS DONE ========"

# ============================================================================
# 레지스터 매핑: 소형 가중치/BN 배열
# ============================================================================
puts "======== REGISTER MAPPINGS ========"

# PA Conv1×1 / DW3×3 가중치 및 BN (모두 < 8192b → 레지스터)
map_register "w1_a_mem" {w1_a_mem run/w1_a_mem} $roots
map_register "w_dw_mem" {w_dw_mem run/w_dw_mem} $roots
map_register "shift_a"  {shift_a  run/shift_a}  $roots
map_register "bias_a"   {bias_a   run/bias_a}   $roots

# PB Conv1×1 가중치
map_register "w1_b_mem" {w1_b_mem run/w1_b_mem} $roots
map_register "shift_b"  {shift_b  run/shift_b}  $roots
map_register "bias_b"   {bias_b   run/bias_b}   $roots

# PC Conv1×1 가중치
map_register "w1_c_mem" {w1_c_mem run/w1_c_mem} $roots
map_register "shift_c"  {shift_c  run/shift_c}  $roots
map_register "bias_c"   {bias_c   run/bias_c}   $roots

# CCAT 가중치/BN
map_register "w_ccat_mem"  {w_ccat_mem  run/w_ccat_mem}  $roots
map_register "shift_ccat"  {shift_ccat  run/shift_ccat}  $roots
map_register "bias_ccat"   {bias_ccat   run/bias_ccat}   $roots

puts "======== REGISTER MAPPINGS DONE ========"

# ============================================================================
# 루프 의존성 디렉티브
# ============================================================================
puts "======== LOOP DEPENDENCE ========"

# STRAIGHT Conv3×3: IC_GRP 루프의 acc 배열 loop-carried dep
set_dependence "STRAIGHT IC_GRP acc" {
    run_straight/STRAIGHT_MAIN:for:STR_COL:for:STR_IC_GRP:for/acc_a
    STRAIGHT_MAIN:for:STR_COL:for:STR_IC_GRP:for/acc_a
} $roots

# BRANCH_CAT: IC_GRP 루프의 acc_a 배열 loop-carried dep
set_dependence "BRANCH IC_GRP acc_a" {
    run_branch_cat/BRANCH_MAIN:for:BR_COL:for:BR_IC_GRP:for/acc_a
    BRANCH_MAIN:for:BR_COL:for:BR_IC_GRP:for/acc_a
} $roots

# SHORTCUT: IC_GRP 루프의 acc_a 배열 loop-carried dep
set_dependence "SHORTCUT IC_GRP acc_a" {
    run_shortcut/SC_MAIN:for:SC_COL:for:SC_IC_GRP:for/acc_a
    SC_MAIN:for:SC_COL:for:SC_IC_GRP:for/acc_a
} $roots

# SHUFFLE2V_S1: SH_OC_B 루프의 acc_b 배열 (Stage1 내 PB 1×1)
set_dependence "SHUFFLE SH_OC_B acc_b" {
    run_shuffle2v_s1/SH_MAIN:for:SH_READ_ROW:for:SH_OC_B:for/acc_b
    SH_MAIN:for:SH_READ_ROW:for:SH_OC_B:for/acc_b
} $roots

# SHUFFLE2V_S1: SH_OC_C 루프의 acc_c 배열 (Stage2 내 PC 1×1)
set_dependence "SHUFFLE SH_OC_C acc_c" {
    run_shuffle2v_s1/SH_MAIN:for:SH_COL:for:SH_OC_C:for/acc_c
    SH_MAIN:for:SH_COL:for:SH_OC_C:for/acc_c
} $roots

puts "======== LOOP DEPENDENCE DONE ========"

go compile

# ============================================================================
# .ccs 런처 파일 생성
# ============================================================================
set ccs_fd [open ./three_pe_block.ccs w]
puts $ccs_fd {// Auto-generated by run_three_pe_catapult.tcl}
puts $ccs_fd {if {[info script] != {} && [file isdirectory [file rootname [info script]]]} {
  project load [file rootname [info script]] 2025.2
} else {
  set _dirs [lsort -dictionary [glob -nocomplain three_pe_block*]]
  set _proj {}
  foreach d $_dirs { if {[file isdirectory $d]} { set _proj $d } }
  if {$_proj ne {}} {
    project load $_proj 2025.2
  } else {
    error {unable to locate project directory 'three_pe_block*'}
  }
}}
close $ccs_fd

if {![info exists KEEP_GUI_OPEN] || !$KEEP_GUI_OPEN} {
  exit
}
