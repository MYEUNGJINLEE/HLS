# Catapult GUI script for StemProcessor
# Run example:
#   cd Binary_cnn
#   catapult -gui -file scripts/run_stem_catapult_gui.tcl

# Reuse batch flow, but keep GUI open after compile.
set KEEP_GUI_OPEN 1
source ./scripts/run_stem_catapult.tcl

# NOTE: no 'exit' so GUI stays open
