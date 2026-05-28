set_ccopt_property buffer_cells   { BUF_X1 BUF_X2 BUF_X4 BUF_X8 BUF_X16 BUF_X32 }
set_ccopt_property inverter_cells { INV_X1 INV_X2 INV_X4 INV_X8 INV_X16 INV_X32 }
set_ccopt_property target_skew      0.05  ;# 50 ps
set_ccopt_property target_max_trans 0.15  ;# 150 ps

ccopt_design

# CCOpt owns skew now; drop uncertainty to jitter-only.
set_interactive_constraint_modes [list func]
set_clock_uncertainty 0.05 [get_clocks clk]
set_interactive_constraint_modes {}
puts "INFO: post-CTS clock uncertainty → 50 ps (jitter only)."

file mkdir "$REPORT_DIR/cts"
redirect "$REPORT_DIR/cts/clock_tree.rpt" {
    report_ccopt_clock_trees -summary
}
redirect "$REPORT_DIR/cts/skew_summary.rpt" {
    report_ccopt_skew_groups
}
