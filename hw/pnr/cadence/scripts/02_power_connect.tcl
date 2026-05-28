# Stage 02 — Global net rules. Re-invoked after each cell-insertion pass.

proc connectGlobalNets {} {
    # Stdcells and bsg_fakeram SRAMs both expose PG pins as VDD/VSS.
    globalNetConnect VDD -type pgpin -pin VDD -all
    globalNetConnect VSS -type pgpin -pin VSS -all
    globalNetConnect VDD -type tiehi -all
    globalNetConnect VSS -type tielo -all
    applyGlobalNets
}

connectGlobalNets
