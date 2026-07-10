# ══════════════════════════════════════════════════════════════════════
# pr_main.tcl — 参数化 OpenROAD PR 主流程(所有变体 × 策略共用这一份)
# ══════════════════════════════════════════════════════════════════════
# 调用方式(由 run_all.sh 驱动):
#   RUN_DIR=<pr/runs/xxx> openroad -exit -threads N -metrics metrics.json \
#       -log openroad.log pr_main.tcl
# RUN_DIR/run_cfg.tcl 由 gen_runs.py 生成,包含该 run 的全部参数(cfg 数组);
# 本文件不含任何硬编码坐标/路径。
#
# ── 3nm 工艺下 OpenROAD 的已知局限(实验解读时注意)──────────────────
# 1. USC-3N-2D 是简化学术 PDK:tech LEF 只有基本 WIDTH/PITCH/SPACING,
#    没有真实 3nm 的 LEF58 规则(SADP 双图形着色、EUV 最小面积、cut 间距
#    阵列规则等)。DRT 报告的 DRC 数会显著低于真实签核;修复建议:
#    最终 DRC 以厂商 signoff 工具(如 Calibre + 真实 runset)为准,OpenROAD
#    的 DRV 只做趋势比较(A/B 策略同一口径,可比)。
# 2. 库中没有 filler / tapcell / tie / 天线二极管单元:阱连续性、密度
#    填充与天线修复无法完成;repair_antennas 被跳过 —— 天线违例需在
#    真实 PDK 上重新评估。
# 3. Liberty 只有单角(0.7V/25C NLDM):无多角(setup/hold 同库),hold
#    数字偏乐观;SRAM 宏用占位时序(tcq=218ps),宏内部功耗未建模。
# 4. 本 build 无 write_gds,且 PDK 不带单元 GDS:最终版图输出 DEF/ODB,
#    GDSII 需拿到单元 GDS 后用 KLayout 依 DEF 合并(见 pr/README.md)。

# ── 0. 读取 run 配置 ──────────────────────────────────────────────────
if {![info exists ::env(RUN_DIR)]} { error "请设置 RUN_DIR 环境变量(pr/runs/<run_id>)" }
set run_dir [file normalize $::env(RUN_DIR)]
source [file join $run_dir run_cfg.tcl]
set cfg(run_dir) $run_dir
cd $run_dir
puts "\[PR\] run=$cfg(run_id) 策略=$cfg(strategy) die=$cfg(die_name) 变体=$cfg(variant)"

source [file join $cfg(script_dir) floorplan_hb.tcl]

# ── 1. 读入 PDK 与设计 ────────────────────────────────────────────────
read_lef $cfg(tech_lef)
read_lef $cfg(std_lef)
foreach lef $cfg(macro_lefs) { read_lef $lef }
read_liberty $cfg(std_lib)
foreach lib $cfg(macro_libs) { read_liberty $lib }

read_verilog $cfg(netlist)
link_design $cfg(top)
read_sdc $cfg(sdc)

# 门级网表中的常量网(assign x=1'b1/1'b0,如 SRAM 的 ce_in 拉高、AND 门
# 的封端输入):此 PDK 没有 tie cell,把它们标为 special 让 GRT/DRT 跳过
#(两者都按 isSpecial 判定)—— 物理实现时这类引脚应就近接电源/地轨
#(或补 tie cell 后重跑)。
# 同时把 sigtype 改回 SIGNAL:dbNetwork 不为 POWER/GROUND 网上的引脚建
# 时序图顶点,挂在常量网上的标准单元输入(base_die 有 4.1 万个)会让
# repair_design 在 checkDriverArcSlew 里解引用空 Vertex 直接段错误。
set n_const 0
foreach net [[ord::get_db_block] getNets] {
  set st [$net getSigType]
  if {$st eq "POWER" || $st eq "GROUND"} {
    if {![$net isSpecial]} { $net setSpecial }
    $net setSigType "SIGNAL"
    incr n_const
  }
}
if {$n_const} { puts "\[PR\] $n_const 个常量网标记为 special+SIGNAL(无 tie cell,跳过布线且保留时序图顶点)" }

utl::metric "IFP::instance_count" [sta::network_instance_count]

# ── 2. Floorplan(尺寸来自 arch_model 的变体物理模型) ────────────────
set m $cfg(core_margin)
initialize_floorplan -site coreSite \
  -die_area  [list 0 0 $cfg(die_w) $cfg(die_h)] \
  -core_area [list $m $m [expr {$cfg(die_w)-$m}] [expr {$cfg(die_h)-$m}]]
source $cfg(tracks_file)

# 去掉综合期插入的 buffer,由后端时序修复重新决定
remove_buffers

utl::metric "FP::die_area_um2" [format %.1f [expr {$cfg(die_w)*$cfg(die_h)}]]

# ── 3+4. 宏阵列 + HB 接口(簇/KOZ/面阵引脚/通道,策略核心逻辑)────────
# 宏摆放也在 hb::setup 内(参数化阵列,与 tile/bank 分组对齐;
# 策略 A 的 KOZ 簇落宏间通道 —— 见 floorplan_hb.tcl 头注释)
hb::setup

proc have_macros {} {
  foreach inst [[ord::get_db_block] getInsts] {
    if {[[$inst getMaster] getType] eq "BLOCK"} { return 1 }
  }
  return 0
}
if {[have_macros]} {
  # 宏下方的 site row 切除,避免 std cell 布进宏区
  cut_rows -halo_width_x 2 -halo_width_y 2
}

# 其余(非 HB)引脚按普通边缘 IO 放置;受 up: 约束的 HB 引脚落面阵
place_pins -hor_layers $cfg(io_hor_layer) -ver_layers $cfg(io_ver_layer)

# ── 5. 电源网络(M1 followpins;学术 PDK 无上层电源条带定义)─────────
source $cfg(pdn_tcl)
pdngen

# ── 6. 全局布线资源设置(在布局前设好,routability-driven 布局要用)──
set_routing_layers -signal $cfg(routing_layers) -clock $cfg(clock_layers)
foreach {layer adj} $cfg(layer_adjustments) {
  set_global_routing_layer_adjustment $layer $adj
}
set_macro_extension 2
# HB 区域容量调整:KOZ 深度语义 + 策略 B 的通道引导(等效 add_routing_guide)
hb::apply_grt_adjustments

# ── 7. 全局布局 ───────────────────────────────────────────────────────
global_placement -routability_driven -density $cfg(place_density) \
  -pad_left 2 -pad_right 2

# ── 8. 寄生估计 + 违例修复 ────────────────────────────────────────────
source $cfg(rc_tcl)
set_wire_rc -signal -layer $cfg(wire_rc_layer)
set_wire_rc -clock  -layer $cfg(wire_rc_layer_clk)
estimate_parasitics -placement
repair_design -slew_margin $cfg(slew_margin) -cap_margin $cfg(cap_margin)
set_placement_padding -global -left 1 -right 1
detailed_placement -max_displacement {200 100}   ;# KOZ 阻挡内的 cell 需要更大位移窗口才能合法化

report_worst_slack -max -digits 3
utl::metric "RSZ::worst_slack_max" [sta::worst_slack -max]

# ── 9. 时钟树综合(库中仅 BUFx4 一种缓冲,见文件头局限 3)─────────────
repair_clock_inverters
clock_tree_synthesis -root_buf $cfg(cts_buffer) -buf_list $cfg(cts_buffer) \
  -sink_clustering_enable -sink_clustering_max_diameter $cfg(cts_cluster_diameter)
repair_clock_nets
detailed_placement -max_displacement {200 100}   ;# KOZ 阻挡内的 cell 需要更大位移窗口才能合法化

# ── 10. 时序修复(传播时钟)────────────────────────────────────────────
# 只修 setup:此 PDK 的 liberty 全部 area:0,hold buffer 选择在本
# OpenROAD build 中会空指针崩溃(rsz::RepairHold);单角库下 hold 修复
# 意义也有限 —— hold 数字仅作 A/B 趋势参考(文件头局限 3)。
set_propagated_clock [all_clocks]
estimate_parasitics -placement
repair_timing -setup -skip_gate_cloning
detailed_placement -max_displacement {200 100}   ;# KOZ 阻挡内的 cell 需要更大位移窗口才能合法化
check_placement -verbose

utl::metric "DPL::utilization" [format %.1f [expr [rsz::utilization] * 100]]
# 单元面积从 ODB/LEF 求和(此 PDK liberty 全部 area:0,rsz::design_area 不可用)
proc placed_area_um2 {} {
  set dbu [[ord::get_db_tech] getDbUnitsPerMicron]
  set a 0.0
  foreach inst [[ord::get_db_block] getInsts] {
    set mst [$inst getMaster]
    set a [expr {$a + double([$mst getWidth])*[$mst getHeight]/$dbu/$dbu}]
  }
  return $a
}
utl::metric "DPL::design_area_um2" [format %.1f [placed_area_um2]]
write_db [file join $run_dir post_cts.odb]

# ── 11. 全局布线(拥堵是 A/B 对比的核心指标之一)──────────────────────
pin_access
set grt_congested 0
if {[catch {
  global_route -guide_file [file join $run_dir route.guide] \
    -congestion_iterations $cfg(grt_congestion_iterations) \
    -congestion_report_file [file join $run_dir congestion.rpt] \
    -verbose
} err]} {
  # 拥堵不收敛(策略 A 高密度 KOZ 下预期可能发生):
  # 用 -allow_congestion 重跑拿到溢出数据,标记本 run 拥堵失败
  puts "\[PR\] 全局布线拥堵不收敛:$err —— 以 allow_congestion 重跑采集溢出指标"
  set grt_congested 1
  global_route -guide_file [file join $run_dir route.guide] \
    -congestion_iterations 5 \
    -congestion_report_file [file join $run_dir congestion.rpt] \
    -allow_congestion -verbose
}
utl::metric "GRT::congested" $grt_congested

# 天线检查:PDK 无二极管单元,只检不修(局限 2)
catch { check_antennas }

# ── 12. 详细布线 ──────────────────────────────────────────────────────
set drt_args [list detailed_route \
  -output_drc  [file join $run_dir route_drc.rpt] \
  -output_maze [file join $run_dir maze.log] \
  -save_guide_updates -verbose 1]
if {$cfg(droute_end_iter) >= 0} {
  lappend drt_args -droute_end_iter $cfg(droute_end_iter)
}
set drt_failed 0
if {[catch { eval $drt_args } err]} {
  puts "\[PR\] 详细布线异常:$err"
  set drt_failed 1
}
utl::metric "DRT::failed" $drt_failed
if {!$drt_failed} {
  utl::metric "DRT::drv" [detailed_route_num_drvs]
}

# ── 13. 版图输出(无 write_gds:DEF/ODB + 网表,GDS 走 KLayout 合并)──
write_def     [file join $run_dir $cfg(run_id).def]
write_db      [file join $run_dir $cfg(run_id).odb]
write_verilog [file join $run_dir $cfg(run_id).pnr.v]

# ── 14. 最终寄生与报告 ────────────────────────────────────────────────
estimate_parasitics -global_routing

puts "===== FINAL TIMING ====="
report_worst_slack -min -digits 3
report_worst_slack -max -digits 3
report_tns -digits 3
report_clock_skew -digits 3
report_check_types -max_slew -max_capacitance -max_fanout -violators -digits 3
puts "===== FINAL POWER ====="
catch { report_power }
puts "===== FINAL AREA ====="
report_design_area

utl::metric "DRT::worst_slack_min" [sta::worst_slack -min]
utl::metric "DRT::worst_slack_max" [sta::worst_slack -max]
utl::metric "DRT::tns_max" [sta::total_negative_slack -max]
utl::metric "DRT::clock_skew" [expr abs([sta::worst_clock_skew -setup])]
utl::metric "DRT::clock_period" [get_property [lindex [all_clocks] 0] period]

utl::metric "FLOW::complete" 1
puts "\[PR\] run $cfg(run_id) 完成"
