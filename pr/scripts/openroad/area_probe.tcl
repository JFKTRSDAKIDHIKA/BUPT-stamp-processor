# ══════════════════════════════════════════════════════════════════════
# area_probe.tcl — 网表精确面积探测(arch_model.py 调用,结果缓存)
# ══════════════════════════════════════════════════════════════════════
# 为什么不用 yosys synth_stat:stat 的模块局部计数不乘实例化次数,多实例
# 化子模块(如 16 个 ring_router)会被严重低估。此处让 OpenROAD 读入网表
# 并 link(自动展平),逐实例按 LEF 尺寸求和,得到精确的标准单元/宏面积。
#
# 输入(环境变量):
#   AP_TECH_LEF / AP_STD_LEF / AP_MACRO_LEFS(空格分隔)
#   AP_STD_LIB  / AP_MACRO_LIBS(空格分隔)
#   AP_NETLIST / AP_TOP / AP_OUT(输出 JSON 路径)

suppress_message ODB 127
suppress_message ODB 128

read_lef $::env(AP_TECH_LEF)
read_lef $::env(AP_STD_LEF)
foreach lef $::env(AP_MACRO_LEFS) { read_lef $lef }
read_liberty $::env(AP_STD_LIB)
foreach lib $::env(AP_MACRO_LIBS) { read_liberty $lib }

read_verilog $::env(AP_NETLIST)
link_design $::env(AP_TOP)

set block [[ [ord::get_db] getChip ] getBlock]
set dbu   [[ord::get_db_tech] getDbUnitsPerMicron]

set n_std 0; set a_std 0.0
set n_mac 0; set a_mac 0.0
array set mac_count {}

foreach inst [$block getInsts] {
  set m [$inst getMaster]
  set w [expr {double([$m getWidth])  / $dbu}]
  set h [expr {double([$m getHeight]) / $dbu}]
  set a [expr {$w * $h}]
  if {[$m getType] eq "BLOCK"} {
    incr n_mac
    set a_mac [expr {$a_mac + $a}]
    set nm [$m getName]
    if {[info exists mac_count($nm)]} { incr mac_count($nm) } else { set mac_count($nm) 1 }
  } else {
    incr n_std
    set a_std [expr {$a_std + $a}]
  }
}

# 顶层端口数(供 HB 引脚映射 sanity check)
set n_ports [llength [$block getBTerms]]

set f [open $::env(AP_OUT) w]
puts $f "{"
puts $f "  \"top\": \"$::env(AP_TOP)\","
puts $f "  \"n_std\": $n_std,"
puts $f "  \"std_um2\": [format %.1f $a_std],"
puts $f "  \"n_macro\": $n_mac,"
puts $f "  \"macro_um2\": [format %.1f $a_mac],"
puts $f "  \"n_ports\": $n_ports,"
set pairs {}
foreach nm [array names mac_count] { lappend pairs "\"$nm\": $mac_count($nm)" }
puts $f "  \"macros\": {[join $pairs ,]}"
puts $f "}"
close $f
puts "AREA_PROBE_OK $::env(AP_TOP) std=$n_std/$a_std macro=$n_mac/$a_mac"
exit
