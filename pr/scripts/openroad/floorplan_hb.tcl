# ══════════════════════════════════════════════════════════════════════
# floorplan_hb.tcl — HB（混合键合）接口布局：几何定义 + KOZ + 重分布通道
# ══════════════════════════════════════════════════════════════════════
#
# 本文件集中实现策略 A（no_redist）与策略 B（edge_redist）在物理布局上的全部差异。
# 所有坐标均通过 cfg() 参数动态计算，禁止硬编码。单位：µm。
#
# 簇（cluster）定义：一条垂直链路对应的 HB 焊盘方阵，分为两类：
#   - tl：tile↔bank 私有链路（见 SPEC §7.1，每个 tile 一条）
#   - dc：DRAM 通道键合（见 SPEC §7.2/§8，共 16 通道；策略 A 中为“直穿”簇）
#
# KOZ 深度模式（koz_mode）：
#   hard    ：本 die 键合面上的焊盘簇 → 硬布局阻挡，且顶两层（M8/M9）布线容量清零。
#   punch   ：策略 A 中 DRAM 直穿 buffer die 的 feed-through → 硬布局阻挡，
#             且全层（M1–M9）布线容量清零（模拟 TSV 阵列贯穿整个金属栈）。
#   shallow ：策略 B 中 buffer die 顶面的 DRAM 落点 → 不占用布局资源（std cell
#             可正常摆放在其下方），仅削减 M8/M9 容量（因焊盘和再分布走线占用）。
#
# 对外接口（按 pr_main.tcl 顺序调用）：
#   hb::setup                 — 生成簇、创建 KOZ 布局阻挡、约束 HB 引脚、建立通道软阻挡。
#   hb::apply_grt_adjustments — 全局布线前，根据 koz_mode 施加区域容量削减，
#                               并对通道外高层布线进行收紧（等效于 add_routing_guide 的引导作用）。
#   hb::dump_geometry         — 将簇/通道几何信息写入 JSON，供结果收集脚本进行分区拥堵分析。
#
# 全局变量：
#   clusters : 列表，每项为 {name kind mode x1 y1 x2 y2}
#   channels : 列表，每项为 {dir x1 y1 x2 y2}
#   m_geom   : 宏阵列几何信息 {ax0 ay0 cols rows mw mh gapx gapy}，无宏时为空

namespace eval hb {
  variable clusters {}   ;# list of {name kind mode x1 y1 x2 y2}
  variable channels {}   ;# list of {dir x1 y1 x2 y2}
  variable m_geom  {}    ;# 宏阵列几何 {ax0 ay0 cols rows mw mh gapx gapy}(无宏则空)
}

# ── 工具 ──────────────────────────────────────────────────────────────
proc hb::core_bbox {} {
  # 经 ODB 取 core 区域(µm),比 ord::get_core_area 在不同 build 间更稳
  set block [ord::get_db_block]
  set rect [$block getCoreArea]
  set dbu [[ord::get_db_tech] getDbUnitsPerMicron]
  list [expr {double([$rect xMin])/$dbu}] [expr {double([$rect yMin])/$dbu}] \
       [expr {double([$rect xMax])/$dbu}] [expr {double([$rect yMax])/$dbu}]
}

proc hb::cluster_side {nbits} {
  global cfg
  set cols [expr {int(ceil(sqrt(double($nbits))))}]
  return [expr {$cols * $cfg(hb_pitch)}]
}

proc hb::clamp {v lo hi} { expr {$v < $lo ? $lo : ($v > $hi ? $hi : $v)} }

# 以中心+边长生成 KOZ 矩形(pad 方阵外扩 koz_margin),并夹在 core 内
proc hb::rect_of {cx cy side} {
  global cfg
  lassign [hb::core_bbox] X1 Y1 X2 Y2
  set h [expr {$side / 2.0 + $cfg(koz_margin)}]
  list [hb::clamp [expr {$cx-$h}] $X1 $X2] [hb::clamp [expr {$cy-$h}] $Y1 $Y2] \
       [hb::clamp [expr {$cx+$h}] $X1 $X2] [hb::clamp [expr {$cy+$h}] $Y1 $Y2]
}

# ── 宏阵列摆放(参数化,替代 rtl_macro_placer) ──────────────────────
# 为什么不用 rtl_macro_placer:它不支持核心区中部的矩形布局阻挡
# (MPL-0060),而策略 A 的 KOZ 正是这种阻挡。且本设计的宏天然是规则
# 阵列(tile SRAM 按 4×4 tile 网格、bank SRAM 按 bank 阵列)—— 与架构
# 分组(SPEC §1)对齐的确定性摆放本身就是正确的物理意图。
# 布局规则:宏阵列在可用区域(策略 B 先向内让出边缘带)内均匀展开,
# 宏间间隙 = (可用宽 − 宏总宽)/(列数+1);策略 A 的 KOZ 簇随后落在
# 这些间隙的交叉带上(见 gen_clusters),互不重叠由 arch_model 的
# die 尺寸推导保证(gapA ≥ KOZ 边长 + 2×halo)。
proc hb::place_macros {} {
  global cfg
  variable m_geom
  set m_geom {}
  set macros {}
  foreach inst [[ord::get_db_block] getInsts] {
    if {[[$inst getMaster] getType] eq "BLOCK"} { lappend macros $inst }
  }
  if {[llength $macros] == 0} { return }
  set macros [lsort -command {apply {{a b} {string compare [$a getName] [$b getName]}}} $macros]

  lassign [hb::core_bbox] X1 Y1 X2 Y2
  if {$cfg(strategy) eq "edge_redist"} {
    # 策略 B:宏阵列避开边缘 HB 带
    set X1 [expr {$X1 + $cfg(edge_band_w)}]; set Y1 [expr {$Y1 + $cfg(edge_band_w)}]
    set X2 [expr {$X2 - $cfg(edge_band_w)}]; set Y2 [expr {$Y2 - $cfg(edge_band_w)}]
  }
  set cols $cfg(macro_cols); set rows $cfg(macro_rows)
  set mw $cfg(macro_w);      set mh $cfg(macro_h)
  set gapx [expr {(($X2-$X1) - $cols*$mw) / ($cols+1)}]
  set gapy [expr {(($Y2-$Y1) - $rows*$mh) / ($rows+1)}]
  if {$gapx < 0 || $gapy < 0} {
    error "\[HB\] 宏阵列放不下:请检查 arch_model 的 die 尺寸推导 (gapx=$gapx gapy=$gapy)"
  }
  set dbu [[ord::get_db_tech] getDbUnitsPerMicron]
  set site_h 0.144    ;# coreSite 高度:宏 y 对齐 site row
  set k 0
  foreach inst $macros {
    set c [expr {$k % $cols}]; set r [expr {$k / $cols}]
    set x [expr {$X1 + $gapx*($c+1) + $mw*$c}]
    set y [expr {$Y1 + $gapy*($r+1) + $mh*$r}]
    set y [expr {$Y1 + round(($y-$Y1)/$site_h)*$site_h}]
    $inst setOrient R0
    $inst setLocation [expr {int(round($x*$dbu))}] [expr {int(round($y*$dbu))}]
    $inst setPlacementStatus FIRM
    incr k
  }
  set m_geom [list $X1 $Y1 $cols $rows $mw $mh $gapx $gapy]
  puts "\[HB\] 宏阵列 ${cols}×${rows} 摆放完成 (gap = [format %.1f $gapx] × [format %.1f $gapy] µm)"
}

# ── 簇坐标生成 ────────────────────────────────────────────────────────
# 网格模式:簇按 grid_x × grid_y 阵列均匀散布核心区(策略 A;与 tile/bank
# 物理分组对齐:SPEC §1 group g = tile 4g..4g+3)
proc hb::grid_positions {n} {
  global cfg
  lassign [hb::core_bbox] X1 Y1 X2 Y2
  set cw [expr {$X2-$X1}]; set ch [expr {$Y2-$Y1}]
  set pos {}
  for {set j 0} {$j < $cfg(grid_y)} {incr j} {
    for {set i 0} {$i < $cfg(grid_x)} {incr i} {
      if {[llength $pos] >= $n} { break }
      lappend pos [list \
        [expr {$X1 + ($i+0.5)*$cw/$cfg(grid_x)}] \
        [expr {$Y1 + ($j+0.5)*$ch/$cfg(grid_y)}]]
    }
  }
  return $pos
}

# 宏间隙模式(策略 A 且有宏):KOZ 簇落在宏间通道的交叉带上
# tl 簇 → 垂直间隙 × 宏行中心;dc 簇 → 水平间隙 × 宏列中心
proc hb::gap_positions {kind n} {
  variable m_geom
  lassign $m_geom ax0 ay0 cols rows mw mh gapx gapy
  set pos {}
  if {$kind eq "v"} {
    for {set r 0} {$r < $rows} {incr r} {
      for {set i 0} {$i <= $cols} {incr i} {
        if {[llength $pos] >= $n} { break }
        lappend pos [list [expr {$ax0 + $i*($mw+$gapx) + $gapx/2.0}] \
                          [expr {$ay0 + $gapy*($r+1) + $mh*$r + $mh/2.0}]]
      }
    }
  } else {
    for {set j 0} {$j <= $rows} {incr j} {
      for {set c 0} {$c < $cols} {incr c} {
        if {[llength $pos] >= $n} { break }
        lappend pos [list [expr {$ax0 + $gapx*($c+1) + $mw*$c + $mw/2.0}] \
                          [expr {$ay0 + $j*($mh+$gapy) + $gapy/2.0}]]
      }
    }
  }
  return $pos
}

# 边缘模式:簇沿核心四边的"边缘带"中线均匀分布(策略 B)
proc hb::edge_positions {n} {
  global cfg
  lassign [hb::core_bbox] X1 Y1 X2 Y2
  set inset [expr {$cfg(edge_band_w) / 2.0}]
  set x1 [expr {$X1+$inset}]; set y1 [expr {$Y1+$inset}]
  set x2 [expr {$X2-$inset}]; set y2 [expr {$Y2-$inset}]
  set w [expr {$x2-$x1}]; set h [expr {$y2-$y1}]
  set P [expr {2.0*($w+$h)}]
  set pos {}
  for {set k 0} {$k < $n} {incr k} {
    set t [expr {($k+0.5)*$P/$n}]
    if {$t < $w} {                       ;# 下边 →
      lappend pos [list [expr {$x1+$t}] $y1]
    } elseif {$t < $w+$h} {              ;# 右边 ↑
      lappend pos [list $x2 [expr {$y1+($t-$w)}]]
    } elseif {$t < 2*$w+$h} {            ;# 上边 ←
      lappend pos [list [expr {$x2-($t-$w-$h)}] $y2]
    } else {                             ;# 左边 ↓
      lappend pos [list $x1 [expr {$y2-($t-2*$w-$h)}]]
    }
  }
  return $pos
}

# 生成全部簇:根据 (strategy × role) 决定每类簇的位置与 KOZ 深度
proc hb::gen_clusters {} {
  global cfg
  variable clusters
  set clusters {}
  set side_tl [hb::cluster_side $cfg(hb_bits_tile_link)]
  set side_dc [hb::cluster_side $cfg(hb_bits_dram_ch)]

  if {$cfg(strategy) eq "no_redist"} {
    # ── 策略 A:全部垂直接口散布核心区 ──
    # 有宏:簇落宏间通道交叉带(与宏互不重叠,由 arch_model 尺寸推导保证);
    # 无宏:tl 簇放网格点左下、dc 簇放右上,避免同格重叠
    variable m_geom
    if {[llength $m_geom] > 0} {
      set pos_tl [hb::gap_positions v $cfg(hb_n_tile_links)]
      set pos_dc [hb::gap_positions h $cfg(hb_n_dram_ch)]
      set off_tl 0.0; set off_dc 0.0
    } else {
      set pos_tl [hb::grid_positions $cfg(hb_n_tile_links)]
      set pos_dc [hb::grid_positions $cfg(hb_n_dram_ch)]
      set off_tl [expr {$side_tl * 0.55 + $cfg(koz_margin)}]
      set off_dc [expr {$side_dc * 0.55 + $cfg(koz_margin)}]
    }
    set i 0
    foreach p $pos_tl {
      lassign $p cx cy
      lappend clusters [list tl$i tl hard \
        {*}[hb::rect_of [expr {$cx-$off_tl}] [expr {$cy-$off_tl}] $side_tl]]
      incr i
    }
    # DRAM 直穿簇:base die 上是普通 pad 簇(hard);buffer die 上是
    # feed-through(punch)—— 这正是策略 A 破坏两个 die 核心区的机制
    set mode_dc [expr {$cfg(role) eq "buffer" ? "punch" : "hard"}]
    set i 0
    foreach p $pos_dc {
      lassign $p cx cy
      lappend clusters [list dc$i dc $mode_dc \
        {*}[hb::rect_of [expr {$cx+$off_dc}] [expr {$cy+$off_dc}] $side_dc]]
      incr i
    }
  } else {
    # ── 策略 B:base die 的所有 HB 全部移到边缘带;buffer die 的
    #    bank→base 键合(tl)也在边缘,而 DRAM 顶面落点(dc)保持在
    #    核心上方但只作浅层(shallow)占用,由 M8/M9 通道重分布到边缘 ──
    if {$cfg(role) eq "buffer"} {
      set pos_tl [hb::edge_positions $cfg(hb_n_tile_links)]
      set i 0
      foreach p $pos_tl {
        lassign $p cx cy
        lappend clusters [list tl$i tl hard {*}[hb::rect_of $cx $cy $side_tl]]
        incr i
      }
      set pos_dc [hb::grid_positions $cfg(hb_n_dram_ch)]
      set i 0
      foreach p $pos_dc {
        lassign $p cx cy
        lappend clusters [list dc$i dc shallow {*}[hb::rect_of $cx $cy $side_dc]]
        incr i
      }
    } else {
      # base die:tl + dc 混排沿边缘均匀分布,核心区零 KOZ(实验目标)
      set n_all [expr {$cfg(hb_n_tile_links) + $cfg(hb_n_dram_ch)}]
      set pos [hb::edge_positions $n_all]
      for {set k 0} {$k < $n_all} {incr k} {
        lassign [lindex $pos $k] cx cy
        if {$k % 2 == 0 && $k/2 < $cfg(hb_n_tile_links)} {
          lappend clusters [list tl[expr {$k/2}] tl hard {*}[hb::rect_of $cx $cy $side_tl]]
        } elseif {$k % 2 == 1 && $k/2 < $cfg(hb_n_dram_ch)} {
          lappend clusters [list dc[expr {$k/2}] dc hard {*}[hb::rect_of $cx $cy $side_dc]]
        } elseif {$k/2 >= $cfg(hb_n_tile_links)} {
          lappend clusters [list dc[expr {$k - $cfg(hb_n_tile_links)}] dc hard \
            {*}[hb::rect_of $cx $cy $side_dc]]
        } else {
          lappend clusters [list tl[expr {$k - $cfg(hb_n_dram_ch)}] tl hard \
            {*}[hb::rect_of $cx $cy $side_tl]]
        }
      }
    }
  }
  puts "\[HB\] 策略=$cfg(strategy) role=$cfg(role) 生成 [llength $clusters] 个 HB 簇 (tl边长=${side_tl}µm dc边长=${side_dc}µm)"
}

# ── KOZ 布局阻挡 ──────────────────────────────────────────────────────
proc hb::apply_koz {} {
  global cfg
  variable clusters
  set a_koz 0.0
  foreach c $clusters {
    lassign $c name kind mode x1 y1 x2 y2
    if {$mode eq "shallow"} { continue }  ;# 浅层落点不占布局资源
    # KOZ = 硬布局阻挡(std cell 与宏都不可入内)
    create_blockage -region [list $x1 $y1 $x2 $y2]
    set a_koz [expr {$a_koz + ($x2-$x1)*($y2-$y1)}]
  }
  lassign [hb::core_bbox] X1 Y1 X2 Y2
  set a_core [expr {($X2-$X1)*($Y2-$Y1)}]
  utl::metric "HB::n_clusters" [llength $clusters]
  utl::metric "HB::koz_area_um2" [format %.1f $a_koz]
  utl::metric "HB::koz_core_frac" [format %.4f [expr {$a_koz/$a_core}]]
  puts "\[HB\] KOZ 总面积 [format %.0f $a_koz] µm² = 核心区 [format %.2f [expr {100.0*$a_koz/$a_core}]]%"
}

# ── HB 引脚约束(面阵键合引脚 → up: 区域) ───────────────────────────
# 匹配 cfg(hb_pin_patterns) 的顶层总线是"垂直方向"信号,把它们按 bus
# 轮询绑定到各 tl 簇的 up: 面阵区域(OpenROAD 的 3D 面键合引脚机制);
# dc 簇在现有网表中没有对应端口(DRAM 侧未综合),作纯物理 KOZ 占位。
proc hb::assign_pins {} {
  global cfg
  variable clusters
  set block [[ [ord::get_db] getChip ] getBlock]

  # 1) 收集匹配的 bterm,按 bus 基名分组
  array set buses {}
  foreach bt [$block getBTerms] {
    set nm [$bt getName]
    foreach pat $cfg(hb_pin_patterns) {
      if {[string match $pat $nm]} {
        regsub {\[\d+\]$} $nm "" base
        lappend buses($base) $nm
        break
      }
    }
  }
  set bus_names [lsort [array names buses]]
  if {[llength $bus_names] == 0} {
    puts "\[HB\] 警告:hb_pin_patterns 未匹配到任何顶层端口,HB 引脚跳过"
    return
  }

  # 2) 在整个核心区定义顶层(M9)面阵引脚网格;实际落点由 up: 区域约束限定
  lassign [hb::core_bbox] X1 Y1 X2 Y2
  define_pin_shape_pattern -layer $cfg(hb_layer) \
    -x_step $cfg(hb_pitch) -y_step $cfg(hb_pitch) \
    -region [list $X1 $Y1 $X2 $Y2] \
    -size [list $cfg(hb_pad) $cfg(hb_pad)]

  # 3) tl 簇列表(有网表引脚的簇)
  set tl_rects {}
  foreach c $clusters {
    lassign $c name kind mode x1 y1 x2 y2
    if {$kind eq "tl"} { lappend tl_rects [list $x1 $y1 $x2 $y2] }
  }
  set n_tl [llength $tl_rects]

  # 4) 按位连续切片分配 + 容量守护
  # 不能按"整条总线→单簇"轮询:base die 的 ext_wdata 达 8192 位,单簇
  # 装不下(PPL-0111)。物理上每条 tile 链路本来就只承载 512b×W 的
  # flit lane,因此把所有匹配引脚按总线序拼接后,连续等分为 n_tl 段
  # (每簇一段连续位片 = 一条链路的 lane 组)。
  set all_pins {}
  foreach b $bus_names {
    # 总线内按位序排列(名称数值感知排序)
    lappend all_pins {*}[lsort -dictionary $buses($b)]
  }
  set total [llength $all_pins]
  set chunk [expr {($total + $n_tl - 1) / $n_tl}]
  set assigned {}
  for {set i 0} {$i < $n_tl} {incr i} {
    lappend assigned [lrange $all_pins [expr {$i*$chunk}] [expr {($i+1)*$chunk - 1}]]
  }
  for {set i 0} {$i < $n_tl} {incr i} {
    set pins [lindex $assigned $i]
    if {[llength $pins] == 0} { continue }
    lassign [lindex $tl_rects $i] x1 y1 x2 y2
    # 容量守护:若引脚多于网格点(留 15% 余量,PPL 自身有 keepout 折损),
    # 向外扩簇的引脚区域(仅引脚区,KOZ 不变)
    set need [expr {int([llength $pins] * 1.15) + 4}]
    set cap [expr {(int(($x2-$x1)/$cfg(hb_pitch))+1) * (int(($y2-$y1)/$cfg(hb_pitch))+1)}]
    while {$cap < $need} {
      set x1 [hb::clamp [expr {$x1-$cfg(hb_pitch)}] $X1 $X2]
      set y1 [hb::clamp [expr {$y1-$cfg(hb_pitch)}] $Y1 $Y2]
      set x2 [hb::clamp [expr {$x2+$cfg(hb_pitch)}] $X1 $X2]
      set y2 [hb::clamp [expr {$y2+$cfg(hb_pitch)}] $Y1 $Y2]
      set new_cap [expr {(int(($x2-$x1)/$cfg(hb_pitch))+1) * (int(($y2-$y1)/$cfg(hb_pitch))+1)}]
      if {$new_cap == $cap} {   ;# 四面都被 core 边界钳住,无法再扩
        puts "\[HB\] 警告:簇 tl$i 引脚区扩至 core 边界仍不足($cap < $need),交由 PPL 尽力放置"
        break
      }
      set cap $new_cap
    }
    set_io_pin_constraint -pin_names $pins \
      -region [format "up:{%.3f %.3f %.3f %.3f}" $x1 $y1 $x2 $y2]
    puts "\[HB\] 簇 tl$i ← [llength $pins] 个引脚 (容量 $cap)"
  }
}

# ── 策略 B 的重分布布线通道(仅 buffer die) ─────────────────────────
# 通道 = 从 bank 核心通向边缘的高层金属走廊:
#   * 软布局阻挡(max_density 限 25%):std cell 基本让位,允许中继 buffer
#   * 通道宽度由 gen_runs.py 依位宽×间距/利用率算出 —— "留够空间"的保证
proc hb::apply_channels {} {
  global cfg
  variable channels
  set channels {}
  if {!($cfg(strategy) eq "edge_redist" && $cfg(role) eq "buffer")} { return }
  lassign [hb::core_bbox] X1 Y1 X2 Y2
  set cw [expr {$X2-$X1}]; set ch [expr {$Y2-$Y1}]
  set w $cfg(channel_w)

  # 垂直通道(M9 方向)× channels_per_edge:对齐 bank/tile 网格列中心
  for {set i 0} {$i < $cfg(channels_per_edge)} {incr i} {
    set cx [expr {$X1 + ($i+0.5)*$cw/$cfg(channels_per_edge)}]
    set r [list [expr {$cx-$w/2}] $Y1 [expr {$cx+$w/2}] $Y2]
    lappend channels [concat v $r]
    create_blockage -region $r -max_density 25.0   ;# 部分阻挡:留 25% 给中继 buffer
  }
  # 水平通道(M8 方向):至少 1 条过 bank 中线
  set nh [expr {max(1, $cfg(grid_y))}]
  for {set j 0} {$j < $nh} {incr j} {
    set cy [expr {$Y1 + ($j+0.5)*$ch/$nh}]
    set r [list $X1 [expr {$cy-$w/2}] $X2 [expr {$cy+$w/2}]]
    lappend channels [concat h $r]
    create_blockage -region $r -max_density 25.0
  }
  utl::metric "HB::n_channels" [llength $channels]
  utl::metric "HB::channel_w_um" $w
  puts "\[HB\] 建立 [llength $channels] 条重分布通道,宽 ${w}µm"
}

# ── 全局布线容量调整(在 global_route 之前调用) ─────────────────────
# OpenROAD 没有 add_routing_guide 命令;等效机制是区域容量调整:
#   1) 簇上方按 koz_mode 清零/削减容量(pad 与 feed-through 遮挡)
#   2) 策略 B buffer die:通道以外的核心区在 M8/M9 上收紧
#      (offchannel_adj)→ 全局布线器把跨区重分布长线"吸"进通道
proc hb::apply_grt_adjustments {} {
  global cfg
  variable clusters
  variable channels

  foreach c $clusters {
    lassign $c name kind mode x1 y1 x2 y2
    if {$x2 <= $x1 || $y2 <= $y1} { continue }
    switch $mode {
      hard {          ;# pad 落点:顶两层不可走线
        foreach l {M8 M9} {
          set_global_routing_region_adjustment [list $x1 $y1 $x2 $y2] \
            -layer $l -adjustment 1.0
        }
      }
      punch {         ;# 直穿 feed-through:全金属栈被打断
        foreach l {M1 M2 M3 M4 M5 M6 M7 M8 M9} {
          set_global_routing_region_adjustment [list $x1 $y1 $x2 $y2] \
            -layer $l -adjustment 1.0
        }
      }
      shallow {       ;# 顶面落点:M8/M9 部分容量被 pad/再分布占用
        foreach l {M8 M9} {
          set_global_routing_region_adjustment [list $x1 $y1 $x2 $y2] \
            -layer $l -adjustment $cfg(shallow_pad_adj)
        }
      }
    }
  }

  # 通道引导:对"通道之外"的核心区补充收紧 M9(垂直)/M8(水平)
  if {[llength $channels] > 0} {
    lassign [hb::core_bbox] X1 Y1 X2 Y2
    # 垂直通道的补集(x 方向区间)→ 收紧 M9
    set xs {}
    foreach c $channels {
      lassign $c dir x1 y1 x2 y2
      if {$dir eq "v"} { lappend xs [list $x1 $x2] }
    }
    set xs [lsort -real -index 0 $xs]
    set prev $X1
    foreach seg [concat $xs [list [list $X2 $X2]]] {
      lassign $seg sx ex
      if {$sx - $prev > 1.0} {
        set_global_routing_region_adjustment [list $prev $Y1 $sx $Y2] \
          -layer M9 -adjustment $cfg(offchannel_adj)
      }
      set prev $ex
    }
    # 水平通道的补集(y 方向区间)→ 收紧 M8
    set ys {}
    foreach c $channels {
      lassign $c dir x1 y1 x2 y2
      if {$dir eq "h"} { lappend ys [list $y1 $y2] }
    }
    set ys [lsort -real -index 0 $ys]
    set prev $Y1
    foreach seg [concat $ys [list [list $Y2 $Y2]]] {
      lassign $seg sy ey
      if {$sy - $prev > 1.0} {
        set_global_routing_region_adjustment [list $X1 $prev $X2 $sy] \
          -layer M8 -adjustment $cfg(offchannel_adj)
      }
      set prev $ey
    }
    puts "\[HB\] 通道外 M8/M9 容量收紧至 [expr {(1.0-$cfg(offchannel_adj))*100}]% —— 重分布线被引导入通道"
  }
}

# ── 几何导出(供 collect_results.py 做核心区/边缘带分区拥堵统计) ────
proc hb::dump_geometry {} {
  global cfg
  variable clusters
  variable channels
  lassign [hb::core_bbox] X1 Y1 X2 Y2
  set f [open [file join $cfg(run_dir) clusters.json] w]
  puts $f "{"
  puts $f "  \"core\": \[$X1, $Y1, $X2, $Y2\],"
  puts $f "  \"edge_band_w\": $cfg(edge_band_w),"
  set items {}
  foreach c $clusters {
    lassign $c name kind mode x1 y1 x2 y2
    lappend items "    {\"name\": \"$name\", \"kind\": \"$kind\", \"mode\": \"$mode\", \"rect\": \[$x1, $y1, $x2, $y2\]}"
  }
  puts $f "  \"clusters\": \[\n[join $items ,\n]\n  \],"
  set items {}
  foreach c $channels {
    lassign $c dir x1 y1 x2 y2
    lappend items "    {\"dir\": \"$dir\", \"rect\": \[$x1, $y1, $x2, $y2\]}"
  }
  puts $f "  \"channels\": \[\n[join $items ,\n]\n  \]"
  puts $f "}"
  close $f
}

# ── 总入口 ────────────────────────────────────────────────────────────
proc hb::setup {} {
  hb::place_macros      ;# 先定宏阵列(簇坐标依赖宏间隙几何)
  hb::gen_clusters
  hb::apply_koz
  hb::assign_pins
  hb::apply_channels
  hb::dump_geometry
}
