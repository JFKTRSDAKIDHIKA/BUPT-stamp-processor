#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# run_all.sh — Task 4:一键跑完所有 (die × 架构变体 × 重分布策略) 的 PR
# ══════════════════════════════════════════════════════════════════════
# 用法:
#   ./run_all.sh                          # experiments.yaml 默认全集
#   ./run_all.sh --smoke                  # 冒烟:smoke_die × baseline × A/B
#   ./run_all.sh -j 2                     # 2 个 run 并行(注意内存)
#   ./run_all.sh --dies buffer_die --variants baseline dense8192
#   ./run_all.sh --force                  # 忽略 DONE 标记重跑
# 特性:断点续跑(每个完成的 run 落 DONE 标记);结束后自动收集结果。

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PAR=1; FORCE=0; GEN_ARGS=(); SMOKE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -j) PAR="$2"; shift 2;;
    --force) FORCE=1; shift;;
    --smoke) SMOKE=1; shift;;
    --dies|--variants|--strategies)
      key="$1"; shift; vals=()
      while [[ $# -gt 0 && "$1" != -* ]]; do vals+=("$1"); shift; done
      GEN_ARGS+=("$key" "${vals[@]}");;
    *) echo "未知参数: $1"; exit 1;;
  esac
done
if [[ $SMOKE -eq 1 ]]; then
  GEN_ARGS+=(--dies smoke_die --variants baseline)
fi

# OpenROAD 与线程数从 experiments.yaml 读取(单一真值源)
OPENROAD=$(python3 - <<'EOF'
import yaml, pathlib
cfg = yaml.safe_load((pathlib.Path("config/experiments.yaml")).read_text())
print(cfg["openroad"]["bin"]); print(cfg["openroad"]["threads"])
EOF
)
OR_BIN=$(echo "$OPENROAD" | sed -n 1p)
OR_THREADS=$(echo "$OPENROAD" | sed -n 2p)

# 1) 生成/刷新所有 run 配置(含 Task-1 物理模型计算,面积探测有缓存)
echo "── 生成 run 配置 ──"
python3 scripts/gen_runs.py "${GEN_ARGS[@]}" || exit 1

# 2) 逐 run 执行(xargs 控制并行度;每 run 独立日志,失败不阻塞其它 run)
run_one() {
  local rd="$1"
  local id; id=$(basename "$rd")
  if [[ -f "$rd/DONE" && "$FORCE" -eq 0 ]]; then
    echo "[skip] $id(已完成;--force 可重跑)"
    return 0
  fi
  rm -f "$rd/DONE"
  echo "[run ] $id 开始 $(date '+%H:%M:%S')"
  RUN_DIR="$rd" "$OR_BIN" -exit -no_splash -threads "$OR_THREADS" \
      -metrics "$rd/metrics.json" -log "$rd/openroad.log" \
      scripts/openroad/pr_main.tcl > "$rd/stdout.log" 2>&1
  local rc=$?
  # 完成判据:退出码 0 且 metrics 里有 FLOW::complete
  if [[ $rc -eq 0 ]] && grep -q 'FLOW__complete\|FLOW::complete' "$rd/metrics.json" 2>/dev/null; then
    touch "$rd/DONE"
    echo "[ ok ] $id 完成 $(date '+%H:%M:%S')"
  else
    echo "[FAIL] $id (rc=$rc) → 见 $rd/openroad.log"
  fi
  return 0
}
export -f run_one
export OR_BIN OR_THREADS FORCE

mapfile -t RUN_DIRS < <(python3 -c "
import json
for r in json.load(open('runs/manifest.json')): print(r['run_dir'])")

printf '%s\n' "${RUN_DIRS[@]}" | xargs -P "$PAR" -I{} bash -c 'run_one "$@"' _ {}

# 3) 收集与对比
echo "── 收集结果 ──"
python3 scripts/collect_results.py

# 4) 出图:每 run floorplan.png/layout.png + out/plots/ 的 A/B 并排对比
echo "── 生成版图图片 ──"
python3 scripts/plot_layout.py
