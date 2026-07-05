"""Load the unified architecture YAML (config/mobol_arch.yaml) — the SAME
file the C++ simulator reads — so the compiler's structural constants,
schedule defaults and DSE search space are not hardcoded but come from one
source of truth.

The compiler needs: the chip topology (num_tiles/banks, MXU size) for tile
assignment and address encoding, and the schedule list (compiler.dse) for
DSE. Timing/port knobs are baked into each emitted trace's .config line and
consumed by the simulator; the compiler only needs their VALUES to write
that line, which it gets from the chosen schedule here.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Dict

try:
    import yaml
except Exception as e:  # pragma: no cover
    raise RuntimeError("edgc requires PyYAML (pip install pyyaml)") from e


def _default_arch_path() -> str:
    # tools/edgc/edgc/arch.py -> repo_root/config/mobol_arch.yaml
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    return os.path.join(root, "config", "mobol_arch.yaml")


@dataclass
class ArchParams:
    # structural (must match the compiled C++ build)
    num_tiles: int = 16
    num_banks: int = 4
    tiles_per_group: int = 4
    mxu_m: int = 16
    mxu_n: int = 16
    mxu_k: int = 16
    # local scratchpad size (bytes) — for the compiler's LOCAL memory map
    local_bytes: int = 1 << 18
    shared_bytes: int = 1 << 23
    # dram device model + default ramulator path (relative to repo root)
    ramulator_config: str = "config/ramulator_3d_dram.yaml"
    # schedule defaults + DSE list (each is a dict of trace .config knobs)
    default_sched: str = "baseline"
    dse: List[Dict] = field(default_factory=list)
    # absolute path this was loaded from (to resolve ramulator relative paths)
    _path: str = ""


def load_arch(path: str = "") -> ArchParams:
    path = path or _default_arch_path()
    with open(path) as f:
        y = yaml.safe_load(f)
    a = ArchParams()
    a._path = os.path.abspath(path)
    s = y.get("structural", {})
    a.num_tiles = s.get("num_tiles", a.num_tiles)
    a.num_banks = s.get("num_banks", a.num_banks)
    a.tiles_per_group = s.get("tiles_per_group", a.tiles_per_group)
    a.mxu_m = s.get("mxu_m", a.mxu_m)
    a.mxu_n = s.get("mxu_n", a.mxu_n)
    a.mxu_k = s.get("mxu_k", a.mxu_k)
    dram = y.get("dram", {})
    a.ramulator_config = dram.get("ramulator_config", a.ramulator_config)
    comp = y.get("compiler", {})
    a.default_sched = comp.get("default_sched", a.default_sched)
    a.dse = comp.get("dse", [])
    return a


def resolve_ramulator(arch: ArchParams) -> str:
    """Absolute path to the ramulator config referenced by the arch YAML."""
    rc = arch.ramulator_config
    if os.path.isabs(rc):
        return rc
    root = os.path.abspath(os.path.join(os.path.dirname(arch._path), ".."))
    cand = os.path.join(root, rc)
    return cand if os.path.exists(cand) else rc
