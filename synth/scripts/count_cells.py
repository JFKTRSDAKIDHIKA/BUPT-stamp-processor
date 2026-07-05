#!/usr/bin/env python3
"""count_cells.py — hierarchy-aware cell counting in a gate-level netlist.

Yosys' `stat` (and a naive grep) count cell instantiations per MODULE BODY;
a macro instantiated once inside a module that is itself instantiated 16x
shows up as "1". PnR floorplanning needs the FLATTENED instance count, so
this script parses the structural Verilog, builds the hierarchy DAG and
multiplies instance counts down from the top.

Usage:  python3 count_cells.py <netlist.v> <top> [cell_type ...]
        (default cell types: SRAM macro + DFFs + latch)

Output: one line per cell type: "<type> <flattened_count>".
Exit 1 if the top module is not found.
"""

import re
import sys
from collections import defaultdict

KEYWORDS = {"module", "endmodule", "input", "output", "inout", "wire",
            "reg", "assign", "parameter", "localparam", "supply0", "supply1"}

MODULE_RE = re.compile(r"^\s*module\s+(\\?\S+?)\s*[(;]")
# "  CellType InstName (" — the instance header of a structural netlist.
# The type must be \S+ (not \w): yosys paramod names contain = ' $ \ etc.
INST_RE = re.compile(r"^\s*(\\?\S+)\s+(\\?\S+)\s*\($")


def parse(netlist_path):
    """module name -> {child cell/module type: count}"""
    children = defaultdict(lambda: defaultdict(int))
    current = None
    with open(netlist_path) as fh:
        for line in fh:
            m = MODULE_RE.match(line)
            if m:
                current = m.group(1)
                continue
            if line.strip() == "endmodule":
                current = None
                continue
            if current is None:
                continue
            m = INST_RE.match(line.rstrip())
            if m and m.group(1) not in KEYWORDS:
                children[current][m.group(1)] += 1
    return children


def flat_count(children, module, cell_type, memo):
    """flattened number of cell_type instances under one `module` instance"""
    if module == cell_type:
        return 1
    if module not in children:          # leaf cell of another type
        return 0
    if module in memo:
        return memo[module]
    total = sum(cnt * flat_count(children, child, cell_type, memo)
                for child, cnt in children[module].items())
    memo[module] = total
    return total


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    netlist, top = sys.argv[1], sys.argv[2]
    types = sys.argv[3:] or ["mobol_sram_4096x512_2r1w",
                             "DFFHx1", "DFFHQNx1", "DHLx1"]
    children = parse(netlist)
    if top not in children:
        sys.exit(f"error: top module '{top}' not found in {netlist}")
    for t in types:
        print(t, flat_count(children, t if False else top, t, {}))


if __name__ == "__main__":
    main()
