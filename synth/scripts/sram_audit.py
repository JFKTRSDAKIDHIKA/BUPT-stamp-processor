#!/usr/bin/env python3
"""sram_audit.py — audit SRAM usage across the RTL and the synthesis lists.

Checks (exit code 1 on any failure):
  1. Every sram_2r1w / ssp_bank / mobol_sram_* instantiation in rtl/ is
     listed, so a new SRAM consumer can't sneak in unnoticed.
  2. Each synthesis file list in synth/filelists/ pulls in EXACTLY ONE
     implementation of module sram_2r1w (behavioral or macro wrapper, never
     both, never neither-when-instantiated).
  3. No non-blackboxed behavioral memory big enough to be a macro
     (>= 2 KB) hides in a file a synthesis list uses: those would silently
     synthesize into a sea of flip-flops.

Usage: python3 sram_audit.py   (from anywhere; paths are repo-relative)
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # repo root
RTL = ROOT / "rtl"
FILELISTS = ROOT / "synth" / "filelists"

# module_name -> instance regex ("module_name #(...)? inst_name (")
INST_RE = re.compile(
    r"^\s*(sram_2r1w|ssp_bank|mobol_sram_\w+)\s*(?:#\s*\(|\w+\s*\()",
    re.MULTILINE)
MODULE_RE = re.compile(r"^\s*module\s+(\w+)", re.MULTILINE)
# unpacked array declaration: logic [W-1:0] name [DEPTH];
MEM_RE = re.compile(
    r"^\s*(?:logic|reg)\s*\[(\d+):0\]\s*(\w+)\s*\[(?:0:)?(\d+)\]",
    re.MULTILINE)

BIG_MEM_BITS = 8 * 2048          # >= 2 KB should be a macro, not flops


def scan_instances():
    print("── SRAM instantiations in rtl/ ──")
    for f in sorted(RTL.rglob("*.sv")):
        for m in INST_RE.finditer(f.read_text()):
            line = f.read_text()[: m.start()].count("\n") + 1
            print(f"  {f.relative_to(ROOT)}:{line}  {m.group(1)}")


def check_filelists() -> bool:
    print("── synthesis file lists ──")
    ok = True
    impls = {"rtl/sram/sram_2r1w.sv": "behavioral",
             "rtl/sram/sram_2r1w_macro.sv": "macro-backed"}
    for fl in sorted(FILELISTS.glob("*.f")):
        files = [l.strip() for l in fl.read_text().splitlines()
                 if l.strip() and not l.strip().startswith("#")]
        used = [impls[f] for f in files if f in impls]
        text = "".join((ROOT / f).read_text() for f in files)
        wants_sram = bool(re.search(r"^\s*sram_2r1w\s*#", text, re.MULTILINE))
        if wants_sram and len(used) != 1:
            print(f"  FAIL {fl.name}: sram_2r1w instantiated but "
                  f"{len(used)} implementations listed ({used})")
            ok = False
        else:
            print(f"  ok   {fl.name}: sram_2r1w impl = "
                  f"{used[0] if used else 'n/a (not instantiated)'}")
    return ok


def check_big_mems() -> bool:
    print("── behavioral memories in synthesis file lists ──")
    ok = True
    seen = set()
    for fl in FILELISTS.glob("*.f"):
        for l in fl.read_text().splitlines():
            l = l.strip()
            if l and not l.startswith("#"):
                seen.add(l)
    for rel in sorted(seen):
        f = ROOT / rel
        for m in MEM_RE.finditer(f.read_text()):
            width, name, depth = int(m.group(1)) + 1, m.group(2), int(m.group(3))
            bits = width * depth
            if bits >= BIG_MEM_BITS:
                print(f"  FAIL {rel}: memory '{name}' = {bits/8/1024:.0f} KB "
                      f"would synthesize to flops — use a macro")
                ok = False
    if ok:
        print("  ok   no un-macroed memory >= 2 KB in any synthesis list")
    return ok


if __name__ == "__main__":
    scan_instances()
    good = check_filelists() & check_big_mems()
    sys.exit(0 if good else 1)
