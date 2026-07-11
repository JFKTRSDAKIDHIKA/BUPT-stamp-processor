#!/usr/bin/env python3
"""Generate + build one structural variant of the MOBOL chip.

Input: (num_tiles, tiles_per_group, shared_mb, topology).
Output: dse_bo/builds/t{N}_g{G}_s{MB}_{topo}/ containing
  - mobol_arch.yaml   (baseline YAML with the structural: block patched —
                       the same file drives the C++ ASSERT and the Python
                       compiler via MOBOL_ARCH_YAML)
  - build/            (independent CMake build: mobol_sim, trace_sim, tests)

Exit code 0 and a final line "VARIANT_READY <dir>" on success.

  python3 gen_variant.py --num-tiles 32 --tiles-per-group 8 \
      --shared-mb 16 --topology mesh [--jobs 16] [--force]
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RAMULATOR2_DIR = os.environ.get("RAMULATOR2_DIR", "/home/jiashuao/ramulator2-classic")

VALID = dict(num_tiles=(8, 16, 32), tiles_per_group=(2, 4, 8),
             shared_mb=(4, 8, 16), topology=("ring", "mesh", "torus"))


def variant_name(nt, tpg, smb, topo):
    return f"t{nt}_g{tpg}_s{smb}_{topo}"


def check_feasible(nt, tpg, smb, topo):
    """Mirror of the C++ static_asserts + derived constraints (§2)."""
    if nt not in VALID["num_tiles"]:
        return f"num_tiles {nt} not in {VALID['num_tiles']}"
    if tpg not in VALID["tiles_per_group"]:
        return f"tiles_per_group {tpg} not in {VALID['tiles_per_group']}"
    if smb not in VALID["shared_mb"]:
        return f"shared_mb {smb} not in {VALID['shared_mb']}"
    if topo not in VALID["topology"]:
        return f"topology {topo} not in {VALID['topology']}"
    if nt % tpg != 0:
        return f"num_tiles {nt} not divisible by tiles_per_group {tpg}"
    nb = nt // tpg
    if nb < 1:
        return "num_banks < 1"
    # 40-bit layout: bank selector [36 : 37-max(2,clog2(nb))] must sit above
    # the shared offset field (20 + clog2(smb) bits).
    clog2 = lambda v: (v - 1).bit_length()
    bank_lo = 37 - max(2, clog2(nb))
    off_bits = 20 + clog2(smb)
    if bank_lo < off_bits:
        return (f"address conflict: bank selector LSB {bank_lo} overlaps "
                f"{off_bits}-bit shared offset")
    tile_lo = 37 - max(4, clog2(nt))
    if tile_lo < 18:
        return "address conflict: tile selector overlaps 18-bit local offset"
    return None


def patch_yaml(src, dst, nt, tpg, smb, topo):
    """Rewrite the structural: keys of the baseline YAML (text-level patch
    keeps comments/format so both loaders see the identical file shape)."""
    out = []
    for ln in open(src):
        s = ln.strip()
        if s.startswith("num_tiles:"):
            ln = f"  num_tiles: {nt}\n"
        elif s.startswith("num_banks:"):
            ln = f"  num_banks: {nt // tpg}\n"
        elif s.startswith("tiles_per_group:"):
            ln = f"  tiles_per_group: {tpg}\n"
        elif s.startswith("shared_mb:"):
            ln = f"  shared_mb: {smb}\n"
        elif s.startswith("topology:"):
            ln = f"  topology: {topo}\n"
        elif s.startswith("ramulator_config:") and "config/" in ln:
            # keep absolute so the variant YAML works from any CWD
            ln = ln.replace("config/ramulator_3d_dram.yaml",
                            os.path.join(ROOT, "config/ramulator_3d_dram.yaml"))
        out.append(ln)
    with open(dst, "w") as f:
        f.writelines(out)


def build_variant(nt, tpg, smb, topo, jobs=16, force=False):
    """Returns (variant_dir, None) on success, (None, reason) on failure."""
    err = check_feasible(nt, tpg, smb, topo)
    if err:
        return None, "INFEASIBLE: " + err

    name = variant_name(nt, tpg, smb, topo)
    vdir = os.path.join(HERE, "builds", name)
    bdir = os.path.join(vdir, "build")
    yaml_path = os.path.join(vdir, "mobol_arch.yaml")
    stamp = os.path.join(vdir, ".build_ok")

    if os.path.exists(stamp) and not force:
        return vdir, None  # cached build

    os.makedirs(vdir, exist_ok=True)
    # Serialize concurrent builders of the SAME variant (BO + random
    # baseline share the build cache).
    import fcntl
    lockf = open(os.path.join(vdir, ".lock"), "w")
    fcntl.flock(lockf, fcntl.LOCK_EX)
    try:
        if os.path.exists(stamp) and not force:
            return vdir, None  # another process finished it while we waited
        return _do_build(vdir, yaml_path, bdir, stamp, nt, tpg, smb, topo, jobs)
    finally:
        fcntl.flock(lockf, fcntl.LOCK_UN)
        lockf.close()


def _do_build(vdir, yaml_path, bdir, stamp, nt, tpg, smb, topo, jobs):
    patch_yaml(os.path.join(ROOT, "config", "mobol_arch.yaml"),
               yaml_path, nt, tpg, smb, topo)

    cfg = ["cmake", "-S", ROOT, "-B", bdir,
           "-DCMAKE_BUILD_TYPE=Release",
           f"-DRAMULATOR2_DIR={RAMULATOR2_DIR}",
           f"-DMOBOL_NUM_TILES={nt}",
           f"-DMOBOL_TILES_PER_GROUP={tpg}",
           f"-DMOBOL_SHARED_MB={smb}",
           f"-DMOBOL_TOPOLOGY={topo}"]
    log = os.path.join(vdir, "build.log")
    with open(log, "w") as lf:
        r = subprocess.run(cfg, stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            return None, f"CMAKE_FAILED (see {log})"
        r = subprocess.run(["cmake", "--build", bdir, f"-j{jobs}"],
                           stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            return None, f"BUILD_FAILED (see {log})"
    open(stamp, "w").write("ok\n")
    return vdir, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tiles", type=int, required=True)
    ap.add_argument("--tiles-per-group", type=int, required=True)
    ap.add_argument("--shared-mb", type=int, required=True)
    ap.add_argument("--topology", default="ring")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    vdir, err = build_variant(a.num_tiles, a.tiles_per_group, a.shared_mb,
                              a.topology, a.jobs, a.force)
    if err:
        print(err)
        sys.exit(1)
    print(f"VARIANT_READY {vdir}")


if __name__ == "__main__":
    main()
