"""edgc — Edge-LLM AI compiler for the MOBOL 3D accelerator.

Pipeline: ModelConfig (metadata) -> lowering + DSE -> tile-ISA trace +
DRAM image -> cycle-accurate simulator. Correctness is pinned by a golden
ISA interpreter (golden==sim) and a high-level reference forward
(golden==reference), both in bit-exact tile numerics.
"""
from .model import ModelConfig, gen_weights, reference_forward
from .compile import Compiler, Sched
from .golden import Golden
