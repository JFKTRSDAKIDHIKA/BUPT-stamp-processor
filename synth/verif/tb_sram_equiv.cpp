// tb_sram_equiv.cpp — random equivalence check: behavioral SRAM vs macros.
//
// Drives both pairs in sram_equiv_top with constrained-random traffic:
//   - addresses drawn from a small pool half the time, so read-during-write
//     and back-to-back write/read collisions happen constantly;
//   - SSP addresses hop across sub-macro boundaries (upper bits) to
//     exercise the ssp_bank decode + registered read mux.
// Any cycle with lsp_mismatch/ssp_mismatch after reset is a failure.
#include "Vsram_equiv_top.h"
#include "verilated.h"
#include <cstdlib>
#include <iostream>

static uint32_t rng_state = 0xC0FFEE42;
static uint32_t rnd() {                       // xorshift32, reproducible
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto* dut = new Vsram_equiv_top;

    // small address pools force constant collisions
    uint32_t lsp_pool[8], ssp_pool[8];
    for (int i = 0; i < 8; i++) {
        lsp_pool[i] = rnd() & 0xFFF;
        // spread the SSP pool over sub-macros 0,1,2,31 (17-bit line addr)
        uint32_t sub = (i < 6) ? (i % 3) : 31;
        ssp_pool[i] = (sub << 12) | (rnd() & 0xFFF);
    }
    auto lsp_addr = [&]{ return (rnd() & 1) ? lsp_pool[rnd() & 7]
                                            : (rnd() & 0xFFF); };
    auto ssp_addr = [&]{ return (rnd() & 1) ? ssp_pool[rnd() & 7]
                                            : (rnd() & 0x1FFFF); };

    const int CYCLES = 20000;
    int fails = 0;
    for (int cyc = 0; cyc < CYCLES; cyc++) {
        // new stimulus for this cycle
        dut->we     = rnd() & 1;
        dut->waddr  = lsp_addr();
        dut->raddr0 = lsp_addr();
        dut->raddr1 = lsp_addr();
        dut->swe     = rnd() & 1;
        dut->swaddr  = ssp_addr();
        dut->sraddr0 = ssp_addr();
        dut->sraddr1 = ssp_addr();
        for (int w = 0; w < 16; w++) {
            dut->wdata[w]  = rnd();
            dut->swdata[w] = rnd();
        }

        dut->clk = 0; dut->eval();
        dut->clk = 1; dut->eval();

        // let one cycle of read latency fill the output regs before checking
        if (cyc > 1 && (dut->lsp_mismatch || dut->ssp_mismatch)) {
            std::cerr << "FAIL cycle " << cyc
                      << " lsp=" << (int)dut->lsp_mismatch
                      << " ssp=" << (int)dut->ssp_mismatch << "\n";
            if (++fails > 5) break;
        }
    }

    std::cout << "tb_sram_equiv: " << (fails ? "FAILED" : "PASSED")
              << " (" << CYCLES << " random cycles, both pairs)\n";
    delete dut;
    return fails ? 1 : 0;
}
