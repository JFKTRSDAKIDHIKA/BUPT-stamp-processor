// tb_mxu.cpp — Verilator testbench for mxu16, bit-exact vs blockops::mxu.
// Drives random f16 A,B blocks (and f32 A for MXU_F32F16) through the RTL and
// compares the 256 f32 outputs to the C golden the simulator uses.
#include "Vmxu16.h"
#include "verilated.h"
#include "cycle/blockops.h"
#include "common/f16.h"
#include <cstdint>
#include <cstring>
#include <random>
#include <iostream>

using mobol::f16;
namespace bo = mobol::cycle::blockops;

static uint32_t f2u(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}
static float u2f(uint32_t u){float f;std::memcpy(&f,&u,4);return f;}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto* dut = new Vmxu16;
    std::mt19937 rng(2024);
    int fails = 0, cases = 0;

    auto tick = [&]{ dut->clk=0; dut->eval(); dut->clk=1; dut->eval(); };

    // Reset.
    dut->rst_n = 0; dut->start = 0; dut->acc = 0;
    tick(); tick();
    dut->rst_n = 1;

    auto rnd_f16 = [&]()->uint16_t {
        // f16 in ~[-2,2], no inf/nan
        float v = ((int)(rng()%8000)-4000)/2000.0f;
        return f16(v).bits;
    };

    for (int c = 0; c < 300; c++) {
        bool a_f32 = (c & 1);
        bool acc  = (c % 3 == 0);
        // Build operands.
        uint8_t A16[512], B16[512];   // f16 blocks
        float   A32[256];             // f32 A (for a_f32)
        float   Cin[256];
        for (int i = 0; i < 256; i++) {
            uint16_t ah = rnd_f16(), bh = rnd_f16();
            std::memcpy(&A16[i*2], &ah, 2);
            std::memcpy(&B16[i*2], &bh, 2);
            A32[i] = a_f32 ? (((int)(rng()%8000)-4000)/1500.0f) : f16(ah).to_f32();
            Cin[i] = acc ? (((int)(rng()%8000)-4000)/1000.0f) : 0.0f;
        }
        // C golden: blockops::mxu(A, a_is_f32, B, C, acc).
        float Cgold[256];
        if (acc) std::memcpy(Cgold, Cin, sizeof(Cgold));
        const uint8_t* Aptr = a_f32 ? reinterpret_cast<uint8_t*>(A32) : A16;
        bo::mxu(Aptr, a_f32, B16, Cgold, acc);

        // Drive RTL: a_f32 bus is the 256x32 A (promoted f16->f32 if !a_f32).
        for (int i = 0; i < 256; i++) {
            uint32_t au = a_f32 ? f2u(A32[i]) : f2u(f16(*(uint16_t*)&A16[i*2]).to_f32());
            dut->a_f32[i] = au;
            dut->b_f16[i] = *(uint16_t*)&B16[i*2];
            dut->cin_f32[i] = f2u(Cin[i]);
        }
        dut->acc = acc;
        dut->start = 1; tick(); dut->start = 0;
        int guard = 0;
        while (!dut->done && guard++ < 40) tick();
        if (!dut->done) { std::cerr << "case "<<c<<" never done\n"; fails++; continue; }

        for (int i = 0; i < 256; i++) {
            uint32_t got = dut->cout_f32[i];
            uint32_t exp = f2u(Cgold[i]);
            if (got != exp) {
                if (++fails <= 10)
                    std::cerr << "case "<<c<<" a_f32="<<a_f32<<" acc="<<acc
                              <<" [i="<<i<<"] got=0x"<<std::hex<<got
                              <<" exp=0x"<<exp<<std::dec
                              <<" ("<<u2f(got)<<" vs "<<u2f(exp)<<")\n";
            }
        }
        cases++;
    }
    std::cout << "tb_mxu: " << cases << " matmul cases, " << fails << " failures\n";
    delete dut;
    return fails ? 1 : 0;
}
