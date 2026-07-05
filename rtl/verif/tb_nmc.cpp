// tb_nmc.cpp — NMC reduce engine, bit-exact vs blockops::add_f32 chained.
#include "Vnmc_engine.h"
#include "verilated.h"
#include "cycle/blockops.h"
#include <cstdint>
#include <cstring>
#include <random>
#include <iostream>

namespace bo = mobol::cycle::blockops;
static uint32_t f2u(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}
static float u2f(uint32_t u){float f;std::memcpy(&f,&u,4);return f;}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto* dut = new Vnmc_engine;
    std::mt19937 rng(99);
    auto tick=[&]{ dut->clk=0; dut->eval(); dut->clk=1; dut->eval(); };
    dut->rst_n=0; dut->start=0; dut->blk_valid=0;
    tick(); tick(); dut->rst_n=1;

    int fails=0, cases=0;
    for (int c=0;c<200;c++){
        int count = 1 + rng()%16;
        // random blocks
        std::vector<std::vector<float>> blocks(count, std::vector<float>(256));
        for (auto& b: blocks) for (auto& v: b) v = ((int)(rng()%20000)-10000)/1000.0f;
        // golden: acc = block0; acc = add_f32(acc, block_j)
        float acc[256]; std::memcpy(acc, blocks[0].data(), sizeof(acc));
        for (int j=1;j<count;j++) bo::add_f32(acc, blocks[j].data(), acc);

        dut->op_ln=0; dut->count=count; dut->start=1; tick(); dut->start=0;
        int fed=0, guard=0;
        while (!dut->done && guard++<400){
            dut->clk=0; dut->eval();                    // settle: blk_ready
            if (dut->blk_ready && fed<count){
                for (int i=0;i<256;i++) dut->blk_in[i]=f2u(blocks[fed][i]);
                dut->blk_valid=1;
            } else dut->blk_valid=0;
            dut->eval();
            bool accept = dut->blk_ready && dut->blk_valid && (fed<count);
            dut->clk=1; dut->eval();                     // posedge consumes
            if (accept) fed++;
        }
        dut->blk_valid=0;
        if (!dut->done){ std::cerr<<"case "<<c<<" never done (fed "<<fed<<")\n"; fails++; continue; }
        for (int i=0;i<256;i++){
            uint32_t got=dut->res_f32[i], exp=f2u(acc[i]);
            if (got!=exp){ if(++fails<=8) std::cerr<<"case "<<c<<" count="<<count
                <<" [i="<<i<<"] got=0x"<<std::hex<<got<<" exp=0x"<<exp<<std::dec
                <<" ("<<u2f(got)<<" vs "<<u2f(exp)<<")\n"; }
        }
        cases++;
    }
    std::cout<<"tb_nmc: "<<cases<<" reduce cases, "<<fails<<" failures\n";
    delete dut;
    return fails?1:0;
}
