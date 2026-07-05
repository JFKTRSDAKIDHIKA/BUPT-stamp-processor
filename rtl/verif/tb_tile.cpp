// tb_tile.cpp — integrated compute tile: preload A,B into LOCAL SRAM, run the
// matmul micro-op, read C back, compare bit-exact to blockops::mxu. Also
// checks the join counter release/acquire path through the tile.
#include "Vtile_top.h"
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

// SRAM line layout in the TB (must match tile_top).
static const int A_LINE0 = 0;    // 16 lines
static const int B_LINE0 = 16;   // 8 lines
static const int C_LINE0 = 24;   // 16 lines

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto* dut = new Vtile_top;
    std::mt19937 rng(31);
    auto tick=[&]{ dut->clk=0; dut->eval(); dut->clk=1; dut->eval(); };

    dut->rst_n=0; dut->start=0; dut->ext_we=0; dut->acc=0;
    dut->rel_valid=0; dut->acq_valid=0;
    tick(); tick(); dut->rst_n=1;

    int fails=0, cases=0;
    for (int c=0;c<40;c++){
        uint8_t A16[512], B16[512]; float A32[256];
        for (int i=0;i<256;i++){
            float av=((int)(rng()%8000)-4000)/2000.0f;
            float bv=((int)(rng()%8000)-4000)/2000.0f;
            A32[i]=av; uint16_t bh=f16(bv).bits;
            std::memcpy(&B16[i*2], &bh, 2);
        }
        // golden (A f32, B f16)
        float Cg[256];
        bo::mxu(reinterpret_cast<uint8_t*>(A32), true, B16, Cg, false);

        // preload A: 16 lines of 16 f32
        auto wr_line=[&](int line, const uint8_t* data){
            dut->ext_we=1; dut->ext_waddr=line;
            for (int w=0;w<16;w++){ uint32_t v; std::memcpy(&v,&data[w*4],4);
                dut->ext_wdata[w]=v; }
            tick();
        };
        for (int L=0;L<16;L++) wr_line(A_LINE0+L, reinterpret_cast<uint8_t*>(&A32[L*16]));
        // preload B: 8 lines of 32 f16
        for (int L=0;L<8;L++) wr_line(B_LINE0+L, &B16[L*64]);
        dut->ext_we=0;

        // run
        dut->a_line0=A_LINE0; dut->b_line0=B_LINE0; dut->c_line0=C_LINE0; dut->acc=0;
        dut->start=1; tick(); dut->start=0;
        int guard=0; while(!dut->done && guard++<200) tick();
        if (!dut->done){ std::cerr<<"case "<<c<<" tile never done\n"; fails++; continue; }
        tick();  // let last SRAM write settle

        // read C: 16 lines via backdoor read port (1-cycle latency)
        float Cr[256];
        for (int L=0;L<16;L++){
            dut->ext_raddr=C_LINE0+L; tick();   // present addr, next cycle data
            dut->eval();
            for (int w=0;w<16;w++){ uint32_t v=dut->ext_rdata[w];
                std::memcpy(&Cr[L*16+w], &v, 4); }
        }
        for (int i=0;i<256;i++){
            if (f2u(Cr[i])!=f2u(Cg[i])){ if(++fails<=8) std::cerr<<"case "<<c
                <<" C[i="<<i<<"] got="<<Cr[i]<<" exp="<<Cg[i]<<"\n"; }
        }
        cases++;
    }

    // join path through the tile: 2 releases then acquire arity2 passes.
    auto jstep=[&](bool rv,int rt,bool av,int at,int ar)->bool{
        dut->rel_valid=rv; dut->rel_tag=rt; dut->acq_valid=av;
        dut->acq_tag=at; dut->acq_arity=ar; dut->clk=0; dut->eval();
        bool p=dut->acq_pass; dut->clk=1; dut->eval(); return p; };
    jstep(1,3,0,0,0); bool p1=jstep(0,0,1,3,2);
    jstep(1,3,0,0,0); bool p2=jstep(0,0,1,3,2);
    if (p1 || !p2){ std::cerr<<"tile join path wrong ("<<p1<<","<<p2<<")\n"; fails++; }

    std::cout<<"tb_tile: "<<cases<<" matmul cases + join, "<<fails<<" failures\n";
    delete dut;
    return fails?1:0;
}
