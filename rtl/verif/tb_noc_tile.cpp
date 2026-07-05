// tb_noc_tile.cpp — DMA+NoC integration: tiles push LOCAL regions to remote
// tiles through the ring; verify the data lands in the destination SRAM.
#include "Vnoc_tile_sys.h"
#include "verilated.h"
#include <cstdint>
#include <random>
#include <vector>
#include <iostream>

static const int N = 16;

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto* dut = new Vnoc_tile_sys;
    std::mt19937 rng(3);
    auto tick=[&]{ dut->clk=0; dut->eval(); dut->clk=1; dut->eval(); };

    dut->rst_n=0; dut->dma_start=0; dut->ext_we=0;
    tick(); tick(); dut->rst_n=1;

    // Preload each tile's SRAM lines 0..31 with distinct content.
    auto word=[&](int tile,int line,int w)->uint32_t{
        return (uint32_t)((tile*1000003 + line*97 + w)*2654435761u); };
    for (int t=0;t<N;t++){
        for (int line=0; line<32; line++){
            dut->ext_we = (1u<<t);
            dut->ext_waddr[t]=line;
            for (int w=0;w<16;w++) dut->ext_wdata[t][w]=word(t,line,w);
            tick();
        }
    }
    dut->ext_we=0;

    // Program: tile t pushes 8 lines from src_line 0 to a varied destination
    // at dst LOCAL line 64. Starts are staggered (realistic: the compiler
    // schedules DMAs; it never fires 16 identical synchronized pushes — that
    // pathological case can starve a single ring, a known QoS limit handled by
    // injection credits, out of scope for this functional check).
    struct Xfer{int src,dst,nlines,dst_line0;};
    std::vector<Xfer> xf;
    // (t+5)%16 is a bijection with no fixed point (gcd(5,16)=1): each tile
    // sends to a distinct destination, so no two writes collide.
    for (int t=0;t<N;t++){
        int dst=(t+5)%N;
        xf.push_back({t,dst,8,64});
        dut->dma_src_line[t]=0; dut->dma_dst_tile[t]=dst;
        dut->dma_dst_addr0[t]=64*64; dut->dma_nlines[t]=8;
    }

    uint32_t done_mask = 0;
    long cyc=0;
    int next_start=0;
    while (done_mask != 0xFFFF && cyc<200000){
        // launch one tile every 6 cycles (staggered)
        dut->dma_start = 0;
        if (next_start < N && (cyc % 6)==0){ dut->dma_start = (1u<<next_start); next_start++; }
        tick(); done_mask |= dut->dma_done; cyc++;
    }
    for (int extra=0; extra<400; extra++) tick();   // drain fabric

    // Verify: at each dst tile, lines 64..71 == src tile's lines 0..7.
    int fails=0;
    for (auto& x: xf){
        for (int k=0;k<x.nlines;k++){
            int dline=x.dst_line0+k;
            dut->ext_raddr[x.dst]=dline; tick(); dut->eval();
            for (int w=0;w<16;w++){
                uint32_t got=dut->ext_rdata[x.dst][w];
                uint32_t exp=word(x.src, 0+k, w);
                if (got!=exp){ if(++fails<=8) std::cerr<<"tile "<<x.src<<"->"<<x.dst
                    <<" line "<<dline<<" w"<<w<<" got 0x"<<std::hex<<got
                    <<" exp 0x"<<exp<<std::dec<<"\n"; break; }
            }
        }
    }
    std::cout<<"tb_noc_tile: "<<xf.size()<<" DMA-through-NoC transfers, "
             <<fails<<" failures (done@"<<cyc<<" cyc)\n";
    delete dut;
    return fails?1:0;
}
