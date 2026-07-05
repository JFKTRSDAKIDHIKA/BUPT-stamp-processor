// tb_prefetch.cpp — weight prefetch engine: verify a 2D strided DRAM->bank
// copy transfers the correct bytes. Models DRAM (data = f(addr)) and bank SRAM
// (a map of writes), then checks every destination beat matches its source.
#include "Vprefetch_engine.h"
#include "verilated.h"
#include <cstdint>
#include <map>
#include <deque>
#include <random>
#include <iostream>

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto* dut = new Vprefetch_engine;
    std::mt19937 rng(5);
    auto tick=[&]{ dut->clk=0; dut->eval(); dut->clk=1; dut->eval(); };

    // DRAM content model: byte at address A = (A*2654435761) low 8 bits.
    auto dram_beat=[&](uint64_t addr, uint32_t* out){    // 512b = 16 words
        for (int w=0; w<16; w++){ uint32_t v=0;
            for (int b=0;b<4;b++){ uint64_t a=addr + w*4 + b;
                v |= (uint32_t)((a*2654435761ull)&0xFF) << (b*8); }
            out[w]=v; }
    };

    dut->rst_n=0; dut->start=0; dut->rd_gnt=0; dut->rd_data_valid=0;
    tick(); tick(); dut->rst_n=1;

    int fails=0, cases=0;
    for (int c=0;c<50;c++){
        int rows = 1 + rng()%16;
        int bpr  = 1 + rng()%4;                 // beats per row
        int row_bytes = bpr*64;
        int64_t src_stride = (int64_t)row_bytes + 64*(rng()%3);  // gaps allowed
        int64_t dst_stride = 64*bpr;             // packed dest
        uint64_t src_base = 0x1000 + 64*(rng()%64);
        uint64_t dst_base = 0x800000 + 64*(rng()%64);

        // reference: dst beat r (0..rows*bpr-1) source addr
        std::map<uint64_t, std::array<uint32_t,16>> sram;
        std::deque<uint64_t> pending_src;        // in-order read addresses

        dut->src_base=src_base; dut->dst_base=dst_base;
        dut->rows=rows; dut->row_bytes=row_bytes;
        dut->src_stride=(int32_t)src_stride; dut->dst_stride=(int32_t)dst_stride;
        dut->start=1; tick(); dut->start=0;

        int total = rows*bpr;
        int guard=0;
        // simple model: grant every read; return data 2 cycles later in order.
        std::deque<uint64_t> inflight;
        int returned=0;
        while (!dut->done && guard++<10000){
            dut->clk=0; dut->eval();
            // grant reads
            dut->rd_gnt = dut->rd_req;
            // schedule data return for a granted read (this cycle addr)
            uint64_t raddr = 0; bool has=false;
            if (dut->rd_req){ raddr = dut->rd_addr; has=true; }
            // present a returned beat if any inflight are "ready" (>=2 old)
            dut->rd_data_valid=0;
            if (!inflight.empty()){
                uint64_t sa = inflight.front();
                uint32_t beat[16]; dram_beat(sa, beat);
                for (int w=0;w<16;w++) dut->rd_data[w]=beat[w];
                dut->rd_data_valid=1;
            }
            dut->eval();
            // capture write
            if (dut->wr_en){
                std::array<uint32_t,16> a; for(int w=0;w<16;w++)a[w]=dut->wr_data[w];
                sram[dut->wr_addr]=a;
            }
            bool grant = dut->rd_req && dut->rd_gnt;
            bool consume = dut->rd_data_valid;
            dut->clk=1; dut->eval();
            if (grant && has) inflight.push_back(raddr);
            if (consume && !inflight.empty()){ inflight.pop_front(); returned++; }
        }
        if (!dut->done){ std::cerr<<"case "<<c<<" never done\n"; fails++; continue; }
        // verify: each dst beat matches its source
        for (int r=0;r<rows;r++) for (int k=0;k<bpr;k++){
            uint64_t sa = src_base + (uint64_t)((int64_t)r*src_stride) + k*64;
            uint64_t da = dst_base + (uint64_t)((int64_t)r*dst_stride) + k*64;
            uint32_t exp[16]; dram_beat(sa, exp);
            auto it=sram.find(da);
            if (it==sram.end()){ if(++fails<=6) std::cerr<<"case "<<c<<" missing dst 0x"
                <<std::hex<<da<<std::dec<<"\n"; continue; }
            for (int w=0;w<16;w++) if (it->second[w]!=exp[w]){
                if(++fails<=6) std::cerr<<"case "<<c<<" beat("<<r<<","<<k<<") w"<<w
                    <<" got 0x"<<std::hex<<it->second[w]<<" exp 0x"<<exp[w]<<std::dec<<"\n";
                break; }
        }
        cases++;
    }
    std::cout<<"tb_prefetch: "<<cases<<" copy cases, "<<fails<<" failures\n";
    delete dut;
    return fails?1:0;
}
