// tb_router.cpp — functional check of the 16-node ring NoC via noc_test_top.
// Every injected packet must be ejected exactly once at its destination.
#include "Vnoc_test_top.h"
#include "verilated.h"
#include <cstdint>
#include <deque>
#include <map>
#include <random>
#include <iostream>

static const int N = 16;
struct Pkt { int src, dst; bool is_resp; uint32_t id; };

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto* dut = new Vnoc_test_top;
    std::mt19937 rng(7);
    auto tick = [&]{ dut->clk=0; dut->eval(); dut->clk=1; dut->eval(); };

    dut->rst_n = 0; dut->inj_valid = 0;
    tick(); tick();
    dut->rst_n = 1;

    std::deque<Pkt> q[N];
    std::map<uint32_t,Pkt> outstanding;
    uint32_t id = 1;
    const int TOTAL = 6000;
    for (int i=0;i<TOTAL;i++){
        Pkt p; p.src=rng()%N; p.dst=rng()%N; if(p.dst==p.src)p.dst=(p.dst+1)%N;
        p.is_resp = rng()&1; p.id=id++;
        q[p.src].push_back(p); outstanding[p.id]=p;
    }

    int delivered=0, mis=0; long cyc=0;
    while (delivered<TOTAL && cyc<2000000){
        // present injection candidates
        uint32_t vbits=0, rbits=0;
        for (int t=0;t<N;t++){
            if(!q[t].empty()){
                Pkt&p=q[t].front();
                vbits |= (1u<<t);
                if(p.is_resp) rbits |= (1u<<t);
                dut->inj_dst[t]=p.dst; dut->inj_src[t]=p.src; dut->inj_id[t]=p.id;
            }
        }
        dut->inj_valid = vbits; dut->inj_is_resp = rbits;
        dut->clk=0; dut->eval();                     // settle comb
        // sample ejects from both ring directions
        auto sample=[&](int t, bool valid, uint32_t eid, int edst){
            if(!valid) return;
            auto it=outstanding.find(eid);
            if(it==outstanding.end()||it->second.dst!=edst||edst!=t) mis++;
            else { outstanding.erase(it); delivered++; }
        };
        for (int t=0;t<N;t++){
            sample(t,(dut->ej_valid_cw>>t)&1, dut->ej_id_cw[t], dut->ej_dst_cw[t]);
            sample(t,(dut->ej_valid_ccw>>t)&1,dut->ej_id_ccw[t],dut->ej_dst_ccw[t]);
        }
        // pop accepted injections
        for (int t=0;t<N;t++)
            if(!q[t].empty() && ((dut->inj_ready>>t)&1)) q[t].pop_front();
        dut->clk=1; dut->eval();
        cyc++;
    }
    std::cout<<"tb_router: delivered "<<delivered<<"/"<<TOTAL
             <<", misdeliveries "<<mis<<", cycles "<<cyc<<"\n";
    delete dut;
    return (delivered==TOTAL && mis==0)?0:1;
}
