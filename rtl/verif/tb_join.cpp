// tb_join.cpp — directed check of the release/acquire join counter.
#include "Vjoin_ctr.h"
#include "verilated.h"
#include <iostream>

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto* dut = new Vjoin_ctr;
    int fails = 0;
    auto step=[&](bool rv,int rt,bool av,int at,int ar)->bool{
        dut->rel_valid=rv; dut->rel_tag=rt;
        dut->acq_valid=av; dut->acq_tag=at; dut->acq_arity=ar;
        dut->clk=0; dut->eval();
        bool pass = dut->acq_pass;          // combinational
        dut->clk=1; dut->eval();
        return pass;
    };
    auto chk=[&](const char*w,bool got,bool exp){
        if(got!=exp){std::cerr<<"FAIL "<<w<<" got "<<got<<" exp "<<exp<<"\n";fails++;}
    };
    dut->rst_n=0; step(0,0,0,0,0); step(0,0,0,0,0); dut->rst_n=1;

    // 1. arity-2 join on tag 5: two releases, then acquire passes & consumes 2.
    step(1,5,0,0,0);                                   // cnt[5]=1
    chk("acq before 2nd release", step(0,0,1,5,2), false);
    step(1,5,0,0,0);                                   // cnt[5]=2
    chk("acq after 2 releases",   step(0,0,1,5,2), true);   // consumes 2 -> 0
    chk("acq again (empty)",      step(0,0,1,5,2), false);

    // 2. same-cycle release+acquire (release satisfies the acquire).
    step(1,7,0,0,0);                                   // cnt[7]=1
    chk("same-cycle rel+acq arity2", step(1,7,1,7,2), true); // 1+1>=2, consume2 ->0
    chk("tag7 now empty",            step(0,0,1,7,1), false);

    // 3. arity-16 join (FFN reduction): 16 releases then acquire.
    for(int i=0;i<15;i++) step(1,9,0,0,0);
    chk("arity16 with 15",  step(0,0,1,9,16), false);
    step(1,9,0,0,0);                                   // 16th
    chk("arity16 with 16",  step(0,0,1,9,16), true);

    // 4. independent tags don't interfere.
    step(1,3,0,0,0); step(1,4,0,0,0);
    chk("tag3 arity1", step(0,0,1,3,1), true);
    chk("tag4 arity1", step(0,0,1,4,1), true);

    std::cout<<"tb_join: "<<(fails?"FAILED":"PASSED")<<" ("<<fails<<" failures)\n";
    delete dut;
    return fails?1:0;
}
