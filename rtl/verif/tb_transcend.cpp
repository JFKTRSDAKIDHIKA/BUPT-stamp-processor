// tb_transcend.cpp — tolerance check of the synthesizable transcendental units
// vs libm. These are hardware approximations (Newton / minimax poly); the goal
// is functional correctness (tight relative error), not libm bit-exactness.
#include "Vtranscend_test_top.h"
#include "verilated.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>
#include <iostream>

static uint32_t f2u(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}
static float u2f(uint32_t u){float f;std::memcpy(&f,&u,4);return f;}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto* dut = new Vtranscend_test_top;
    std::mt19937 rng(17);
    double rs_max=0, rc_max=0, e2_max=0;
    int n=0, fails=0;

    for (int i=0;i<200000;i++){
        // rsqrt: x in [1e-6, 1e4]
        float rx = std::pow(10.0f, ((int)(rng()%10000))/1000.0f - 6.0f);
        dut->rsqrt_x = f2u(rx);
        // recip: x in [1e-4, 1e4]
        float cx = std::pow(10.0f, ((int)(rng()%8000))/1000.0f - 4.0f);
        dut->recip_x = f2u(cx);
        // exp2: x in [-30, 5]
        float ex = ((int)(rng()%35000))/1000.0f - 30.0f;
        dut->exp2_x = f2u(ex);
        dut->eval();

        float rs = u2f(dut->rsqrt_y), rc = u2f(dut->recip_y), e2 = u2f(dut->exp2_y);
        double rs_ref = 1.0/std::sqrt((double)rx);
        double rc_ref = 1.0/(double)cx;
        double e2_ref = std::pow(2.0,(double)ex);
        double rs_e = std::fabs(rs - rs_ref)/rs_ref;
        double rc_e = std::fabs(rc - rc_ref)/rc_ref;
        double e2_e = e2_ref>1e-30 ? std::fabs(e2 - e2_ref)/e2_ref : std::fabs(e2-e2_ref);
        rs_max=std::max(rs_max,rs_e); rc_max=std::max(rc_max,rc_e); e2_max=std::max(e2_max,e2_e);
        if (rs_e>1e-4){ if(++fails<=6) std::cerr<<"rsqrt x="<<rx<<" got="<<rs<<" ref="<<rs_ref<<"\n"; }
        if (rc_e>1e-4){ if(++fails<=6) std::cerr<<"recip x="<<cx<<" got="<<rc<<" ref="<<rc_ref<<"\n"; }
        if (e2_e>1e-4){ if(++fails<=6) std::cerr<<"exp2 x="<<ex<<" got="<<e2<<" ref="<<e2_ref<<"\n"; }
        n++;
    }
    std::cout<<"tb_transcend: "<<n<<" pts | max rel err  rsqrt="<<rs_max
             <<"  recip="<<rc_max<<"  exp2="<<e2_max<<"  ("<<fails<<" over-tol)\n";
    delete dut;
    return fails?1:0;
}
