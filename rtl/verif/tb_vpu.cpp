// tb_vpu.cpp — VPU: elementwise ops bit-exact vs blockops; SILU/RMSNORM within
// tolerance vs a float reference.
#include "Vvpu16.h"
#include "verilated.h"
#include "cycle/blockops.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>
#include <iostream>

namespace bo = mobol::cycle::blockops;
static uint32_t f2u(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}
static float u2f(uint32_t u){float f;std::memcpy(&f,&u,4);return f;}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto* dut = new Vvpu16;
    std::mt19937 rng(41);
    auto run=[&](int op, float scal, const float*A, const float*B, float*Y){
        for (int i=0;i<256;i++){ dut->a_in[i]=f2u(A[i]); dut->b_in[i]=f2u(B[i]); }
        dut->op=op; dut->scalar=f2u(scal);
        dut->start=1; dut->clk=0; dut->eval(); dut->clk=1; dut->eval(); dut->start=0;
        int g=0; while(!dut->done && g++<80){ dut->clk=0; dut->eval(); dut->clk=1; dut->eval(); }
        dut->clk=0; dut->eval();
        for (int i=0;i<256;i++) Y[i]=u2f(dut->y_out[i]);
    };
    dut->rst_n=0; dut->start=0; dut->clk=0; dut->eval(); dut->clk=1; dut->eval();
    dut->clk=0; dut->eval(); dut->clk=1; dut->eval(); dut->rst_n=1;

    int fails=0;
    double silu_max=0, rms_max=0;
    for (int c=0;c<60;c++){
        float A[256],B[256],Y[256];
        for (int i=0;i<256;i++){ A[i]=((int)(rng()%16000)-8000)/1500.0f;
                                 B[i]=((int)(rng()%16000)-8000)/1500.0f; }
        // ADD bit-exact
        run(0,0,A,B,Y);
        { float g[256]; bo::add_f32(A,B,g);
          for(int i=0;i<256;i++) if(f2u(Y[i])!=f2u(g[i])){ if(++fails<=5)
            std::cerr<<"ADD c"<<c<<" i"<<i<<"\n"; break; } }
        // MUL bit-exact
        run(1,0,A,B,Y);
        { float g[256]; bo::mul_f32(A,B,g);
          for(int i=0;i<256;i++) if(f2u(Y[i])!=f2u(g[i])){ if(++fails<=5)
            std::cerr<<"MUL c"<<c<<" i"<<i<<"\n"; break; } }
        // SILU tolerance
        run(3,0,A,B,Y);
        for(int i=0;i<256;i++){ double ref=(double)A[i]/(1.0+std::exp(-(double)A[i]));
          double e=std::fabs(Y[i]-ref)/(std::fabs(ref)+1e-6);
          silu_max=std::max(silu_max,e);
          if(e>2e-3){ if(++fails<=5) std::cerr<<"SILU c"<<c<<" i"<<i<<" got"<<Y[i]<<" ref"<<ref<<"\n"; } }
        // RMSNORM tolerance (eps=1e-5), per 16-wide row
        run(4,1e-5f,A,B,Y);
        for(int r=0;r<16;r++){ double ss=0; for(int k=0;k<16;k++){double v=A[r*16+k]; ss+=v*v;}
          double inv=1.0/std::sqrt(ss/16.0+1e-5);
          for(int k=0;k<16;k++){ double ref=A[r*16+k]*inv;
            double e=std::fabs(Y[r*16+k]-ref)/(std::fabs(ref)+1e-6);
            rms_max=std::max(rms_max,e);
            if(e>1e-3){ if(++fails<=5) std::cerr<<"RMS c"<<c<<" r"<<r<<" k"<<k
              <<" got"<<Y[r*16+k]<<" ref"<<ref<<"\n"; } } }
    }
    std::cout<<"tb_vpu: ADD/MUL bit-exact; SILU max rel "<<silu_max
             <<", RMSNORM max rel "<<rms_max<<" ("<<fails<<" failures)\n";
    delete dut;
    return fails?1:0;
}
