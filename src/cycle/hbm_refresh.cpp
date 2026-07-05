/// @file hbm_refresh.cpp
/// @brief All-bank refresh manager for rank-less DRAM devices (HBM-class).
///
/// Ramulator2's stock "AllBank" refresh manager iterates over the "rank"
/// level, which HBM2/HBM3 do not have — with those devices it silently
/// issues NO refreshes at all, making every simulation optimistic.
/// This implementation registers "AllBankNoRank" into the Ramulator2
/// factory (we compile Ramulator2 from source into the same image, so
/// static registration works) and issues one REFab per channel every
/// nREFI controller cycles, per the HBM3 command scope table.

#include <vector>

#include "base/base.h"
#include "dram_controller/controller.h"
#include "dram_controller/refresh.h"

namespace Ramulator {

class AllBankRefreshNoRank : public IRefreshManager, public Implementation {
    RAMULATOR_REGISTER_IMPLEMENTATION(IRefreshManager, AllBankRefreshNoRank,
        "AllBankNoRank", "All-bank refresh for rank-less (HBM-class) DRAMs.")

  private:
    Clk_t m_clk = 0;
    IDRAM* m_dram = nullptr;
    IDRAMController* m_ctrl = nullptr;

    int m_dram_org_levels = -1;
    int m_nrefi = -1;
    int m_ref_req_id = -1;
    Clk_t m_next_refresh_cycle = -1;

  public:
    void init() override {
        m_ctrl = cast_parent<IDRAMController>();
    }

    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
        m_dram = m_ctrl->m_dram;
        m_dram_org_levels = m_dram->m_levels.size();
        m_nrefi = m_dram->m_timing_vals("nREFI");
        m_ref_req_id = m_dram->m_requests("all-bank-refresh");
        m_next_refresh_cycle = m_nrefi;
    }

    void tick() override {
        m_clk++;
        if (m_clk == m_next_refresh_cycle) {
            m_next_refresh_cycle += m_nrefi;
            // REFab has channel scope: one refresh request per channel
            // controller, all banks of the channel refresh together.
            std::vector<int> addr_vec(m_dram_org_levels, -1);
            addr_vec[0] = m_ctrl->m_channel_id;
            Request req(addr_vec, m_ref_req_id);
            if (!m_ctrl->priority_send(req)) {
                throw std::runtime_error("AllBankNoRank: failed to send refresh!");
            }
        }
    }
};

} // namespace Ramulator
