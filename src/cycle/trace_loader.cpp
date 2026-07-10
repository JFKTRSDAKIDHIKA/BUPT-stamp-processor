/// @file trace_loader.cpp
/// @brief Text-ISA trace parser + binary image I/O.

#include "cycle/trace_loader.h"
#include "common/address.h"

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace mobol::cycle {

namespace {

[[noreturn]] void err(size_t line, const std::string& msg, const std::string& raw) {
    std::ostringstream os;
    os << "trace parse error, line " << line << ": " << msg << "\n  >> " << raw;
    throw std::runtime_error(os.str());
}

// Tokenize; supports key=value and bare tokens. '#' starts a comment.
std::vector<std::string> tokenize(const std::string& line) {
    std::string s = line;
    auto hash = s.find('#');
    if (hash != std::string::npos) s = s.substr(0, hash);
    std::vector<std::string> t;
    std::istringstream is(s);
    std::string w;
    while (is >> w) t.push_back(w);
    return t;
}

struct KV {
    std::unordered_map<std::string, std::string> m;
    void put(const std::string& tok, size_t line, const std::string& raw) {
        auto eq = tok.find('=');
        if (eq == std::string::npos) err(line, "expected key=value: " + tok, raw);
        m[tok.substr(0, eq)] = tok.substr(eq + 1);
    }
    bool has(const std::string& k) const { return m.count(k); }
    std::string s(const std::string& k, size_t line, const std::string& raw) const {
        auto it = m.find(k);
        if (it == m.end()) err(line, "missing field '" + k + "'", raw);
        return it->second;
    }
    uint64_t u(const std::string& k, size_t line, const std::string& raw,
               uint64_t def, bool required) const {
        auto it = m.find(k);
        if (it == m.end()) { if (required) err(line, "missing '" + k + "'", raw); return def; }
        return std::stoull(it->second, nullptr, 0);
    }
    int64_t i(const std::string& k, int64_t def) const {
        auto it = m.find(k);
        return it == m.end() ? def : std::stoll(it->second, nullptr, 0);
    }
    float f(const std::string& k, float def) const {
        auto it = m.find(k);
        return it == m.end() ? def : std::stof(it->second);
    }
};

const std::unordered_map<std::string, Op> OPS = {
    {"NOP", Op::NOP}, {"DMA", Op::DMA}, {"DMA_FENCE", Op::DMA_FENCE},
    {"MXU_F16F16", Op::MXU_F16F16}, {"MXU_F32F16", Op::MXU_F32F16},
    {"WAIT_MXU", Op::WAIT_MXU},
    {"VPU_ADD_F32", Op::VPU_ADD_F32}, {"VPU_ADD_F16", Op::VPU_ADD_F16},
    {"VPU_CVT_F32_F16", Op::VPU_CVT_F32_F16}, {"VPU_CVT_F16_F32", Op::VPU_CVT_F16_F32},
    {"VPU_TRANS_F16", Op::VPU_TRANS_F16}, {"VPU_SCALE_F32", Op::VPU_SCALE_F32},
    {"VPU_SOFTMAX_F32", Op::VPU_SOFTMAX_F32}, {"VPU_GELU_F32", Op::VPU_GELU_F32},
    {"VPU_LAYERNORM_F32", Op::VPU_LAYERNORM_F32},
    {"VPU_MUL_F32", Op::VPU_MUL_F32}, {"VPU_SILU_F32", Op::VPU_SILU_F32},
    {"VPU_RMSNORM_BLK", Op::VPU_RMSNORM_BLK}, {"VPU_SOFTMAX_BLK", Op::VPU_SOFTMAX_BLK},
    {"VPU_ROPE_F16", Op::VPU_ROPE_F16},
    {"LOAD_SHARED", Op::LOAD_SHARED}, {"STORE_SHARED", Op::STORE_SHARED},
    {"NMC", Op::NMC}, {"RELEASE", Op::RELEASE}, {"ACQUIRE", Op::ACQUIRE},
    {"HALT", Op::HALT},
};

} // namespace

TraceProgram load_trace(const std::string& path, const CycleConfig& base) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open trace: " + path);

    TraceProgram tp;
    tp.cfg = base;  // architecture YAML defaults; .config lines override below
    std::string raw;
    size_t line = 0;
    int cur_tile = -1;

    while (std::getline(in, raw)) {
        line++;
        auto tok = tokenize(raw);
        if (tok.empty()) continue;

        const std::string& head = tok[0];

        // ── Directives ──
        if (head == ".config") {
            KV kv; for (size_t i = 1; i < tok.size(); i++) kv.put(tok[i], line, raw);
            auto& c = tp.cfg;
            if (kv.has("nmc_enable")) c.nmc_enable = kv.u("nmc_enable", line, raw, 0, false);
            if (kv.has("tile_link")) c.vbond_flits_per_cycle = kv.u("tile_link", line, raw, 1, false);
            if (kv.has("wr_ports")) c.local_wr_ports = kv.u("wr_ports", line, raw, 1, false);
            if (kv.has("rd_ports")) c.local_rd_ports = kv.u("rd_ports", line, raw, 2, false);
            if (kv.has("dma_rate")) c.dma_chunks_per_cycle = kv.u("dma_rate", line, raw, 1, false);
            if (kv.has("dram_density"))
                c.vbond_dram_flits_per_cycle = kv.u("dram_density", line, raw, 2048, false) / 512;
            if (kv.has("ramulator")) c.ramulator_config = kv.s("ramulator", line, raw);
            continue;
        }
        if (head == ".tile") {
            if (tok.size() < 2) err(line, ".tile needs an index", raw);
            cur_tile = std::stoi(tok[1]);
            if (cur_tile < 0 || cur_tile >= NUM_TILES) err(line, "tile out of range", raw);
            continue;
        }
        if (head == ".dump") {
            KV kv; for (size_t i = 1; i < tok.size(); i++) kv.put(tok[i], line, raw);
            tp.dumps.push_back({kv.s("name", line, raw),
                                kv.u("off", line, raw, 0, true),
                                kv.u("bytes", line, raw, 0, true)});
            continue;
        }

        // ── Instruction ──
        auto oit = OPS.find(head);
        if (oit == OPS.end()) err(line, "unknown opcode '" + head + "'", raw);
        if (cur_tile < 0) err(line, "instruction before any .tile", raw);

        KV kv; for (size_t i = 1; i < tok.size(); i++) kv.put(tok[i], line, raw);
        Instr in;
        in.op = oit->second;
        in.src = kv.u("src", line, raw, 0, false);
        in.dst = kv.u("dst", line, raw, 0, false);
        in.rows = static_cast<uint32_t>(kv.u("rows", line, raw, 1, false));
        in.row_bytes = static_cast<uint32_t>(kv.u("row_bytes", line, raw, 0, false));
        in.src_stride = kv.i("src_stride", 0);
        in.dst_stride = kv.i("dst_stride", 0);
        in.a_off = static_cast<uint32_t>(kv.u("a", line, raw, 0, false));
        in.b_off = static_cast<uint32_t>(kv.u("b", line, raw, 0, false));
        in.d_off = static_cast<uint32_t>(kv.u("d", line, raw, 0, false));
        in.acc = kv.u("acc", line, raw, 0, false) != 0;
        in.scalar = kv.f("scalar", 0.0f);
        in.count = static_cast<uint32_t>(kv.u("count", line, raw, 1, false));
        in.keep = static_cast<uint32_t>(kv.u("keep", line, raw, 0, false));
        in.tag = static_cast<uint32_t>(kv.u("tag", line, raw, 0, false));
        in.consumer = static_cast<TileId>(kv.u("consumer", line, raw, 0, false));
        in.arity = static_cast<uint32_t>(kv.u("arity", line, raw, 0, false));
        in.aux = static_cast<int32_t>(kv.i("aux", 0));
        if (kv.has("note")) in.note = kv.s("note", line, raw);
        tp.prog.add(static_cast<TileId>(cur_tile), std::move(in));
    }
    return tp;
}

void load_mem_image(MemoryModel& mem, const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open mem image: " + path);
    while (true) {
        uint64_t off = 0, len = 0;
        if (!in.read(reinterpret_cast<char*>(&off), 8)) break;
        if (!in.read(reinterpret_cast<char*>(&len), 8))
            throw std::runtime_error("mem image truncated (len)");
        std::vector<uint8_t> buf(len);
        if (len && !in.read(reinterpret_cast<char*>(buf.data()),
                            static_cast<std::streamsize>(len)))
            throw std::runtime_error("mem image truncated (data)");
        mem.write(make_dram_addr(off), buf.data(), len);
    }
}

void dump_region(const MemoryModel& mem, uint64_t off, uint64_t bytes,
                 const std::string& path) {
    std::vector<uint8_t> buf(bytes);
    const_cast<MemoryModel&>(mem).read(make_dram_addr(off), buf.data(), bytes);
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write dump: " + path);
    out.write(reinterpret_cast<const char*>(buf.data()),
              static_cast<std::streamsize>(bytes));
}

} // namespace mobol::cycle
