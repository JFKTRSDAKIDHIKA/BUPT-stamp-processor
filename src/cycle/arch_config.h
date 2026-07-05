/// @file arch_config.h
/// @brief Load the unified architecture YAML (config/mobol_arch.yaml) into a
///        CycleConfig — the single source of truth for every microarch knob.
///
/// Structural constants (tile/bank counts, MXU native size) are compile-time
/// in this build; the loader validates the YAML against them and throws on
/// mismatch (changing topology requires a rebuild).
#pragma once

#include "cycle/cycle_config.h"
#include <string>

namespace mobol::cycle {

/// Parse an architecture YAML into a CycleConfig. Relative paths inside the
/// YAML (e.g. dram.ramulator_config) are resolved against the YAML file's
/// directory if not absolute. Throws std::runtime_error on a malformed file
/// or a structural mismatch with the compiled build.
CycleConfig load_arch_config(const std::string& yaml_path);

} // namespace mobol::cycle
