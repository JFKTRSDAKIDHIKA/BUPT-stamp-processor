# ─── Common library (types, address, memory, sync, engine, timing) ─
add_library(mobol_common STATIC
  common/address.cpp
  common/memory_model.cpp
  common/sync_manager.cpp
  common/functional_engine.cpp
  workload/dual_gemm.cpp
  workload/cpu_reference.cpp
  timing/trace.cpp
  timing/timing_model.cpp
  cycle/fabric.cpp
  cycle/dram_ctrl.cpp
  cycle/tile_core.cpp
  cycle/chip.cpp
  cycle/arch_config.cpp
  cycle/trace_loader.cpp
  workload/dual_gemm_cycle.cpp
  workload/llm_layer.cpp
)
target_include_directories(mobol_common PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${RAMULATOR2_EXT_INCLUDES}

)
target_link_libraries(mobol_common PUBLIC ramulator2)

# ─── Main simulator executable (placeholder for Phase 2+) ─────
add_executable(mobol_sim main.cpp)
target_link_libraries(mobol_sim PRIVATE
  mobol_common
  ramulator2
)
target_include_directories(mobol_sim PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${RAMULATOR2_EXT_INCLUDES}
)
# Set rpath so the binary can find libramulator.dylib at runtime
set_target_properties(mobol_sim PROPERTIES
  BUILD_RPATH "${RAMULATOR2_DIR}"
  INSTALL_RPATH "${RAMULATOR2_DIR}"
)

# ─── Trace-driven simulator (compiler backend target) ─────────
add_executable(trace_sim trace_sim.cpp)
target_link_libraries(trace_sim PRIVATE mobol_common ramulator2)
target_include_directories(trace_sim PRIVATE ${CMAKE_SOURCE_DIR}/src)
set_target_properties(trace_sim PROPERTIES
  BUILD_RPATH "${RAMULATOR2_DIR}" INSTALL_RPATH "${RAMULATOR2_DIR}")
