# ─── Ramulator2 Smoke Test ─────────────────────────────────────
add_executable(smoke_ramulator smoke_ramulator.cpp)
target_link_libraries(smoke_ramulator PRIVATE
  mobol_common
  ramulator2
)
target_include_directories(smoke_ramulator PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${RAMULATOR2_EXT_INCLUDES}
)
set_target_properties(smoke_ramulator PROPERTIES
  BUILD_RPATH "${RAMULATOR2_DIR}"
  INSTALL_RPATH "${RAMULATOR2_DIR}"
)

# ─── §5.1 Functional Correctness Test ─────────────────────────
add_executable(test_functional test_functional.cpp)
target_link_libraries(test_functional PRIVATE
  mobol_common
  ramulator2
)
target_include_directories(test_functional PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${RAMULATOR2_EXT_INCLUDES}
)
set_target_properties(test_functional PROPERTIES
  BUILD_RPATH "${RAMULATOR2_DIR}"
  INSTALL_RPATH "${RAMULATOR2_DIR}"
)

# ─── §5.2 + §5.3 Timing Verification Test ─────────────────────
add_executable(test_timing test_timing.cpp)
target_link_libraries(test_timing PRIVATE
  mobol_common
  ramulator2
)
target_include_directories(test_timing PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${RAMULATOR2_EXT_INCLUDES}
)
set_target_properties(test_timing PROPERTIES
  BUILD_RPATH "${RAMULATOR2_DIR}"
  INSTALL_RPATH "${RAMULATOR2_DIR}"
)

# ─── Cycle-accurate dual GEMM test ────────────────────────────
add_executable(test_cycle_dual_gemm test_cycle_dual_gemm.cpp)
target_link_libraries(test_cycle_dual_gemm PRIVATE mobol_common ramulator2)
target_include_directories(test_cycle_dual_gemm PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${RAMULATOR2_EXT_INCLUDES}
)
set_target_properties(test_cycle_dual_gemm PROPERTIES
  BUILD_RPATH "${RAMULATOR2_DIR}"
  INSTALL_RPATH "${RAMULATOR2_DIR}"
)

# ─── Cycle-accurate LLM layer test ────────────────────────────
add_executable(test_cycle_llm test_cycle_llm.cpp)
target_link_libraries(test_cycle_llm PRIVATE mobol_common ramulator2)
target_include_directories(test_cycle_llm PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${RAMULATOR2_EXT_INCLUDES}
)
set_target_properties(test_cycle_llm PROPERTIES
  BUILD_RPATH "${RAMULATOR2_DIR}"
  INSTALL_RPATH "${RAMULATOR2_DIR}"
)

# ─── Cycle-accuracy known-answer microbenchmarks ──────────────
add_executable(test_cycle_micro test_cycle_micro.cpp)
target_link_libraries(test_cycle_micro PRIVATE mobol_common ramulator2)
target_include_directories(test_cycle_micro PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${RAMULATOR2_EXT_INCLUDES}
)

# Register as CTest
add_test(NAME smoke_ramulator COMMAND smoke_ramulator
  "${CMAKE_SOURCE_DIR}/config/ramulator_smoke.yaml")
add_test(NAME test_functional COMMAND test_functional)
add_test(NAME test_timing COMMAND test_timing)
set_tests_properties(test_timing PROPERTIES
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
add_test(NAME test_cycle_dual_gemm COMMAND test_cycle_dual_gemm
  "${CMAKE_SOURCE_DIR}/config/ramulator_3d_dram.yaml")
add_test(NAME test_cycle_llm COMMAND test_cycle_llm
  "${CMAKE_SOURCE_DIR}/config/ramulator_3d_dram.yaml")
add_test(NAME test_cycle_micro COMMAND test_cycle_micro
  "${CMAKE_SOURCE_DIR}/config/ramulator_3d_dram.yaml")
