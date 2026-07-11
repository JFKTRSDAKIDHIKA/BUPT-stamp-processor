cmake_minimum_required(VERSION 3.20)
project(mobol_sim VERSION 0.1.0 LANGUAGES CXX)

# ─── C++ standard ──────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Warnings as errors ────────────────────────────────────────
add_compile_options(-Wall -Wextra -Wpedantic -Werror)
# Suppress specific warnings that are noisy but benign
add_compile_options(-Wno-unused-parameter)

# The simulator models byte-addressable SRAM/DRAM as uint8_t arrays that
# datapaths access as f16/f32 (as hardware does). Disable type-based alias
# analysis so those reinterpret casts are well-defined at any -O level.
add_compile_options(-fno-strict-aliasing)

# ─── Structural DSE knobs (compile-time; one build dir per variant) ───
set(MOBOL_NUM_TILES 16 CACHE STRING "Compute tiles on the base die (power of two)")
set(MOBOL_TILES_PER_GROUP 4 CACHE STRING "Tiles sharing one bank/DRAM column")
set(MOBOL_SHARED_MB 8 CACHE STRING "Shared scratchpad MB per buffer-die bank")
set(MOBOL_TOPOLOGY ring CACHE STRING "Base-die NoC topology: ring|mesh|torus")
if(MOBOL_TOPOLOGY STREQUAL "ring")
  set(MOBOL_TOPOLOGY_ID 0)
elseif(MOBOL_TOPOLOGY STREQUAL "mesh")
  set(MOBOL_TOPOLOGY_ID 1)
elseif(MOBOL_TOPOLOGY STREQUAL "torus")
  set(MOBOL_TOPOLOGY_ID 2)
else()
  message(FATAL_ERROR "MOBOL_TOPOLOGY must be ring, mesh or torus")
endif()
add_compile_definitions(
  MOBOL_NUM_TILES=${MOBOL_NUM_TILES}
  MOBOL_TILES_PER_GROUP=${MOBOL_TILES_PER_GROUP}
  MOBOL_SHARED_MB=${MOBOL_SHARED_MB}
  MOBOL_TOPOLOGY=${MOBOL_TOPOLOGY_ID}
)
message(STATUS "MOBOL structural: tiles=${MOBOL_NUM_TILES} tiles/group=${MOBOL_TILES_PER_GROUP} shared=${MOBOL_SHARED_MB}MB topology=${MOBOL_TOPOLOGY}")

# ─── Build type default ───────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ─── Ramulator2 ────────────────────────────────────────────────
set(RAMULATOR2_DIR "/Users/sjia/ramulator2" CACHE PATH "Path to Ramulator2 root")

if(NOT EXISTS "${RAMULATOR2_DIR}/src/base/base.h")
  message(FATAL_ERROR "Ramulator2 sources not found at ${RAMULATOR2_DIR}")
endif()

# Compile Ramulator2 (and its bundled yaml-cpp) FROM SOURCE into this
# build. The prebuilt libramulator.dylib embeds its own copies of
# yaml-cpp/spdlog inline code; loading it beside an executable that also
# instantiates those inline symbols causes cross-image weak-symbol
# coalescing conflicts on macOS — observed as intermittent (ASLR-
# dependent) heap corruption inside YAML statics and DRAM controller
# queues. Building from source with one toolchain removes the hazard,
# and lets sanitizers instrument Ramulator2 too.
#
# OBJECT library: keeps every translation unit, so Ramulator2's static
# factory registrations (Impl classes) survive linking.
file(GLOB_RECURSE RAMULATOR2_SRCS "${RAMULATOR2_DIR}/src/*.cpp")
# trr.cpp and GDDR6.cpp are unfinished upstream and excluded from
# Ramulator2's own build lists as well.
list(FILTER RAMULATOR2_SRCS EXCLUDE REGEX "(main\\.cpp|/test/|/example/|impl/plugin/trr\\.cpp|impl/GDDR6\\.cpp)")
file(GLOB YAML_CPP_SRCS "${RAMULATOR2_DIR}/ext/yaml-cpp/src/*.cpp")

# hbm_refresh.cpp registers our HBM-compatible refresh manager into the
# Ramulator2 factory; it lives in this OBJECT library so its static
# registration object cannot be dead-stripped by the linker.
add_library(ramulator2 OBJECT ${RAMULATOR2_SRCS} ${YAML_CPP_SRCS}
  "${CMAKE_SOURCE_DIR}/src/cycle/hbm_refresh.cpp")
# SYSTEM: our -Werror policy must not apply to third-party headers.
target_include_directories(ramulator2 SYSTEM PUBLIC
  "${RAMULATOR2_DIR}/src"
  "${RAMULATOR2_DIR}/ext/yaml-cpp/include"
  "${RAMULATOR2_DIR}/ext/spdlog/include"
)
# Third-party code: don't apply our warning policy to it.
target_compile_options(ramulator2 PRIVATE -w)

# Ramulator2 include dirs now flow through the ramulator2 target itself
# (SYSTEM PUBLIC); keep the variable for legacy references.
set(RAMULATOR2_EXT_INCLUDES "")

# ─── Testing ────────────────────────────────────────────────────
enable_testing()

# ─── Source subdirectories ─────────────────────────────────────
add_subdirectory(src)
add_subdirectory(tests)
