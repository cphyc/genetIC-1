cmake_minimum_required(VERSION 3.2)

project(IC)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/genetIC)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

# OpenMP
find_package(OpenMP COMPONENTS CXX)

# HDF5
find_package(HDF5 COMPONENTS HL CXX)

# ---------------------------------------------------------------------------
# Code options. Probably don't fiddle with these unless you know what you're doing.
# Each can be overridden on the command line, e.g.:
#   cmake .. -DCUBIC_INTERPOLATION=OFF -DFRACTIONAL_K_SPLIT=0.5 -DDEBUG_INFO=ON
# ---------------------------------------------------------------------------

option(CUBIC_INTERPOLATION
  "Use cubic interpolation when moving information from a coarse to a fine grid. Slower but much better." ON)

option(VELOCITY_MODIFICATION_GRADIENT_FOURIER_SPACE
  "Use 'ik' as the gradient operator for velocity modification, instead of a fourth-order real-space stencil. Enabling this is essential if CUBIC_INTERPOLATION and/or FILTER_ON_COARSE_GRID are disabled." OFF)

option(ZELDOVICH_GRADIENT_FOURIER_SPACE
  "Same as VELOCITY_MODIFICATION_GRADIENT_FOURIER_SPACE, but for computing the Zeldovich displacement field." ON)

option(FILTER_ON_COARSE_GRID
  "When moving long-wavelength information onto a zoom grid, filter on the base grid then interpolate into the zoom, rather than the default of interpolating then filtering." ON)

option(SAFER_SLOWER
  "Add bounds-checking (with an exception on out-of-range access) to unwrapped grid coordinate indexing. Slower; useful for debugging grid geometry issues. Not referenced in earlier documentation, but present in the source (grid.hpp)." OFF)

option(USE_COMPLEX_FFT
  "Use complex-valued FFTs throughout (ICGenerator<complex<float>>/<complex<double>>) instead of the default real-valued FFTs." OFF)

option(DEBUG_INFO
  "Output extra debug/performance information. Automatically enabled for Debug builds regardless of this setting; set this ON to force it on for Release/RelWithDebInfo builds too." OFF)

set(FRACTIONAL_K_SPLIT "0.3" CACHE STRING
  "Characteristic k0 of the filter used to split information between zoom levels, defined by k0 = k_nyquist * FRACTIONAL_K_SPLIT, where k_nyquist is the Nyquist frequency of the lower-resolution grid. Higher values correlate the zoom grids more precisely with the unzoomed base grids, at the cost of errors on small scales. The default is an empirically-determined trade-off that works well.")

set(FRACTIONAL_FILTER_TEMPERATURE "0.1" CACHE STRING
  "Characteristic temperature of the Fermi filter used to split information between levels, as k0 * FRACTIONAL_FILTER_TEMPERATURE. Smaller values give sharper transitions, better accuracy in Fourier space but worse in real space. The default is an empirically-determined trade-off that works well.")

# --- Options present in the source but NOT exposed as clean toggles here ---
#
# USE_CUFFT (fourier.hpp): switches from FFTW to cuFFT headers/calls. Not wired
# up as an option() because nothing in this file finds or links a cuFFT
# library yet -- enabling it via a bare `-DUSE_CUFFT` will fail to link. Add
# find_package(CUDAToolkit) and link against CUDA::cufft first if you need this.
#
# IGNORE_APPLE_FFTW_THREAD_LIMIT (fourier.hpp): lifts an 8-thread FFTW cap on
# macOS. Niche/platform-specific enough that it's left as a manual
# `-DIGNORE_APPLE_FFTW_THREAD_LIMIT` rather than a first-class option.

# ---------------------------------------------------------------------------
# Git version info (for embedding in binaries, not a user-facing option)
# ---------------------------------------------------------------------------

exec_program(
        "git"
        ${CMAKE_CURRENT_SOURCE_DIR}
        ARGS "rev-parse --short HEAD"
        OUTPUT_VARIABLE GIT_VERSION
)

exec_program(
        "bash"
        ${CMAKE_CURRENT_SOURCE_DIR}
        ARGS "-c 'git ls-files -m | tr \"\\n\" \" \"'"
        OUTPUT_VARIABLE GIT_MODIFIED
)

# ---------------------------------------------------------------------------
# Assemble the definitions list
# ---------------------------------------------------------------------------

set(GENETIC_DEFINITIONS
  FFTW_THREADS
  GIT_VERSION="${GIT_VERSION}"
  GIT_MODIFIED="${GIT_MODIFIED}"
  FRACTIONAL_K_SPLIT=${FRACTIONAL_K_SPLIT}
  FRACTIONAL_FILTER_TEMPERATURE=${FRACTIONAL_FILTER_TEMPERATURE}
)

foreach(opt CUBIC_INTERPOLATION VELOCITY_MODIFICATION_GRADIENT_FOURIER_SPACE
            ZELDOVICH_GRADIENT_FOURIER_SPACE FILTER_ON_COARSE_GRID
            SAFER_SLOWER USE_COMPLEX_FFT)
  if(${opt})
    list(APPEND GENETIC_DEFINITIONS ${opt})
  endif()
endforeach()

if(DEBUG_INFO OR CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND GENETIC_DEFINITIONS DEBUG_INFO)
endif()

# NB: two source files (randomfieldgenerator.hpp, twolevelmapper.hpp) guard
# some OpenMP pragmas with `#ifdef OPENMP` (no underscore) instead of the
# compiler-provided `_OPENMP`, so those blocks were previously dead code even
# when OpenMP was found and linked. Defining OPENMP here activates them
# whenever OpenMP is actually available, matching the evident intent.
if(OpenMP_CXX_FOUND)
  list(APPEND GENETIC_DEFINITIONS OPENMP)
endif()

# ---------------------------------------------------------------------------
# Sources and targets
# ---------------------------------------------------------------------------

set(SOURCE_FILES
        genetIC/src/tools/util_functions.hpp
        genetIC/src/cosmology/parameters.hpp
        genetIC/src/dummyic.hpp
        genetIC/src/tools/numerics/fourier.hpp
        genetIC/src/tools/data_types/float_types.hpp
        genetIC/src/simulation/grid/grid.hpp
        genetIC/src/ic.hpp
        genetIC/src/io.hpp
        genetIC/src/io/grafic.hpp
        genetIC/src/io/gadget.hpp
        genetIC/src/io/gadgethdf.hpp
        genetIC/src/io/input.hpp
        genetIC/src/io/tipsy.hpp
        genetIC/src/simulation/coordinate.hpp
        genetIC/src/cosmology/camb.hpp
        genetIC/src/simulation/particles/mapper/mapper.hpp
        genetIC/src/io/numpy.hpp
        genetIC/src/tools/parser.hpp
        genetIC/src/tools/numerics/vectormath.hpp
        genetIC/src/tools/signaling.hpp
        genetIC/src/simulation/field/randomfieldgenerator.hpp
        genetIC/src/tools/filesystem.h
        genetIC/src/simulation/filters/filter.hpp
        genetIC/src/simulation/multilevelgrid/multilevelgrid.hpp
        genetIC/src/tools/filesystem.cpp
        genetIC/src/simulation/field/multilevelfield.hpp
        genetIC/src/simulation/particles/particle.hpp
        genetIC/src/tools/progress/progress.cpp
        genetIC/src/simulation/field/field.hpp
        genetIC/src/tools/numerics/interpolation.hpp
        genetIC/Makefile
        genetIC/src/simulation/particles/zeldovich.hpp
        genetIC/src/simulation/particles/generator.hpp
        genetIC/src/simulation/particles/multilevelgenerator.hpp
        genetIC/src/tools/data_types/complex.hpp
        genetIC/src/simulation/filters/filterfamily.hpp
        genetIC/src/simulation/grid/virtualgrid.hpp
        genetIC/src/simulation/particles/mapper/mapperiterator.hpp
        genetIC/src/simulation/particles/mapper/onelevelmapper.hpp
        genetIC/src/simulation/particles/mapper/twolevelmapper.hpp
        genetIC/src/simulation/particles/mapper/gasmapper.hpp
        genetIC/src/simulation/particles/mapper/graficmapper.hpp
        genetIC/src/simulation/particles/offsetgenerator.hpp
        genetIC/src/simulation/field/evaluator.hpp
        genetIC/src/simulation/window.hpp
        genetIC/src/simulation/modifications/modification.hpp
        genetIC/src/simulation/modifications/modificationmanager.hpp
        genetIC/src/simulation/modifications/linearmodification.hpp
        genetIC/src/simulation/modifications/quadraticmodification.hpp
        genetIC/src/simulation/multilevelgrid/mask.hpp
        genetIC/src/tools/memmap.hpp
        genetIC/src/tools/numerics/tricubic.hpp
	genetIC/src/tools/logging.hpp
	genetIC/src/tools/logging.cpp
	genetIC/src/simulation/modifications/splice.hpp
	genetIC/src/tools/lru_cache.hpp
        genetIC/src/io/swift.hpp
        genetIC/src/bindings.hpp
)

include_directories( genetIC/  )
include_directories( genetIC/HighFive/include )

link_libraries(fftw3 m fftw3f fftw3_threads gsl gslcblas HDF5::HDF5 OpenMP::OpenMP_CXX)

add_compile_options(-Wextra)

add_executable(genetIC ${SOURCE_FILES} genetIC/src/main.cpp)
add_executable(genetIC_mapper ${SOURCE_FILES} genetIC/src/main_mapper.cpp)

# Add directories containing HDF5 headers
target_include_directories(genetIC PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(genetIC_mapper PRIVATE ${HDF5_INCLUDE_DIRS})

# Apply the code options to both targets
target_compile_definitions(genetIC PRIVATE ${GENETIC_DEFINITIONS})
target_compile_definitions(genetIC_mapper PRIVATE ${GENETIC_DEFINITIONS})
