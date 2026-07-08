cmake_minimum_required(VERSION 3.17)

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

option(USE_CUFFT
  "Use NVIDIA cuFFT (via its FFTW-compatible cufftw wrapper) instead of CPU FFTW. Requires a CUDA toolkit on the build machine." OFF)

option(DEBUG_INFO
  "Output extra debug/performance information. Automatically enabled for Debug builds regardless of this setting; set this ON to force it on for Release/RelWithDebInfo builds too." OFF)

set(FRACTIONAL_K_SPLIT "0.3" CACHE STRING
  "Characteristic k0 of the filter used to split information between zoom levels, defined by k0 = k_nyquist * FRACTIONAL_K_SPLIT, where k_nyquist is the Nyquist frequency of the lower-resolution grid. Higher values correlate the zoom grids more precisely with the unzoomed base grids, at the cost of errors on small scales. The default is an empirically-determined trade-off that works well.")

set(FRACTIONAL_FILTER_TEMPERATURE "0.1" CACHE STRING
  "Characteristic temperature of the Fermi filter used to split information between levels, as k0 * FRACTIONAL_FILTER_TEMPERATURE. Smaller values give sharper transitions, better accuracy in Fourier space but worse in real space. The default is an empirically-determined trade-off that works well.")

# --- Options present in the source but NOT exposed as clean toggles here ---
#
# IGNORE_APPLE_FFTW_THREAD_LIMIT (fourier.hpp): lifts an 8-thread FFTW cap on
# macOS. Niche/platform-specific enough that it's left as a manual
# `-DIGNORE_APPLE_FFTW_THREAD_LIMIT` rather than a first-class option.

# ---------------------------------------------------------------------------
# Git version info (for embedding in binaries, not a user-facing option)
# ---------------------------------------------------------------------------

execute_process(
        COMMAND git rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
        COMMAND bash -c "git ls-files -m | tr \"\\n\" \" \""
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_MODIFIED
        OUTPUT_STRIP_TRAILING_WHITESPACE
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
            SAFER_SLOWER USE_COMPLEX_FFT USE_CUFFT)
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

# Special case on macos
if(APPLE)
  include_directories(/opt/homebrew/include)
  link_directories(/opt/homebrew/lib)
  include_directories( /opt/local/include )
  link_directories(/opt/local/lib )
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

# ---------------------------------------------------------------------------
# FFT backend: CPU FFTW (default) or NVIDIA cuFFT (if USE_CUFFT=ON)
# ---------------------------------------------------------------------------
if(USE_CUFFT)
  if(CMAKE_VERSION VERSION_LESS 3.17)
    message(FATAL_ERROR "USE_CUFFT requires CMake >= 3.17 (for the CUDAToolkit module). Found ${CMAKE_VERSION}. Either upgrade CMake or set USE_CUFFT=OFF.")
  endif()

  find_package(CUDAToolkit REQUIRED)

  set(CUFFT_LIBRARY_DIR "" CACHE PATH
    "Manual override: directory containing libcufft/libcufftw, for installations (e.g. NVIDIA HPC SDK / nvhpc) where these aren't in CUDAToolkit_LIBRARY_DIR.")
  set(CUFFT_INCLUDE_DIR "" CACHE PATH
    "Manual override: directory containing cufftw.h, for installations (e.g. NVIDIA HPC SDK / nvhpc) where it isn't in CUDAToolkit_INCLUDE_DIRS.")

  set(_cufft_search_paths ${CUFFT_LIBRARY_DIR} ${CUDAToolkit_LIBRARY_DIR})
  set(_cufft_include_search_paths ${CUFFT_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})

  # NVIDIA HPC SDK (nvhpc) installations split the math libraries
  # (cufft, cublas, cusparse, ...) -- headers included -- into a separate
  # math_libs/<version>/{include,lib64} tree, sibling to cuda/<version>/.
  # CUDAToolkit_LIBRARY_DIR/CUDAToolkit_INCLUDE_DIRS alone therefore won't
  # find cufft/cufftw on nvhpc-based clusters (e.g. Irene).
  string(REPLACE "/cuda/" "/math_libs/" _cufft_nvhpc_guess "${CUDAToolkit_LIBRARY_DIR}")
  if(NOT _cufft_nvhpc_guess STREQUAL CUDAToolkit_LIBRARY_DIR)
    list(APPEND _cufft_search_paths ${_cufft_nvhpc_guess})
    string(REPLACE "/lib64" "/include" _cufft_nvhpc_include_guess "${_cufft_nvhpc_guess}")
    list(APPEND _cufft_include_search_paths ${_cufft_nvhpc_include_guess})
  endif()

  find_path(CUFFT_HEADER_DIR NAMES cufftw.h HINTS ${_cufft_include_search_paths})
  if(NOT CUFFT_HEADER_DIR)
    message(FATAL_ERROR "USE_CUFFT is ON but cufftw.h could not be found. Searched: ${_cufft_include_search_paths}. Locate it with e.g.:\n  find $(dirname $(dirname ${CUDAToolkit_LIBRARY_DIR}))/.. -name 'cufftw.h' 2>/dev/null\nthen pass it explicitly: -DCUFFT_INCLUDE_DIR=/path/to/math_libs/<version>/include")
  endif()
  set(FFT_INCLUDE_DIRS ${CUFFT_HEADER_DIR})

  if(TARGET CUDA::cufftw)
    # CUDA::cufftw imported target only exists from CMake 3.23 onward
    set(FFT_LIBRARIES CUDA::cufftw CUDA::cufft)
  else()
    find_library(CUFFTW_LIBRARY NAMES cufftw HINTS ${_cufft_search_paths})
    find_library(CUFFT_LIBRARY NAMES cufft HINTS ${_cufft_search_paths})
    if(NOT CUFFTW_LIBRARY OR NOT CUFFT_LIBRARY)
      message(FATAL_ERROR "USE_CUFFT is ON but the cufftw/cufft libraries could not be found. Searched: ${_cufft_search_paths}. If you're on the NVIDIA HPC SDK (nvhpc), locate the real path with e.g.:\n  find $(dirname $(dirname ${CUDAToolkit_LIBRARY_DIR}))/.. -name 'libcufft.so*' 2>/dev/null\nthen pass it explicitly: -DCUFFT_LIBRARY_DIR=/path/to/math_libs/<version>/lib64")
    endif()
    set(FFT_LIBRARIES ${CUFFTW_LIBRARY} ${CUFFT_LIBRARY})
  endif()

  message(STATUS "genetIC: using cuFFT (via cufftw) as the FFT backend")
  message(STATUS "genetIC: cufftw header dir = ${CUFFT_HEADER_DIR}")
  message(STATUS "genetIC: cufftw = ${CUFFTW_LIBRARY}")
  message(STATUS "genetIC: cufft  = ${CUFFT_LIBRARY}")
else()
  # FFTW3
  find_package(FFTW3 COMPONENTS Threads)

  set(FFT_INCLUDE_DIRS "")
  # cufftw re-exports the same symbol names as real FFTW3, so these must
  # never be linked at the same time as the cuFFT libraries above.
  set(FFT_LIBRARIES fftw3 fftw3f fftw3_threads)

  message(STATUS "genetIC: using FFTW3 as the FFT backend")
endif()

link_libraries(${FFT_LIBRARIES} m gsl gslcblas HDF5::HDF5 OpenMP::OpenMP_CXX)

add_compile_options(-Wextra)

add_executable(genetIC ${SOURCE_FILES} genetIC/src/main.cpp)
add_executable(genetIC_mapper ${SOURCE_FILES} genetIC/src/main_mapper.cpp)

# Add directories containing HDF5 and (if enabled) cuFFT headers
target_include_directories(genetIC PRIVATE ${HDF5_INCLUDE_DIRS} ${FFT_INCLUDE_DIRS})
target_include_directories(genetIC_mapper PRIVATE ${HDF5_INCLUDE_DIRS} ${FFT_INCLUDE_DIRS})

# Apply the code options to both targets
target_compile_definitions(genetIC PRIVATE ${GENETIC_DEFINITIONS})
target_compile_definitions(genetIC_mapper PRIVATE ${GENETIC_DEFINITIONS})
