cmake_minimum_required(VERSION 3.10)
project(swarmformer C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    set(MATH_LIB "")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O2 /arch:AVX2 /fp:fast /GL /Gy /Oi /Ot /Qpar /Qvec-report:2")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG")
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
    set(MATH_LIB "m")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -mfma -ffast-math -ftree-vectorize -funroll-loops -fomit-frame-pointer -flto")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")
endif()

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(SOURCES
    src/main.c
    src/core/tensor.c
    src/core/activation.c
    src/core/quantization.c
    src/layers/linear.c
    src/layers/embedding.c
    src/layers/swarm_layers.c
    src/model/swarmformer.c
    src/utils/loader.c
    src/utils/tokenizer.c
    src/utils/profiler.c
)

add_executable(swarmformer ${SOURCES})

target_include_directories(swarmformer PRIVATE src)

if(WIN32)
    target_link_libraries(swarmformer ${MATH_LIB} OpenMP::OpenMP_C)
else()
    target_link_libraries(swarmformer ${MATH_LIB})
endif() 