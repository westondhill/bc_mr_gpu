if(ENABLE_HETERO_GALOIS)
  # for debugging 
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_definitions(-D__GALOIS_CUDA_CHECK_ERROR__)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -lineinfo)

    add_definitions(-D__GALOIS_DEBUG_WORKLIST__)
  endif()
endif()

include_directories(./include)
include_directories(${CMAKE_SOURCE_DIR}/libgluon/include)
include_directories(${CMAKE_SOURCE_DIR}/libgalois/include)

function(distApp name)
  app(${name}_resilient ${name}/gen.cpp DISTSAFE)
  target_compile_definitions(${name}_resilient PRIVATE __GALOIS_CHECKPOINT__=1)
  target_link_libraries(${name}_resilient galois_gluon distbench)
  target_include_directories(${name}_resilient PUBLIC ${CMAKE_SOURCE_DIR}/libgluon/include)

  #if(ENABLE_HETERO_GALOIS)
  #  target_compile_definitions(${name}_resilient PRIVATE __GALOIS_HET_CUDA__=1)
  #  target_link_libraries(${name}_resilient ${name}_cuda distbench)
  #  cuda_add_library(${name}_resilient_cuda ${name}/gen_cuda.cu)
  #  target_include_directories(${name}_resilient_cuda PUBLIC ${CMAKE_SOURCE_DIR}/libgluon/include)
  #  target_link_libraries(${name}_resilient_cuda ggrt distbench)
  #endif()
endfunction()


distApp(bfs_push)
distApp(sssp_push)
distApp(pagerank_pull)
distApp(sgd_pull)
distApp(kcore_push)
distApp(cc_push)
distApp(pagerank_pull_topo)

distApp(bc_level)
