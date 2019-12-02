#pragma once
#include <cuda.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include "bc_mr_cuda.h"
#include "galois/runtime/cuda/DeviceSync.h"

// TODO: WESTON: use the cuda bitset provided by Galois?
//#include "mrbc_bitset.hh"

// TODO: WESTON: include hash map header
unsigned vectorSize;

struct CUDA_BCData {
  struct CUDA_Context_Field<uint32_t> minDistance;
  struct CUDA_Context_Field<double> shortPathCount;
  struct CUDA_Context_Field<float> dependencyValue;
};

struct CUDA_Context : public CUDA_Context_Common {
    // assumes one source per round
    // BCData struct
    struct CUDA_Context_Field<uint32_t> bcData_minDistance;
    struct CUDA_Context_Field<double> bcData_shortPathCount;
    struct CUDA_Context_Field<float> bcData_dependencyValue;

    // replacement for MRBCTree
    // TODO: WESTON: CUDA_Context_Field of hashes
    //struct CUDA_Context_Field<gpu_hash_table<uint32_t, BitSet, SlabHashTypeT::ConcurrentMap>;
    // 101   gpu_hash_table<KeyT, ValueT, SlabHashTypeT::ConcurrentMap>
    //  102       hash_table(num_keys, num_buckets, DEVICE_ID, seed);
    //    galois::gstl::Vector<uint32_t> sourceData;
        uint32_t vectorSize;
	// struct CUDA_BCData* sourceData;
	
	// sourcData members, size nnodes * vectorSize
	struct CUDA_Context_Field<uint32_t> minDistance;
	struct CUDA_Context_Field<double> shortPathCount;
	struct CUDA_Context_Field<float> dependencyValue;

	struct CUDA_Context_Field<float> bc;
	struct CUDA_Context_Field<uint32_t> roundIndexToSend;
};

// TODO: WESTON: update rest of file regarding CUDA_Context field changes

struct CUDA_Context* get_CUDA_context(int id) {
	struct CUDA_Context* ctx;
	ctx = (struct CUDA_Context* ) calloc(1, sizeof(struct CUDA_Context));
	ctx->id = id;
	return ctx;
}

bool init_CUDA_context(struct CUDA_Context* ctx, int device) {
	return init_CUDA_context_common(ctx, device);
}

// A copy of load_graph_CUDA_field except with different size
void load_graph_CUDA_BCData(struct CUDA_Context* ctx, unsigned num_hosts) {
  printf("Entered load_graph_CUDA_BCData\n");
  ctx->minDistance.data.alloc(ctx->gg.nnodes * ctx->vectorSize);
  ctx->shortPathCount.data.alloc(ctx->gg.nnodes * ctx->vectorSize);
  ctx->dependencyValue.data.alloc(ctx->gg.nnodes * ctx->vectorSize);
  // Is this the same for all fields?
  size_t max_shared_size = 0; // for union across master/mirror of all hosts
  for (int32_t h = 0; h < num_hosts; ++h) {
    if (ctx->master.num_nodes[h] * ctx->vectorSize > max_shared_size) {
      max_shared_size = ctx->master.num_nodes[h] * ctx->vectorSize;
    }
  }
  for (uint32_t h = 0; h < num_hosts; ++h) {
    if (ctx->mirror.num_nodes[h] * ctx->vectorSize > max_shared_size) {
      max_shared_size = ctx->mirror.num_nodes[h] * ctx->vectorSize;
    }
  }
  ctx->minDistance.shared_data.alloc(max_shared_size);
  ctx->minDistance.is_updated.alloc(1);
  ctx->minDistance.is_updated.cpu_wr_ptr()->alloc(ctx->gg.nnodes * ctx->vectorSize); 

  ctx->shortPathCount.shared_data.alloc(max_shared_size);
  ctx->shortPathCount.is_updated.alloc(1);
  ctx->shortPathCount.is_updated.cpu_wr_ptr()->alloc(ctx->gg.nnodes * ctx->vectorSize); 
  
  ctx->dependencyValue.shared_data.alloc(max_shared_size);
  ctx->dependencyValue.is_updated.alloc(1);
  ctx->dependencyValue.is_updated.cpu_wr_ptr()->alloc(ctx->gg.nnodes * ctx->vectorSize); 
  printf("Exiting load_graph_CUDA_BCData\n");
}

// TODO: WESTON: likely have to modify this method with "map"
void load_graph_CUDA(struct CUDA_Context* ctx, MarshalGraph &g, unsigned num_hosts) {
	size_t mem_usage = mem_usage_CUDA_common(g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->bc, g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->roundIndexToSend, g, num_hosts);
        //TODO: WESTON:  compute mem_usage for sourceData
	printf("[%d] Host memory for communication context: %3u MB\n", ctx->id, mem_usage/1048756);
	load_graph_CUDA_common(ctx, g, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->bc, num_hosts);
        ctx->vectorSize = vectorSize;
        printf("ctx vectorSize: %u\n", ctx->vectorSize);
	load_graph_CUDA_field(ctx, &ctx->roundIndexToSend, num_hosts);
        load_graph_CUDA_BCData(ctx, num_hosts);
	reset_CUDA_context(ctx);
        printf("exiting load_graph_CUDA\n");
}

void reset_CUDA_context(struct CUDA_Context* ctx) {
	ctx->bc.data.zero_gpu();
	ctx->roundIndexToSend.data.zero_gpu();

	// sourceData fields
	// ctx->minDistance.data.zero_gpu();
	// ctx->shortPathCount.data.zero_gpu();
	// ctx->dependencyValue.data.zero_gpu();
}


/* sourceData if sharing among hosts messes up one long field
void load_graph_CUDA_BCDATA(struct CUDA_Context* ctx, unsigned num_hosts, unsigned vectorSize) {
  // save vector length
  ctx->vectorSize = vectorSize;
  // Allocate array of wanted size
  ctx->sourceData = (struct CUDA_BCData*) calloc(vectorSize, sizeof(CUDA_BCData));

  // Loop through vector and load all the CUDA fields
  for (unsigned i = 0; i < vectorSize; i++) {
    load_graph_CUDA_field(ctx, &ctx->sourceData[i].minDistance, num_hosts);
    load_graph_CUDA_field(ctx, &ctx->sourceData[i].shortPathCount, num_hosts);
    load_graph_CUDA_field(ctx, &ctx->sourceData[i].dependencyValue, num_hosts);
  }
} */

void setCUDAVectorSize(CUDA_Context* ctx, unsigned vectorSize) {
  ::vectorSize  = vectorSize;
}
