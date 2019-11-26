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

// TODO: WESTON: likely have to modify this method with "map"
void load_graph_CUDA(struct CUDA_Context* ctx, MarshalGraph &g, unsigned num_hosts) {
	size_t mem_usage = mem_usage_CUDA_common(g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->bcData_minDistance, g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->bcData_shortPathCount, g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->bcData_dependencyValue, g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->bc, g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->roundIndexToSend, g, num_hosts);
	printf("[%d] Host memory for communication context: %3u MB\n", ctx->id, mem_usage/1048756);
	load_graph_CUDA_common(ctx, g, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->bcData_minDistance, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->bcData_shortPathCount, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->bcData_dependencyValue, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->bc, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->roundIndexToSend, num_hosts);
	reset_CUDA_context(ctx);
}

void reset_CUDA_context(struct CUDA_Context* ctx) {
	ctx->bcData_minDistance.data.zero_gpu();
	ctx->bcData_shortPathCount.data.zero_gpu();
	ctx->bcData_dependencyValue.data.zero_gpu();
	ctx->bc.data.zero_gpu();
	ctx->roundIndexToSend.data.zero_gpu();
}


