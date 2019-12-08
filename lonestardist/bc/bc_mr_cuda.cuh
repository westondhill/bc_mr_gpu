#pragma once
#include <cuda.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include "bc_mr_cuda.h"
#include "galois/runtime/cuda/DeviceSync.h"
#include "mrbc_tree_cuda.cuh"

unsigned vectorSize;

struct CUDA_Context : public CUDA_Context_Common {
  struct CUDA_Context_Field<MRBCTree_cuda> mrbc_tree;

  uint32_t vectorSize;
 
  // sourcData members, size nnodes * vectorSize
  struct CUDA_Context_Field<uint32_t> minDistances;
  struct CUDA_Context_Field<double> shortPathCount;
  struct CUDA_Context_Field<float> dependency;

  struct CUDA_Context_Field<float> bc;
  struct CUDA_Context_Field<uint32_t> roundIndexToSend;
};

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
  ctx->minDistances.data.alloc(ctx->gg.nnodes * ctx->vectorSize);
  ctx->shortPathCount.data.alloc(ctx->gg.nnodes * ctx->vectorSize);
  ctx->dependency.data.alloc(ctx->gg.nnodes * ctx->vectorSize);
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
  ctx->minDistances.shared_data.alloc(max_shared_size);
  ctx->minDistances.is_updated.alloc(1);
  ctx->minDistances.is_updated.cpu_wr_ptr()->alloc(ctx->gg.nnodes * ctx->vectorSize); 

  ctx->shortPathCount.shared_data.alloc(max_shared_size);
  ctx->shortPathCount.is_updated.alloc(1);
  ctx->shortPathCount.is_updated.cpu_wr_ptr()->alloc(ctx->gg.nnodes * ctx->vectorSize); 
  
  ctx->dependency.shared_data.alloc(max_shared_size);
  ctx->dependency.is_updated.alloc(1);
  ctx->dependency.is_updated.cpu_wr_ptr()->alloc(ctx->gg.nnodes * ctx->vectorSize); 
}

void load_graph_CUDA(struct CUDA_Context* ctx, MarshalGraph &g, unsigned num_hosts) {
  size_t mem_usage = mem_usage_CUDA_common(g, num_hosts);
  mem_usage += mem_usage_CUDA_field(&ctx->bc, g, num_hosts);
  mem_usage += mem_usage_CUDA_field(&ctx->roundIndexToSend, g, num_hosts);
  // TODO:  compute mem_usage for sourceData fields and mrbc tree
  printf("[%d] Host memory for communication context: %3u MB\n", ctx->id, mem_usage/1048756);
  load_graph_CUDA_common(ctx, g, num_hosts);
  load_graph_CUDA_field(ctx, &ctx->bc, num_hosts);
        ctx->vectorSize = vectorSize;
  load_graph_CUDA_field(ctx, &ctx->roundIndexToSend, num_hosts);
        load_graph_CUDA_BCData(ctx, num_hosts);

  load_graph_CUDA_field(ctx, &ctx->mrbc_tree, num_hosts);

  reset_CUDA_context(ctx);
}

void reset_CUDA_context(struct CUDA_Context* ctx) {
  ctx->bc.data.zero_gpu();
  ctx->roundIndexToSend.data.zero_gpu();

  // sourceData fields
  ctx->minDistances.data.zero_gpu();
  ctx->shortPathCount.data.zero_gpu();
  ctx->dependency.data.zero_gpu();
}
void setCUDAVectorSize(CUDA_Context* ctx, unsigned vectorSize) {
  ::vectorSize  = vectorSize;
}

// minDistances bitset


void get_bitset_minDistances_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute) {
	ctx->minDistances.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}

void bitset_minDistances_reset_cuda(struct CUDA_Context* ctx) {
	ctx->minDistances.is_updated.cpu_rd_ptr()->reset();
}

void bitset_minDistances_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end) {
	reset_bitset_field(&ctx->minDistances, begin, end);
}

uint32_t get_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned LID) {
	uint32_t *minDistances = ctx->minDistances.data.cpu_rd_ptr();
	return minDistances[LID];
}

void set_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *minDistances = ctx->minDistances.data.cpu_wr_ptr();
	minDistances[LID] = v;
}

void add_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *minDistances = ctx->minDistances.data.cpu_wr_ptr();
	minDistances[LID] += v;
}

bool min_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *minDistances = ctx->minDistances.data.cpu_wr_ptr();
	if (minDistances[LID] > v){
		minDistances[LID] = v;
		return true;
	}
	return false;
}

void batch_get_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
	batch_get_shared_field<uint32_t, sharedMaster, false>(ctx, &ctx->minDistances, from_id, v);
}

void batch_get_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
	batch_get_shared_field<uint32_t, sharedMaster, false>(ctx, &ctx->minDistances, from_id, v, v_size, data_mode);
}

void batch_get_mirror_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
	batch_get_shared_field<uint32_t, sharedMirror, false>(ctx, &ctx->minDistances, from_id, v);
}

void batch_get_mirror_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
	batch_get_shared_field<uint32_t, sharedMirror, false>(ctx, &ctx->minDistances, from_id, v, v_size, data_mode);
}

void batch_get_reset_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, uint32_t i) {
	batch_get_shared_field<uint32_t, sharedMirror, true>(ctx, &ctx->minDistances, from_id, v, i);
}

void batch_get_reset_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode, uint32_t i) {
	batch_get_shared_field<uint32_t, sharedMirror, true>(ctx, &ctx->minDistances, from_id, v, v_size, data_mode, i);
}

void batch_set_mirror_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMirror, setOp>(ctx, &ctx->minDistances, from_id, v, data_mode);
}

void batch_set_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, setOp>(ctx, &ctx->minDistances, from_id, v, data_mode);
}

void batch_add_mirror_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMirror, addOp>(ctx, &ctx->minDistances, from_id, v, data_mode);
}

void batch_add_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, addOp>(ctx, &ctx->minDistances, from_id, v, data_mode);
}

void batch_min_mirror_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMirror, minOp>(ctx, &ctx->minDistances, from_id, v, data_mode);
}

void batch_min_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, minOp>(ctx, &ctx->minDistances, from_id, v, data_mode);
}

void batch_reset_node_minDistances_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, uint32_t v) {
	reset_data_field<uint32_t>(&ctx->minDistances, begin, end, v);
}



// dependency bitset



void get_bitset_dependency_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute) {
	ctx->dependency.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}

void bitset_dependency_reset_cuda(struct CUDA_Context* ctx) {
	ctx->dependency.is_updated.cpu_rd_ptr()->reset();
}

void bitset_dependency_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end) {
	reset_bitset_field(&ctx->dependency, begin, end);
}

float get_node_dependency_cuda(struct CUDA_Context* ctx, unsigned LID) {
	float *dependency = ctx->dependency.data.cpu_rd_ptr();
	return dependency[LID];
}

void set_node_dependency_cuda(struct CUDA_Context* ctx, unsigned LID, float v) {
	float *dependency = ctx->dependency.data.cpu_wr_ptr();
	dependency[LID] = v;
}

void add_node_dependency_cuda(struct CUDA_Context* ctx, unsigned LID, float v) {
	float *dependency = ctx->dependency.data.cpu_wr_ptr();
	dependency[LID] += v;
}

bool min_node_dependency_cuda(struct CUDA_Context* ctx, unsigned LID, float v) {
	float *dependency = ctx->dependency.data.cpu_wr_ptr();
	if (dependency[LID] > v){
		dependency[LID] = v;
		return true;
	}
	return false;
}

void batch_get_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
	batch_get_shared_field<float, sharedMaster, false>(ctx, &ctx->dependency, from_id, v);
}

void batch_get_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
	batch_get_shared_field<float, sharedMaster, false>(ctx, &ctx->dependency, from_id, v, v_size, data_mode);
}

void batch_get_mirror_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
	batch_get_shared_field<float, sharedMirror, false>(ctx, &ctx->dependency, from_id, v);
}

void batch_get_mirror_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
	batch_get_shared_field<float, sharedMirror, false>(ctx, &ctx->dependency, from_id, v, v_size, data_mode);
}

void batch_get_reset_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, float i) {
	batch_get_shared_field<float, sharedMirror, true>(ctx, &ctx->dependency, from_id, v, i);
}

void batch_get_reset_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode, float i) {
	batch_get_shared_field<float, sharedMirror, true>(ctx, &ctx->dependency, from_id, v, v_size, data_mode, i);
}

void batch_set_mirror_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMirror, setOp>(ctx, &ctx->dependency, from_id, v, data_mode);
}

void batch_set_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, setOp>(ctx, &ctx->dependency, from_id, v, data_mode);
}

void batch_add_mirror_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMirror, addOp>(ctx, &ctx->dependency, from_id, v, data_mode);
}

void batch_add_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, addOp>(ctx, &ctx->dependency, from_id, v, data_mode);
}

void batch_min_mirror_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMirror, minOp>(ctx, &ctx->dependency, from_id, v, data_mode);
}

void batch_min_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, minOp>(ctx, &ctx->dependency, from_id, v, data_mode);
}

void batch_reset_node_dependency_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, float v) {
	reset_data_field<float>(&ctx->dependency, begin, end, v);
}
