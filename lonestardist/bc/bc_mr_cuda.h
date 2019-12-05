#pragma once

#include "galois/runtime/DataCommMode.h"
#include "galois/cuda/HostDecls.h"

using ShortPathType = double;

// TODO: WESTON: update args?

//void InitializeGraph_allNodes_cuda(unsigned int __begin, unsigned int __end, struct CUDA_Context* ctx);
//void InitializeGraph_allNodes_cuda(struct CUDA_Context* ctx);
//void FindMessageToSync(struct CUDA_Context* ctx);
//void ConfirmMessageToSend_cuda(struct CUDA_Context* ctx);
//void RoundUpdate_cuda(struct CUDA_Context* ctx);
//void BackFindMessageToSend_cuda(struct CUDA_Context* ctx);
//void BC_masterNodes_cuda(struct CUDA_Context* ctx);
//void Sanity_cuda(struct CUDA_Context* ctx);
//
//void bitset_dependency_reset_cuda(struct CUDA_Context* ctx);
//void bitset_minDistances_reset_cuda(struct CUDA_Context* ctx);
void setCUDAVectorSize(struct CUDA_Context* ctx, unsigned vectorSize);
void InitializeGraph_allNodes_cuda(struct CUDA_Context* ctx);
void InitializeIteration_allNodes_cuda(
    const uint32_t & local_infinity, 
    const uint64_t*  nodeToConsider,
    struct CUDA_Context*  ctx);
void FindMessageToSync_cuda(
    uint32_t roundNumber,
    const uint32_t & local_infinity, 
    uint32_t &dga,
    struct CUDA_Context*  ctx);
void ConfirmMessageToSend_cuda(
    uint32_t roundNumber,
    const uint32_t & local_infinity, 
    struct CUDA_Context*  ctx);
void SendAPSPMessages_cuda(
    const uint32_t & local_infinity, 
    uint32_t &dga,
    struct CUDA_Context*  ctx);
void RoundUpdate_cuda(
    const uint32_t & local_infinity, 
    struct CUDA_Context*  ctx);
void BackFindMessageToSend_cuda(
    const uint32_t & local_infinity, 
    uint32_t roundNumber,
    uint32_t lastRoundNumber,
    struct CUDA_Context*  ctx);
void BackProp_cuda(
    const uint32_t & local_infinity,
    struct CUDA_Context* ctx);
void BC_masterNodes_cuda(
    struct CUDA_Context* ctx,
    const uint64_t* local_nodes_to_consider);
void Sanity_cuda(float & DGAccumulator_sum, float & DGAAccumulator_max, float & DGAccumulator_min, struct CUDA_Context* ctx);
void bitset_dependency_reset_cuda(struct CUDA_Context* ctx);
void bitset_minDistances_reset_cuda(struct CUDA_Context* ctx);


void get_bitset_minDistances_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute);
void bitset_minDistances_reset_cuda(struct CUDA_Context* ctx);
void bitset_minDistances_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end);
uint32_t get_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned LID);
void set_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v);
void add_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v);
bool min_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v);
void batch_get_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v);
void batch_get_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode);
void batch_get_mirror_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v);
void batch_get_mirror_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode);
void batch_get_reset_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, uint32_t i);
void batch_get_reset_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode, uint32_t i);
void batch_set_mirror_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_set_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_add_mirror_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_add_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_min_mirror_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_min_node_minDistances_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_reset_node_minDistances_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, uint32_t v);



void get_bitset_dependency_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute);
void bitset_dependency_reset_cuda(struct CUDA_Context* ctx);
void bitset_dependency_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end);
float get_node_dependency_cuda(struct CUDA_Context* ctx, unsigned LID);
void set_node_dependency_cuda(struct CUDA_Context* ctx, unsigned LID, float v);
void add_node_dependency_cuda(struct CUDA_Context* ctx, unsigned LID, float v);
bool min_node_dependency_cuda(struct CUDA_Context* ctx, unsigned LID, float v);
void batch_get_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v);
void batch_get_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode);
void batch_get_mirror_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v);
void batch_get_mirror_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode);
void batch_get_reset_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, float i);
void batch_get_reset_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode, float i);
void batch_set_mirror_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_set_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_add_mirror_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_add_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_min_mirror_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_min_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode);
void batch_reset_node_dependency_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, float v);


