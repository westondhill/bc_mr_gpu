#pragma once

#include "galois/runtime/DataCommMode.h"
#include "galois/cuda/HostDecls.h"

using ShortPathType = double;

void InitializeGraph_allNodes_cuda(unsigned int __begin, unsigned int __end, struct CUDA_Context* ctx);
void InitializeGraph_allNodes_cuda(struct CUDA_Context* ctx);
void FindMessageToSync(struct CUDA_Context* ctx);
void ConfirmMessageToSend_cuda(struct CUDA_Context* ctx);
void RoundUpdate_cuda(struct CUDA_Context* ctx);
void BackFindMessageToSend_cuda(struct CUDA_Context* ctx);
void BC_masterNodes_cuda(struct CUDA_Context* ctx);
void Sanity_cuda(struct CUDA_Context* ctx);

void bitset_dependency_reset_cuda(struct CUDA_Context* ctx);
void bitset_minDistances_reset_cuda(struct CUDA_Context* ctx);
