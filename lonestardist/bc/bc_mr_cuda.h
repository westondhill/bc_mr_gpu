#pragma once

void InitializeGraph_allNodes_cuda(unsigned int __begin, unsigned int __end, struct CUDA_Context* ctx);
void InitializeGraph_allNodes_cuda(GNode curNode);
void FindMessageToSync(GNode curNode);
void ConfirmMessageToSend_cuda(GNode curNode);
void RoundUpdate_cuda();
void BackFindMessageToSend_cuda(GNode dst);
void BC_masterNodes_cuda(struct CUDA_Context* ctx);
void Sanity_cuda();

void bitset_dependency_reset_cuda(struct CUDA_Context* ctx);
void bitset_minDistances_reset_cuda(struct CUDA_Context* ctx);
