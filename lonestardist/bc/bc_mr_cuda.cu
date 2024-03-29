#include "gg.h"
#include "ggcuda.h"
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "thread_work.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);

// thread block size
#define TB_SIZE 256

#include "bc_mr_cuda.cuh"

#include "mrbc_tree_cuda.cuh"

#include <iostream>

__global__ void InitializeGraph_kernel(
        CSRGraph graph, 
        unsigned int __begin, 
        unsigned int __end, 
        float * p_bc)
 {
   unsigned tid = TID_1D;
   unsigned nthreads = TOTAL_THREADS_1D;

   index_type src_end =  __end;
   for (index_type src = __begin + tid; src < src_end; src += nthreads)
   {
       p_bc[src] = 0;
   }
 }

void InitializeGraph_allNodes_cuda(struct CUDA_Context* ctx) {
  dim3 blocks;
  dim3 threads;

  kernel_sizing(blocks, threads);
  
  InitializeGraph_kernel <<<blocks, threads>>>(
          ctx->gg, 
          0, 
          ctx->gg.nnodes, 
          ctx->bc.data.gpu_wr_ptr());

  cudaDeviceSynchronize();
  check_cuda_kernel;
}

__global__ void InitializeIteration_kernel(
        CSRGraph graph, 
        unsigned int __begin, 
        unsigned int __end, 
        uint64_t *  cuda_nodes_to_consider,
        uint32_t local_infinity,
        unsigned int numSourcesPerRound,
        uint32_t * p_minDistances,
        double   * p_shortPathCount,
        float    * p_dependency,
        uint32_t * p_roundIndexToSend,
        MRBCTree_cuda * p_mrbc_tree)
 {
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  index_type src_end;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads) {
    p_roundIndexToSend[src] = local_infinity;
    p_mrbc_tree[src].initialize();

    for (index_type i = 0; i < numSourcesPerRound; i++) {
      unsigned int index = src + (i * graph.nnodes);
      if (graph.node_data[src] == cuda_nodes_to_consider[i]) {
        p_minDistances[index] = 0;
        p_shortPathCount[index] = 1;
        p_dependency[index] = 0.0;
        p_mrbc_tree[src].setDistance(i, 0);
       } else {
         p_minDistances[index] = local_infinity;
         p_shortPathCount[index] = 0;
         p_dependency[index] = 0.0;
       }
     }
  }
}

void InitializeIteration_allNodes_cuda(
    const uint32_t & local_infinity, 
    const uint64_t* local_nodes_to_consider, 
    struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;

  kernel_sizing(blocks, threads);

  // Make device vector for local_nodes_to_consider
  uint64_t* cuda_nodes_to_consider;
  cudaMalloc((void**) &cuda_nodes_to_consider, ctx->vectorSize*sizeof(uint64_t));
  cudaMemcpy(cuda_nodes_to_consider, local_nodes_to_consider, ctx->vectorSize*sizeof(uint64_t), cudaMemcpyHostToDevice);
 
  InitializeIteration_kernel <<<blocks, threads>>>(
          ctx->gg, 
          0, 
          ctx->gg.nnodes, 
          cuda_nodes_to_consider,
          local_infinity,
          ctx->vectorSize,
          ctx->minDistances.data.gpu_wr_ptr(),
          ctx->shortPathCount.data.gpu_wr_ptr(),
          ctx->dependency.data.gpu_wr_ptr(),
          ctx->roundIndexToSend.data.gpu_wr_ptr(),
          ctx->mrbc_tree.data.gpu_wr_ptr());
  cudaDeviceSynchronize();
  check_cuda_kernel;

  cudaFree(cuda_nodes_to_consider);
}

__global__ void FindMessageToSync_kernel(
        CSRGraph graph, 
        unsigned int __begin, 
        unsigned int __end, 
        uint32_t roundNumber,
        uint32_t local_infinity,
        uint32_t * p_minDistances,
        uint32_t * p_roundIndexToSend,
        MRBCTree_cuda * p_mrbc_tree,
        DynamicBitset& bitset_minDistances,
        HGAccumulator<uint32_t> dga)
{
    unsigned tid = TID_1D;
    unsigned nthreads = TOTAL_THREADS_1D;

    __shared__ cub::BlockReduce<uint32_t, TB_SIZE>::TempStorage dga_ts;
    index_type src_end;

    dga.thread_entry();

    src_end = __end;
    for (index_type src = __begin + tid; src < src_end; src += nthreads)
    {
        p_roundIndexToSend[src] = p_mrbc_tree[src].getIndexToSend(roundNumber, local_infinity);

        if (p_roundIndexToSend[src] != local_infinity) {
            if (p_minDistances[p_roundIndexToSend[src] * graph.nnodes + src] != 0) {
              bitset_minDistances.set(p_roundIndexToSend[src] * graph.nnodes + src);
            }
            dga.reduce(1);
        } else if ( p_mrbc_tree[src].moreWork() ) {
            dga.reduce(1);
        }

    }

    dga.thread_exit<cub::BlockReduce<uint32_t, TB_SIZE> >(dga_ts);
}

void FindMessageToSync_cuda(
    uint32_t roundNumber,
    const uint32_t & local_infinity, 
    uint32_t &dga,
    struct CUDA_Context*  ctx)
{

  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);

  HGAccumulator<uint32_t> _dga;
  Shared<uint32_t> dgaval  = Shared<uint32_t>(1);
  *(dgaval.cpu_wr_ptr()) = 0;
  _dga.rv = dgaval.gpu_wr_ptr();

  FindMessageToSync_kernel <<<blocks, threads>>>(
          ctx->gg, 
          0, 
          ctx->gg.nnodes, 
          roundNumber,
          local_infinity,
          ctx->minDistances.data.gpu_rd_ptr(),
          ctx->roundIndexToSend.data.gpu_wr_ptr(),
          ctx->mrbc_tree.data.gpu_wr_ptr(),
          *(ctx->minDistances.is_updated.gpu_wr_ptr()), 
          _dga);

  cudaDeviceSynchronize();
  check_cuda_kernel;
  
  dga = *(dgaval.cpu_rd_ptr());
}

__global__ void ConfirmMessageToSend_kernel(
        CSRGraph graph, 
        unsigned int __begin, 
        unsigned int __end, 
        uint32_t roundNumber,
        uint32_t local_infinity,
        MRBCTree_cuda * p_mrbc_tree,
        uint32_t * p_roundIndexToSend)
{
   unsigned tid = TID_1D;
   unsigned nthreads = TOTAL_THREADS_1D;

   index_type src_end;
   src_end = __end;
   for (index_type src = __begin + tid; src < src_end; src += nthreads)
   {
     if (p_roundIndexToSend[src] != local_infinity) {
       p_mrbc_tree[src].markSent(roundNumber);
     }
   }
 }

void ConfirmMessageToSend_cuda(
    uint32_t roundNumber,
    const uint32_t & local_infinity, 
    struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;

  kernel_sizing(blocks, threads);
  
  ConfirmMessageToSend_kernel <<<blocks, threads>>>(
          ctx->gg, 
          0, 
          ctx->gg.nnodes, 
          roundNumber,
          local_infinity,
          ctx->mrbc_tree.data.gpu_wr_ptr(),
          ctx->roundIndexToSend.data.gpu_wr_ptr());
  cudaDeviceSynchronize();
  check_cuda_kernel;
}

__global__ void SendAPSPMessages_kernel(
       CSRGraph graph, 
       unsigned int __begin, 
       unsigned int __end, 
       uint32_t local_infinity,
       uint32_t * p_minDistances,
       double   * p_shortPathCount,
       float    * p_dependency,
       uint32_t * p_roundIndexToSend,
       MRBCTree_cuda * p_mrbc_tree,
       HGAccumulator<uint32_t> dga)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  __shared__ cub::BlockReduce<uint32_t, TB_SIZE>::TempStorage dga_ts;
  index_type dst_end;

  dga.thread_entry();

  dst_end = __end;
  for (index_type dst = __begin + tid; dst < dst_end; dst += nthreads)
  {

    index_type current_edge_end = graph.getFirstEdge((dst) + 1);
    for (index_type current_edge = graph.getFirstEdge(dst); 
           current_edge < current_edge_end;   
           current_edge += 1)
    {   
        index_type src = graph.getAbsDestination(current_edge);
        uint32_t indexToSend = p_roundIndexToSend[src];

        if (indexToSend != local_infinity) {
            uint32_t src_index = (indexToSend * graph.nnodes) + src;
            uint32_t dst_index = (indexToSend * graph.nnodes) + dst;
            uint32_t distValue = p_minDistances[src_index];
            uint32_t newValue  = distValue + 1;
            // Update minDistances vector
            uint32_t oldValue = p_minDistances[dst_index];

            if (oldValue > newValue) {
                p_minDistances[dst_index] = newValue;
                p_mrbc_tree[dst].setDistance(indexToSend, oldValue, newValue);
                p_shortPathCount[dst_index] = p_shortPathCount[src_index];
            } else if (oldValue == newValue) {
                // add to short path
                p_shortPathCount[dst_index] += p_shortPathCount[src_index];
            }
            // dga += 1
            dga.reduce(1);
        }
    }
  }
  dga.thread_exit<cub::BlockReduce<uint32_t, TB_SIZE> >(dga_ts);

}

void SendAPSPMessages_cuda(
    const uint32_t & local_infinity, 
    uint32_t &dga,
    struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;

  kernel_sizing(blocks, threads);
  
  HGAccumulator<uint32_t> _dga;
  Shared<uint32_t> dgaval  = Shared<uint32_t>(1);
  *(dgaval.cpu_wr_ptr()) = 0;
  _dga.rv = dgaval.gpu_wr_ptr();

  SendAPSPMessages_kernel <<<blocks, threads>>>(
          ctx->gg, 
          0, 
          ctx->gg.nnodes, 
          local_infinity,
          ctx->minDistances.data.gpu_wr_ptr(),
          ctx->shortPathCount.data.gpu_wr_ptr(),
          ctx->dependency.data.gpu_wr_ptr(),
          ctx->roundIndexToSend.data.gpu_wr_ptr(),
          ctx->mrbc_tree.data.gpu_wr_ptr(),
          _dga);

  cudaDeviceSynchronize();
  check_cuda_kernel;

  dga = *(dgaval.cpu_rd_ptr());
}

__global__ void RoundUpdate_kernel(
       CSRGraph graph, 
       unsigned int __begin, 
       unsigned int __end, 
       MRBCTree_cuda * p_mrbc_tree)
{
    unsigned tid = TID_1D;
    unsigned nthreads = TOTAL_THREADS_1D;

    index_type src_end;

    src_end = __end;
    for (index_type src = __begin + tid; src < src_end; src += nthreads)
    {
      p_mrbc_tree[src].prepForBackPhase();
    }

}


void RoundUpdate_cuda(
    const uint32_t & local_infinity, 
    struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;

  kernel_sizing(blocks, threads);
  
  RoundUpdate_kernel <<<blocks, threads>>>(
          ctx->gg, 
          0, 
          ctx->gg.nnodes, 
          ctx->mrbc_tree.data.gpu_wr_ptr());

  cudaDeviceSynchronize();
  check_cuda_kernel;
}

__global__ void BackFindMessageToSend_kernel(
       CSRGraph graph, 
       unsigned int __begin, 
       unsigned int __end, 
       uint32_t local_infinity,
       uint32_t roundNumber,
       uint32_t lastRoundNumber,
       float    * p_dependency,
       uint32_t * p_roundIndexToSend,
       MRBCTree_cuda * p_mrbc_tree,
       DynamicBitset& bitset_dependency)
{
    unsigned tid = TID_1D;
    unsigned nthreads = TOTAL_THREADS_1D;

    index_type src_end;

    src_end = __end;
    for (index_type src = __begin + tid; src < src_end; src += nthreads)
    {
      if (!p_mrbc_tree[src].isZeroReached()) {
        p_roundIndexToSend[src] = 
          p_mrbc_tree[src].backGetIndexToSend(roundNumber, lastRoundNumber, local_infinity);

        if (p_roundIndexToSend[src] != local_infinity) {
             uint32_t index = p_roundIndexToSend[src] * graph.nnodes + src;
             if (p_dependency[index] != 0) {
               bitset_dependency.set(index);
             }
        }
      }
    }

}


void BackFindMessageToSend_cuda(
    const uint32_t & local_infinity, 
    uint32_t roundNumber,
    uint32_t lastRoundNumber,
    struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;

  kernel_sizing(blocks, threads);
  
  BackFindMessageToSend_kernel <<<blocks, threads>>>(
          ctx->gg, 
          0, 
          ctx->gg.nnodes, 
          local_infinity,
          roundNumber,
          lastRoundNumber,
          ctx->dependency.data.gpu_wr_ptr(),
          ctx->roundIndexToSend.data.gpu_wr_ptr(),
          ctx->mrbc_tree.data.gpu_wr_ptr(),
          *(ctx->dependency.is_updated.gpu_wr_ptr()));

  cudaDeviceSynchronize();
  check_cuda_kernel;
}

__global__ void BackProp_kernel(
       CSRGraph graph, 
       unsigned int __begin, 
       unsigned int __end, 
       uint32_t local_infinity,
       uint32_t * p_minDistances,
       double   * p_shortPathCount,
       float    * p_dependency,
       uint32_t * p_roundIndexToSend,
       MRBCTree_cuda * p_mrbc_tree)
{
    unsigned tid = TID_1D;
    unsigned nthreads = TOTAL_THREADS_1D;

    index_type dst_end;

    dst_end = __end;

    for (index_type dst = __begin + tid; dst < dst_end; dst += nthreads)
    {
      unsigned i = p_roundIndexToSend[dst];

      if (i != local_infinity) {
        uint32_t myDistance = p_minDistances[dst + (i * graph.nnodes)];

        p_dependency[dst + (i * graph.nnodes)] = 
          p_dependency[dst + (i * graph.nnodes)] 
            * p_shortPathCount[dst + (i * graph.nnodes)];

        float toAdd = ((float)1 + p_dependency[dst + (i * graph.nnodes)]) /
          p_shortPathCount[dst + (i * graph.nnodes)];


        index_type current_edge_end = graph.getFirstEdge((dst) + 1);
        for (index_type current_edge = graph.getFirstEdge(dst); 
               current_edge < current_edge_end;   
               current_edge += 1)
        {   
          
          index_type src = graph.getAbsDestination(current_edge);
          uint32_t sourceDistance = p_minDistances[src + (i * graph.nnodes)];

          if (sourceDistance != 0) {
            if (myDistance == (sourceDistance + 1)) {
              atomicAdd(&p_dependency[src + (i * graph.nnodes)], toAdd);
            }
          }
        }
      }
   }
}

void BackProp_cuda(
    const uint32_t & local_infinity, 
    struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;

  kernel_sizing(blocks, threads);
  
  BackProp_kernel <<<blocks, threads>>>(
          ctx->gg, 
          0, 
          ctx->gg.nnodes, 
          local_infinity,
          ctx->minDistances.data.gpu_wr_ptr(),
          ctx->shortPathCount.data.gpu_wr_ptr(),
          ctx->dependency.data.gpu_wr_ptr(),
          ctx->roundIndexToSend.data.gpu_wr_ptr(),
          ctx->mrbc_tree.data.gpu_wr_ptr());

  cudaDeviceSynchronize();
  check_cuda_kernel;
}

__global__ void BC_kernel(
       CSRGraph graph, 
       unsigned int __begin, 
       unsigned int __end, 
       unsigned int numSourcesPerRound,
       uint64_t *  cuda_nodes_to_consider,
       float    * p_dependency,
       float    * p_bc)
{
    unsigned tid = TID_1D;
    unsigned nthreads = TOTAL_THREADS_1D;

    index_type src_end;

    src_end = __end;
    for (index_type src = __begin + tid; src < src_end; src += nthreads)
    {
      for (unsigned i = 0; i < numSourcesPerRound; i++) {
        if (graph.node_data[src] != cuda_nodes_to_consider[i]) {
          p_bc[src] += p_dependency[src + (i * graph.nnodes)];
        }
      }
    }

}


void BC_masterNodes_cuda(
    struct CUDA_Context*  ctx,
    const uint64_t* local_nodes_to_consider)
{
  dim3 blocks;
  dim3 threads;

  kernel_sizing(blocks, threads);

  // Make device vector for local_nodes_to_consider
  uint64_t* cuda_nodes_to_consider;
  cudaMalloc((void**) &cuda_nodes_to_consider, ctx->vectorSize*sizeof(uint64_t));
  cudaMemcpy(cuda_nodes_to_consider, local_nodes_to_consider, ctx->vectorSize*sizeof(uint64_t), cudaMemcpyHostToDevice);
  
  BC_kernel <<<blocks, threads>>>(
          ctx->gg, 
          ctx->beginMaster, 
          ctx->beginMaster + ctx->numOwned, 
          ctx->vectorSize,
          cuda_nodes_to_consider,
          ctx->dependency.data.gpu_wr_ptr(),
          ctx->bc.data.gpu_wr_ptr());
  cudaDeviceSynchronize();
  check_cuda_kernel;

  cudaFree(cuda_nodes_to_consider);
}

__global__
void Sanity_kernel(CSRGraph graph, unsigned int __begin, unsigned int __end, float * p_bc, HGAccumulator<float> DGAccumulator_sum, HGReduceMax<float> DGAccumulator_max, HGReduceMin<float> DGAccumulator_min) {
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  __shared__ cub::BlockReduce<float, TB_SIZE>::TempStorage DGAccumulator_sum_ts;
  __shared__ cub::BlockReduce<float, TB_SIZE>::TempStorage DGAccumulator_max_ts;
  __shared__ cub::BlockReduce<float, TB_SIZE>::TempStorage DGAccumulator_min_ts;
  index_type src_end;
  DGAccumulator_sum.thread_entry();
  DGAccumulator_max.thread_entry();
  DGAccumulator_min.thread_entry();
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
    bool pop  = src < __end;
    if (pop)
    {
      DGAccumulator_max.reduce(p_bc[src]);
      DGAccumulator_min.reduce(p_bc[src]);
      DGAccumulator_sum.reduce(p_bc[src]);
    }
  }
  DGAccumulator_sum.thread_exit<cub::BlockReduce<float, TB_SIZE> >(DGAccumulator_sum_ts);
  DGAccumulator_max.thread_exit<cub::BlockReduce<float, TB_SIZE> >(DGAccumulator_max_ts);
  DGAccumulator_min.thread_exit<cub::BlockReduce<float, TB_SIZE> >(DGAccumulator_min_ts);
}


void Sanity_cuda(float & DGAccumulator_sum, float & DGAccumulator_max, float & DGAccumulator_min, struct CUDA_Context* ctx) {
  dim3 blocks;
  dim3 threads;
  HGAccumulator<float> _DGAccumulator_sum;
  HGReduceMax<float> _DGAccumulator_max;
  HGReduceMin<float> _DGAccumulator_min;
  kernel_sizing(blocks, threads);
  Shared<float> DGAccumulator_sumval  = Shared<float>(1);
  *(DGAccumulator_sumval.cpu_wr_ptr()) = 0;
  _DGAccumulator_sum.rv = DGAccumulator_sumval.gpu_wr_ptr();
  Shared<float> DGAccumulator_maxval  = Shared<float>(1);
  *(DGAccumulator_maxval.cpu_wr_ptr()) = 0;
  _DGAccumulator_max.rv = DGAccumulator_maxval.gpu_wr_ptr();
  Shared<float> DGAccumulator_minval  = Shared<float>(1);
  *(DGAccumulator_minval.cpu_wr_ptr()) = 0;
  _DGAccumulator_min.rv = DGAccumulator_minval.gpu_wr_ptr();
  Sanity_kernel <<<blocks, threads>>>(ctx->gg, 0, ctx->gg.nnodes, ctx->bc.data.gpu_wr_ptr(), _DGAccumulator_sum, _DGAccumulator_max, _DGAccumulator_min);
  cudaDeviceSynchronize();
  check_cuda_kernel;
  DGAccumulator_sum = *(DGAccumulator_sumval.cpu_rd_ptr());
  DGAccumulator_max = *(DGAccumulator_maxval.cpu_rd_ptr());
  DGAccumulator_min = *(DGAccumulator_minval.cpu_rd_ptr());
}

