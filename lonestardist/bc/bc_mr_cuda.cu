#include "gg.h"
#include "ggcuda.h"
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "thread_work.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);

// thread block size
#define TB_SIZE 256

//#include "kernels/reduce.cuh"
#include "bc_mr_cuda.cuh"

#include "mrbc_tree_cuda.cuh"

#include <iostream>

//   galois::do_all(
//       galois::iterate(allNodes.begin(), allNodes.end()),
//       [&](GNode curNode) {
//         NodeData& cur_data = graph.getData(curNode);
//         cur_data.sourceData.resize(vectorSize);
//         cur_data.bc = 0.0;
//       },  

__global__ void InitializeGraph_kernel(
        CSRGraph graph, 
        unsigned int __begin, 
        unsigned int __end, 
        float * p_bc)
 {
   unsigned tid = TID_1D;
   unsigned nthreads = TOTAL_THREADS_1D;

   //const unsigned __kernel_tb_size = TB_SIZE;
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



//   galois::do_all(
//       galois::iterate(allNodes.begin(), allNodes.end()),
//       [&](GNode curNode) {
//         NodeData& cur_data = graph.getData(curNode);
//         cur_data.roundIndexToSend = infinity;
//         cur_data.dTree.initialize();
//         for (unsigned i = 0; i < numSourcesPerRound; i++) {
//           // min distance and short path count setup
//           if (nodesToConsider[i] == graph.getGID(curNode)) { // source node
//             cur_data.sourceData[i].minDistances = 0;
//             cur_data.sourceData[i].shortPathCount = 1;
//             cur_data.sourceData[i].dependency = 0.0;
//             cur_data.dTree.setDistance(i, 0);
//           } else { // non-source node
//             cur_data.sourceData[i].minDistances = infinity;
//             cur_data.sourceData[i].shortPathCount = 0;
//             cur_data.sourceData[i].dependency = 0.0;
//           }
//         }
//       },
//       galois::loopname(syncSubstrate->get_run_identifier("InitializeIteration").c_str()),
//       galois::no_stats());

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

  //const unsigned __kernel_tb_size = TB_SIZE;
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
 

  //std::cout << "\n\nWESTON DEBUG BEGIN\n\n";
  //std::cout << cuda_nodes_to_consider << std::endl;
  //std::cout << ctx->minDistances.data.gpu_wr_ptr() << std::endl;
  //std::cout << ctx->shortPathCount.data.gpu_wr_ptr() << std::endl;
  //std::cout << ctx->dependency.data.gpu_wr_ptr() << std::endl;
  //std::cout << ctx->roundIndexToSend.data.gpu_wr_ptr() << std::endl;
  //std::cout << ctx->mrbc_tree.data.gpu_wr_ptr() << std::endl;
  //std::cout << "\n\nWESTON DEBUG END\n\n";

  //        cuda_nodes_to_consider, " ",
  //        local_infinity, " ",
  //        ctx->vectorSize, " ",
  //        ctx->minDistances.data.gpu_wr_ptr(), " ",
  //        ctx->shortPathCount.data.gpu_wr_ptr(), " ",
  //        ctx->dependency.data.gpu_wr_ptr(), " ",
  //        ctx->roundIndexToSend.data.gpu_wr_ptr(), " ",
  //        ctx->mrbc_tree.data.gpu_wr_ptr(), "\n");

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




//   galois::do_all(
//       galois::iterate(allNodes.begin(), allNodes.end()),
//       [&](GNode curNode) {
//         NodeData& cur_data = graph.getData(curNode);
//         cur_data.roundIndexToSend = cur_data.dTree.getIndexToSend(roundNumber);
// 
//         if (cur_data.roundIndexToSend != infinity) {
//           if (cur_data.sourceData[cur_data.roundIndexToSend].minDistances != 0) {
//             bitset_minDistancess.set(curNode);
//           }
//           dga += 1;
//         } else if (cur_data.dTree.moreWork()) {
//           dga += 1;
//         }
//       },

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

    //const unsigned __kernel_tb_size = TB_SIZE;
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

//   galois::do_all(
//       galois::iterate(allNodes.begin(), allNodes.end()),
//       [&](GNode curNode) {
//         NodeData& cur_data = graph.getData(curNode);
//         if (cur_data.roundIndexToSend != infinity) {
//           cur_data.dTree.markSent(roundNumber);
//         }
//       },
//       galois::loopname(
//           syncSubstrate->get_run_identifier("ConfirmMessageToSend").c_str()),
//       galois::no_stats());

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

   //const unsigned __kernel_tb_size = TB_SIZE;
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

  //const unsigned __kernel_tb_size = TB_SIZE;
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


//   galois::do_all(
//       galois::iterate(allNodes.begin(), allNodes.end()),
//       [&](GNode node) {
//         NodeData& cur_data = graph.getData(node);
//         cur_data.dTree.prepForBackPhase();
//       },  
//       galois::loopname(
//           syncSubstrate->get_run_identifier("RoundUpdate").c_str()),
//       galois::no_stats());


__global__ void RoundUpdate_kernel(
       CSRGraph graph, 
       unsigned int __begin, 
       unsigned int __end, 
       MRBCTree_cuda * p_mrbc_tree)
{
    unsigned tid = TID_1D;
    unsigned nthreads = TOTAL_THREADS_1D;

    //const unsigned __kernel_tb_size = TB_SIZE;
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



//   galois::do_all(
//       galois::iterate(allNodes.begin(), allNodes.end()),
//       [&](GNode dst) {
//         NodeData& dst_data        = graph.getData(dst);
//
//         // if zero distances already reached, there is no point sending things
//         // out since we don't care about dependecy for sources (i.e. distance
//         // 0)
//         if (!dst_data.dTree.isZeroReached()) {
//           dst_data.roundIndexToSend =
//             dst_data.dTree.backGetIndexToSend(roundNumber, lastRoundNumber);
//
//           if (dst_data.roundIndexToSend != infinity) {
//             // only comm if not redundant 0
//             if (dst_data.sourceData[dst_data.roundIndexToSend].dependency != 0) {
//               bitset_dependency.set(dst);
//             }
//           }
//         }
//       },
//       galois::loopname(
//         syncSubstrate->get_run_identifier("BackFindMessageToSend").c_str()
//       ),
//       galois::no_stats());

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

    //const unsigned __kernel_tb_size = TB_SIZE;
    index_type src_end;

    src_end = __end;
    for (index_type src = __begin + tid; src < src_end; src += nthreads)
    {
      if (p_mrbc_tree[src].isZeroReached()) {
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


// void BackPropOp(GNode dst, Graph& graph) {
//   NodeData& dst_data = graph.getData(dst);
//   unsigned i         = dst_data.roundIndexToSend;
// 
//   if (i != infinity) {
//     uint32_t myDistance = dst_data.sourceData[i].minDistances;
// 
//     // calculate final dependency value
//     dst_data.sourceData[i].dependency =
//       dst_data.sourceData[i].dependency *
//         dst_data.sourceData[i].shortPathCount;
// 
//     // get the value to add to predecessors
//     float toAdd = ((float)1 + dst_data.sourceData[i].dependency) /
//                   dst_data.sourceData[i].shortPathCount;
// 
//     for (auto inEdge : graph.edges(dst)) {
//       GNode src      = graph.getEdgeDst(inEdge);
//       auto& src_data = graph.getData(src);
//       uint32_t sourceDistance = src_data.sourceData[i].minDistances;
// 
//       // source nodes of this batch (i.e. distance 0) can be safely
//       // ignored
//       if (sourceDistance != 0) {
//         // determine if this source is a predecessor
//         if (myDistance == (sourceDistance + 1)) {
//           // add to dependency of predecessor using our finalized one
//           galois::atomicAdd(src_data.sourceData[i].dependency, toAdd);
//         }
//       }   
//     }   
//   }
// }


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

    //const unsigned __kernel_tb_size = TB_SIZE;
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
//  galois::do_all(
//       galois::iterate(masterNodes.begin(), masterNodes.end()),
//       [&](GNode node) {
//         NodeData& cur_data = graph.getData(node);
//
//         for (unsigned i = 0; i < numSourcesPerRound; i++) {
//           // exclude sources themselves from BC calculation
//           if (graph.getGID(node) != nodesToConsider[i]) {
//             cur_data.bc += cur_data.sourceData[i].dependency;
//           }
//         }
//       },
//       galois::loopname(syncSubstrate->get_run_identifier("BC").c_str()),
//       galois::no_stats());

__global__ void BC_kernel(
       CSRGraph graph, 
       unsigned int __begin, 
       unsigned int __end, 
       unsigned int numSourcesPerRound,
       uint64_t *  cuda_nodes_to_consider,
       float    * p_dependency,
       float* p_bc)
{
    unsigned tid = TID_1D;
    unsigned nthreads = TOTAL_THREADS_1D;

    //const unsigned __kernel_tb_size = TB_SIZE;
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


//  galois::do_all(galois::iterate(graph.masterNodesRange().begin(),
//                                  graph.masterNodesRange().end()),
//                  [&](auto src) {
//                    NodeData& sdata = graph.getData(src);
//
//                    DGA_max.update(sdata.bc);
//                    DGA_min.update(sdata.bc);
//                    DGA_sum += sdata.bc;
//                  },
//                  galois::no_stats(), galois::loopname("Sanity"));
__global__
void Sanity_kernel(CSRGraph graph, unsigned int __begin, unsigned int __end, float * p_bc, HGAccumulator<float> DGAccumulator_sum, HGReduceMax<float> DGAccumulator_max, HGReduceMin<float> DGAccumulator_min) {
    unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  //const unsigned __kernel_tb_size = TB_SIZE;
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



// TODO: WESTON: implement bitset reset method
//void bitset_dependency_reset_cuda(struct CUDA_Context* ctx);
//void bitset_minDistances_reset_cuda(struct CUDA_Context* ctx);



__global__ void Dump_kernel(
       CSRGraph graph, 
       unsigned int __begin, 
       unsigned int __end, 
       int checkpoint_num,
       int numSourcesPerRound,
       uint32_t * p_roundIndexToSend,
       uint32_t * p_minDistances,
       float* p_bc)
{
    unsigned tid = TID_1D;
    unsigned nthreads = TOTAL_THREADS_1D;

    //const unsigned __kernel_tb_size = TB_SIZE;
    index_type src_end;

    src_end = __end;
    uint64_t node_id = tid;
    for (index_type src = __begin + tid; src < src_end; src += nthreads)
    {
      //for (unsigned i = 0; i < numSourcesPerRound; i++) {
      //  if (graph.node_data[src] != cuda_nodes_to_consider[i]) {
      //    p_bc[src] += p_dependency[src + (i * graph.nnodes)];
      //  }
      //}

      printf("%d CHECKPOINT node: %09lu roundIndexToSend: %u bc: %g\n",
          checkpoint_num,
          //graph.node_data[src],
          node_id,
          p_roundIndexToSend[src],
          p_bc[src]);

      for (unsigned i = 0; i < numSourcesPerRound; i++) {

        printf("%d CHECKPOINT node: %09lu sourceIndex: %d minDistance: %u\n",
            checkpoint_num,
            node_id,
            i,
            p_minDistances[src + (i * graph.nnodes)]);

      }

      node_id += nthreads;
    }

}


void DumpAlgorithmCheckpoint_cuda(
    struct CUDA_Context*  ctx,
    int checkpoint_num)
{
  dim3 blocks;
  dim3 threads;

  kernel_sizing(blocks, threads);

  // Make device vector for local_nodes_to_consider
  
  Dump_kernel <<<blocks, threads>>>(
          ctx->gg, 
          0, 
          ctx->gg.nnodes, 
          checkpoint_num,
          ctx->vectorSize,
          ctx->roundIndexToSend.data.gpu_wr_ptr(),
          ctx->minDistances.data.gpu_wr_ptr(),
          ctx->bc.data.gpu_wr_ptr());
  cudaDeviceSynchronize();
  check_cuda_kernel;
}
