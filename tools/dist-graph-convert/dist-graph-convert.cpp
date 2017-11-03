/** Distributed graph converter -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2017, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * Distributed graph converter tool based on shared-memory graph convert.
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

#include <utility>

#include "galois/DistGalois.h"
#include "llvm/Support/CommandLine.h"

#include "dist-graph-convert-helpers.h"

namespace cll = llvm::cl;

enum ConvertMode {
  edgelist2gr,
  gr2wgr,
  gr2tgr,
  gr2sgr,
  gr2cgr
};

enum EdgeType {
  uint32_,
  void_
};

static cll::opt<std::string> inputFilename(cll::Positional, 
    cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> outputFilename(cll::Positional,
    cll::desc("<output file>"), cll::init(std::string()));
static cll::opt<EdgeType> edgeType("edgeType", 
    cll::desc("Input/Output edge type:"),
    cll::values(
      clEnumValN(EdgeType::uint32_, "uint32", 
                 "32 bit unsigned int edge values"),
      clEnumValN(EdgeType::void_, "void", 
                 "no edge values"),
      clEnumValEnd), 
    cll::init(EdgeType::void_));
static cll::opt<ConvertMode> convertMode(cll::desc("Conversion mode:"),
    cll::values(
      clEnumVal(edgelist2gr, "Convert edge list to binary gr"),
      clEnumVal(gr2wgr, "Convert unweighted binary gr to weighted binary gr "
                         "(in-place)"),
      clEnumVal(gr2tgr, "Convert binary gr to transpose binary gr"),
      clEnumVal(gr2sgr, "Convert binary gr to symmetric binary gr"),
      clEnumVal(gr2cgr, "Convert binary gr to binary gr without self-loops "
                        "or multi-edges; edge data will be ignored"),
      clEnumValEnd
    ), cll::Required);
static cll::opt<unsigned> totalNumNodes("numNodes", 
                                        cll::desc("Nodes in input graph"),
                                        cll::init(0));
static cll::opt<unsigned> threadsToUse("t", cll::desc("Threads to use"),
                                       cll::init(1));
static cll::opt<bool> editInPlace("inPlace", 
                                  cll::desc("Flag specifying conversion is in "
                                            "place"),
                                  cll::init(false));


struct Conversion { };

////////////////////////////////////////////////////////////////////////////////
// BEGIN CONVERT CODE/STRUCTS
////////////////////////////////////////////////////////////////////////////////

/**
 * Convert 1: figure out edge type, then call convert with edge type as
 * an additional template argument.
 */
template<typename C>
void convert() {
  C c;

  switch (edgeType) {
    case EdgeType::uint32_: convert<uint32_t>(c, c); break;
    case EdgeType::void_: convert<void>(c, c); break;
    default: abort();
  };
}

/**
 * Convert 2 called from convert above: calls convert from the appropriate
 * structure
 */
template<typename EdgeTy, typename C>
void convert(C& c, Conversion) {
  auto& net = galois::runtime::getSystemNetworkInterface();

  if (net.ID == 0) {
    printf("Input: %s; Output: %s\n", inputFilename.c_str(), 
                                      outputFilename.c_str());
  }

  galois::runtime::getHostBarrier().wait();

  galois::StatTimer convertTimer("Convert Time", "convert"); 
  convertTimer.start();
  c.template convert<EdgeTy>(inputFilename, outputFilename);
  convertTimer.stop();

  if (net.ID == 0) {
    galois::gPrint("Done with convert\n");
  }
}

/**
 * Converts an edge list to a Galois binary graph.
 */
struct Edgelist2Gr : public Conversion {

  template<typename EdgeTy>
  void convert(const std::string& inputFile, const std::string& outputFile) {
    GALOIS_ASSERT((totalNumNodes != 0), "edgelist2gr needs num nodes");
    GALOIS_ASSERT(!(outputFile.empty()), "edgelist2gr needs an output file");

    auto& net = galois::runtime::getSystemNetworkInterface();
    uint64_t hostID = net.ID;

    std::ifstream edgeListFile(inputFile.c_str());
    uint64_t fileSize = getFileSize(edgeListFile);
    if (hostID == 0) {
      printf("File size is %lu\n", fileSize);
    }

    uint64_t localStartByte;
    uint64_t localEndByte;
    std::tie(localStartByte, localEndByte) = determineByteRange(edgeListFile,
                                                                fileSize);
    //printf("[%lu] Byte start %lu byte end %lu, num bytes %lu\n", hostID, 
    //               localStartByte, localEndByte, localEndByte - localStartByte);
    // load edges into a vector
    std::vector<uint32_t> localEdges = loadEdgesFromEdgeList<EdgeTy>(
                                           edgeListFile, localStartByte,
                                           localEndByte, totalNumNodes);
    edgeListFile.close();

    uint64_t totalEdgeCount = accumulateValue(getNumEdges<EdgeTy>(localEdges));
    if (hostID == 0) {
      printf("Total num edges %lu\n", totalEdgeCount);
    }
    assignAndWriteEdges<EdgeTy>(localEdges, totalNumNodes, totalEdgeCount, 
                                outputFile);
    galois::runtime::getHostBarrier().wait();
  }

};

/**
 * Transpose a Galois binary graph.
 */
struct Gr2TGr : public Conversion {

  template<typename EdgeTy>
  void convert(const std::string& inputFile, const std::string& outputFile) {
    GALOIS_ASSERT(!(outputFile.empty()), "gr2tgr needs an output file");
    auto& net = galois::runtime::getSystemNetworkInterface();
    uint32_t hostID = net.ID;

    uint64_t totalNumNodes;
    uint64_t totalNumEdges;
    std::tie(totalNumNodes, totalNumEdges) = 
        readV1GrHeader(inputFile, std::is_void<EdgeTy>::value);


    // get "read" assignment of nodes (i.e. nodes this host is responsible for)
    Uint64Pair nodesToRead;
    Uint64Pair edgesToRead;
    std::tie(nodesToRead, edgesToRead) = getNodesToReadFromGr(inputFile);
    printf("[%u] Reads nodes %lu to %lu\n", hostID, nodesToRead.first,
                                                    nodesToRead.second);
    printf("[%u] Reads edges %lu to %lu (count %lu)\n", hostID, 
           edgesToRead.first, edgesToRead.second, 
           edgesToRead.second - edgesToRead.first);

    // read edges of assigned nodes using MPI_Graph, load into the same format
    // used by edgelist2gr; key is to do it TRANSPOSED
    std::vector<uint32_t> localEdges =
        loadTransposedEdgesFromMPIGraph<EdgeTy>(inputFile, nodesToRead, 
                                                edgesToRead, totalNumNodes, 
                                                totalNumEdges);
    // sanity check
    uint64_t totalEdgeCount = accumulateValue(getNumEdges<EdgeTy>(localEdges));
    GALOIS_ASSERT(totalEdgeCount == totalNumEdges, 
                  "edges from metadata doesn't match edges in memory");
    assignAndWriteEdges<EdgeTy>(localEdges, totalNumNodes, totalNumEdges, 
                                outputFile);

    galois::runtime::getHostBarrier().wait();
  }

};

/**
 * Makes a Galois binary graph symmetric (i.e. add a directed edge in the
 * opposite direction for every directed edge)
 */
struct Gr2SGr : public Conversion {

  template<typename EdgeTy>
  void convert(const std::string& inputFile, const std::string& outputFile) {
    GALOIS_ASSERT(!(outputFile.empty()), "gr2sgr needs an output file");
    auto& net = galois::runtime::getSystemNetworkInterface();
    uint32_t hostID = net.ID;

    uint64_t totalNumNodes;
    uint64_t totalNumEdges;
    std::tie(totalNumNodes, totalNumEdges) = 
        readV1GrHeader(inputFile, std::is_void<EdgeTy>::value);


    // get "read" assignment of nodes (i.e. nodes this host is responsible for)
    Uint64Pair nodesToRead;
    Uint64Pair edgesToRead;
    std::tie(nodesToRead, edgesToRead) = getNodesToReadFromGr(inputFile);
    printf("[%u] Reads nodes %lu to %lu\n", hostID, nodesToRead.first,
                                                    nodesToRead.second);
    printf("[%u] Reads edges %lu to %lu (count %lu)\n", hostID, 
           edgesToRead.first, edgesToRead.second, 
           edgesToRead.second - edgesToRead.first);

    // read edges of assigned nodes using MPI_Graph, load into the same format
    // used by edgelist2gr; key is to load one edge as 2 edges (i.e. symmetric)
    std::vector<uint32_t> localEdges =
        loadSymmetricEdgesFromMPIGraph<EdgeTy>(inputFile, nodesToRead, 
                                               edgesToRead, totalNumNodes, 
                                               totalNumEdges);
    // sanity check
    uint64_t doubleEdgeCount = accumulateValue(getNumEdges<EdgeTy>(localEdges));
    GALOIS_ASSERT(doubleEdgeCount == 2 * totalNumEdges, 
                  "data needs to have twice as many edges as original graph");
    assignAndWriteEdges<EdgeTy>(localEdges, totalNumNodes, doubleEdgeCount, 
                                outputFile);
    galois::runtime::getHostBarrier().wait();
  }
};


/**
 * Adds random weights to a Galois binary graph.
 */
struct Gr2WGr : public Conversion {
  template<typename EdgeTy>
  void convert(const std::string& inputFile, const std::string& outputFile) {
    GALOIS_ASSERT(outputFile.empty(), "gr2wgr doesn't take an output file");
    GALOIS_ASSERT(editInPlace, "You must use -inPlace with gr2wgr");

    uint64_t totalNumNodes;
    uint64_t totalNumEdges;
    std::tie(totalNumNodes, totalNumEdges) = 
        readV1GrHeader(inputFile, std::is_void<EdgeTy>::value);


    uint64_t localEdgeBegin;
    uint64_t localEdgeEnd;
    std::tie(localEdgeBegin, localEdgeEnd) = getLocalAssignment(totalNumEdges);
    
    uint32_t hostID = galois::runtime::getSystemNetworkInterface().ID;
    printf("[%u] Responsible for edges %lu to %lu\n", hostID, localEdgeBegin,
                                                              localEdgeEnd);
    
    // get edge data to write (random numbers) and get location to start
    // write
    uint64_t numLocalEdges = localEdgeEnd - localEdgeBegin;
    std::vector<uint32_t> edgeDataToWrite = generateRandomNumbers(numLocalEdges,
                                            hostID, 1, 100);
    GALOIS_ASSERT(edgeDataToWrite.size() == numLocalEdges);
    uint64_t byteOffsetToEdgeData = getOffsetToLocalEdgeData(totalNumNodes,
                                      totalNumEdges, localEdgeBegin);

    // do edge data writing
    MPI_File grInPlace;
    MPICheck(MPI_File_open(MPI_COMM_WORLD, inputFile.c_str(), 
                           MPI_MODE_RDWR, MPI_INFO_NULL, &grInPlace));
    writeEdgeDataData(grInPlace, byteOffsetToEdgeData, edgeDataToWrite);
    // if host 0 update header with edge size
    if (hostID == 0) {
      uint64_t edgeSize = 4;
      MPICheck(MPI_File_write_at(grInPlace, sizeof(uint64_t), 
                                &edgeSize, 1, MPI_UINT64_T, MPI_STATUS_IGNORE));
    }
    MPICheck(MPI_File_close(&grInPlace));
  }
};

/**
 * Cleans graph (no multi-edges, no self-loops).
 *
 * ONLY WORKS ON GRAPHS WITH NO EDGE DATA. (If it does have edge data, it will
 * be ignored.)
 */
struct Gr2CGr : public Conversion {
  template<typename EdgeTy>
  void convert(const std::string& inputFile, const std::string& outputFile) {
    GALOIS_ASSERT(std::is_void<EdgeTy>::value, 
                  "Edge type must be void to clean graph");
    GALOIS_ASSERT(!(outputFile.empty()), "gr2cgr needs an output file");

    auto& net = galois::runtime::getSystemNetworkInterface();
    uint32_t hostID = net.ID;

    uint64_t totalNumNodes;
    uint64_t totalNumEdges;
    std::tie(totalNumNodes, totalNumEdges) = 
        readV1GrHeader(inputFile, std::is_void<EdgeTy>::value);

    // get "read" assignment of nodes (i.e. nodes this host is responsible for)
    Uint64Pair nodesToRead;
    Uint64Pair edgesToRead;
    std::tie(nodesToRead, edgesToRead) = getNodesToReadFromGr(inputFile);
    printf("[%u] Reads nodes %lu to %lu\n", hostID, nodesToRead.first,
                                                    nodesToRead.second);
    printf("[%u] Reads edges %lu to %lu (count %lu)\n", hostID, 
           edgesToRead.first, edgesToRead.second, 
           edgesToRead.second - edgesToRead.first);

    std::vector<uint32_t> localEdges = loadCleanEdgesFromMPIGraph(inputFile, 
                                           nodesToRead, edgesToRead, 
                                           totalNumNodes, totalNumEdges);
    uint64_t cleanEdgeCount = accumulateValue(getNumEdges<EdgeTy>(localEdges));
    GALOIS_ASSERT(cleanEdgeCount < totalNumEdges, 
                  "clean should not increase edge count");

    if (hostID == 0) {
      galois::gPrint("From ", totalNumEdges, " edges to ", cleanEdgeCount, 
                     "edges\n");
    }

    assignAndWriteEdges<EdgeTy>(localEdges, totalNumNodes, cleanEdgeCount, 
                                outputFile);
    galois::runtime::getHostBarrier().wait();
  }
};

int main(int argc, char** argv) {
  galois::DistMemSys G;
  llvm::cl::ParseCommandLineOptions(argc, argv);
  galois::setActiveThreads(threadsToUse);

  switch (convertMode) {
    case edgelist2gr: 
      convert<Edgelist2Gr>(); break;
    case gr2wgr: 
      convert<Gr2WGr>(); break;
    case gr2tgr: 
      convert<Gr2TGr>(); break;
    case gr2sgr: 
      convert<Gr2SGr>(); break;
    case gr2cgr: 
      convert<Gr2CGr>(); break;
    default: abort();
  }
  return 0;
}