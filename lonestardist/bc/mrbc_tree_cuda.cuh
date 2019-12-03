/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#pragma once

//#include "mrbc_bitset.hh"
#include "mrbc_bitset_cuda.cuh"

// TODO: WESTON: use __forceinline__ or __inline__ ?

const uint32_t infinity = std::numeric_limits<uint32_t>::max() >> 2;

/**
 * Binary tree class to make finding a source's message to send out during MRBC
 * easier.
 */
class MRBCTree_cuda {
  
  using BitSet = MRBCBitSet_cuda;
  //using FlatMap = boost::container::flat_map<uint32_t, BitSet,
  //                                            std::less<uint32_t>,
  //                                            galois::gstl::Pow2Alloc<std::pair<uint32_t, BitSet>>>;

  //! map to a bitset of nodes that belong in a particular distance group
  uint32_t* dist_vector;
  BitSet* bitset_vector;

  // ancillary fields for vector management
  size_t size;
  size_t capacity;

  //! number of sources that have already been sent out
  uint32_t numSentSources;
  //! number of non-infinity values (i.e. number of sources added already)
  uint32_t numNonInfinity;
  //! indicates if zero distance has been reached for backward iteration
  bool zeroReached;

  //! reverse iterator over map
  //using TreeIter = typename FlatMap::reverse_iterator;
  //! Current iterator for reverse map
  size_t curKey;
  //! End key for reverse map iterator
  size_t endCurKey;

public:
/*** InitializeIteration *****************************************************/

  /**
   * Reset the map, initialize all distances to infinity, and reset the "sent"
   * vector and num sent sources.
   */
  __device__ void initialize() {

    if (dist_vector) {
      free(dist_vector);
    }
    if (bitset_vector) {
      free(bitset_vector);
    }

    size = 0;
    capacity = 0;
    // reset number of sent sources
    numSentSources = 0;
    // reset number of non infinity sources that exist
    numNonInfinity = 0;
    // reset the flag for backward phase
    zeroReached = false;
  }

  __device__ void push_back(uint32_t dist) {
    if (size >= capacity) {
      size_t new_capacity;
      if (capacity == 0) {
        // TODO: WESTON: investigate this default capacity
        new_capacity = 8;
      } else {
        new_capacity = capacity * 2;
      }

      uint32_t* new_dist_vector = (uint32_t*) malloc(new_capacity * sizeof(uint32_t));
      BitSet* new_bitset_vector = (BitSet*) malloc(new_capacity * sizeof(BitSet));

      if (capacity > 0) {
        // copy old vectors to new
        memcpy(dist_vector, new_dist_vector, capacity * sizeof(uint32_t));
        memcpy(bitset_vector, new_bitset_vector, capacity * sizeof(BitSet));

        free(dist_vector);
        free(bitset_vector);
      }

      dist_vector = new_dist_vector;
      bitset_vector = new_bitset_vector;

      capacity = new_capacity;
    }

    dist_vector[size] = dist;
    bitset_vector[size].reset();
    size++;
  }

  __device__ size_t find_index(uint32_t dist) {
    for (size_t i = 0; i < size; i++) {
      if (dist_vector[i] == dist) {
        return i;
      }
    }
    return size;
  }

  __device__ void sort_by_dist() {
    for (size_t i = 0; i < size-1; i++) {

      uint32_t minDist = dist_vector[i];
      uint32_t minDistIndex = i;
      // find minimum element
      for (size_t j = i+1; j < size; j++) {
         if (dist_vector[j] < minDist) {
           minDist = dist_vector[j];
           minDistIndex = j;
         }
      }

      // swap i'th element with min element
      if (minDistIndex != i) {
        uint32_t dist_temp = dist_vector[minDistIndex];
        dist_vector[minDistIndex] = dist_vector[i];
        dist_vector[i] = dist_temp;

        BitSet::swap(&bitset_vector[minDistIndex], &bitset_vector[i]);
      }

    }
  }

  /**
   * Assumes you're adding a NEW distance; i.e. there better not be a duplicate
   * of index somewhere.
   */
   // TODO: WESTON: messes up sorted order
  __device__ void setDistance(uint32_t index, uint32_t newDistance) {
    // Only for iterstion initialization
    // assert(newDistance == 0);
    // assert(distanceTree[newDistance].size() == numSourcesPerRound);
    
    //distanceTree[newDistance].set_indicator(index);
    push_back(newDistance);
    bitset_vector[size-1].set_indicator(index);

    numNonInfinity++;
  }

/*** FindMessageToSync ********************************************************/

  /**
   * Get the index that needs to be sent out this round given the round number.
   */
  __device__ uint32_t getIndexToSend(uint32_t roundNumber) {
    uint32_t distanceToCheck = roundNumber - numSentSources;
    uint32_t indexToSend = infinity;

    size_t setIter = find_index(distanceToCheck);
    if (setIter != size) {
      BitSet& setToCheck = bitset_vector[setIter];
      auto index = setToCheck.getIndicator();
      if (index != setToCheck.npos) {
        indexToSend = index;
      }
    }
    return indexToSend;
  }

  /**
   * Return true if potentially more work exists to be done
   */
  __device__ bool moreWork() { return numNonInfinity > numSentSources; }

/*** ConfirmMessageToSend *****************************************************/

  /**
   * Note that a particular source's message has already been sent in the data
   * structure and increment the number of sent sources.
   */
  __device__ void markSent(uint32_t roundNumber) {
    uint32_t distanceToCheck = roundNumber - numSentSources;
    size_t index = find_index(distanceToCheck);
    BitSet& setToCheck = bitset_vector[index];
    setToCheck.forward_indicator();

    numSentSources++;
  }

/*** SendAPSPMessages *********************************************************/

  /**
   * Update the distance map: given an index to update as well as its old 
   * distance, remove the old distance and replace with new distance.
   */
  __device__ void setDistance(uint32_t index, uint32_t oldDistance, uint32_t newDistance) {
    if (oldDistance == newDistance) {
      return;
    }

    size_t setIter = find_index(oldDistance);
    bool existed = false;
    // if it exists, remove it
    if (setIter != size) {
      BitSet& setToChange = bitset_vector[setIter];
      existed = setToChange.test_set_indicator(index, false); // Test, set, update
    }

    // if it didn't exist before, add to number of non-infinity nodes
    if (!existed) {
      numNonInfinity++;
    }

    // asset(distanceTree[newDistance].size() == numSourcesPerRound);
    size_t newDistanceIndex = find_index(newDistance);
    if (newDistanceIndex > size) {
      push_back(newDistance);
      newDistanceIndex = size-1;
    }
    bitset_vector[newDistanceIndex].set_indicator(index);

  }

/*** RoundUpdate **************************************************************/

  /**
   * Begin the setup for the back propagation phase by setting up the 
   * iterators.
   */
  __device__ void prepForBackPhase() {
    // sort by distance:
    sort_by_dist();

    curKey = size-1;
    endCurKey = (uint64_t)(-1);

    if (curKey != endCurKey) {
      // find non-empty distance if first one happens to be empty
      if (bitset_vector[curKey].none()) {
        for (--curKey; curKey != endCurKey && bitset_vector[curKey].none(); --curKey);
      }
    }

    // setup if not empty
    if (curKey != endCurKey) {
      BitSet& curSet = bitset_vector[curKey];
      #ifdef FLIP_MODE
        curSet.flip();
      #endif
      curSet.backward_indicator();
    }
  }

/*** BackFindMessageToSend *****************************************************/

  /**
   * Given a round number, figure out which index needs to be sent out for the
   * back propagation phase.
   */
  __device__ uint32_t backGetIndexToSend(const uint32_t roundNumber, 
                              const uint32_t lastRound) {
    uint32_t indexToReturn = infinity;

    while (curKey != endCurKey) {
      uint32_t distance = dist_vector[curKey];
      if ((distance + numSentSources - 1) != (lastRound - roundNumber)){
        // round to send not reached yet; get out
        return infinity;
      }

      if (distance == 0) {
        zeroReached = true;
        return infinity;
      }

      BitSet& curSet = bitset_vector[curKey];
      if (!curSet.nposInd()) {
          // this number should be sent out this round
          indexToReturn = curSet.backward_indicator();
          numSentSources--;
          break;
      } else {
        // set exhausted; go onto next set
        for (--curKey; curKey != endCurKey && bitset_vector[curKey].none(); --curKey);

        // if another set exists, set it up, else do nothing
        if (curKey != endCurKey) {
          BitSet& nextSet = bitset_vector[curKey];
          #ifdef FLIP_MODE
            nextSet.flip();
          #endif
          nextSet.backward_indicator();
        }
      }
    }

    if (curKey == endCurKey) {
      //assert(numSentSources == 0);
      // TODO: WESTON: cuda assert?
    }

    return indexToReturn;
  }

  /**
   * Returns zeroReached variable.
   */
  __device__ bool isZeroReached() {
    return zeroReached;
  }
};

