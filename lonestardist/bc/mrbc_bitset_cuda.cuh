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

#include "bc_mr_parameters.h"
// TODO: WESTON: is the /cuda/ part necessary?
#include "galois/cuda/DynamicBitset.h"
//#include "galois/DynamicBitset.h"
//#include <boost/random/detail/integer_log2.hpp>

/** OPTIONS **********/
/** 1. Optimized mode: enable ONLY ONE of them at most **/
// #define REVERSE_MODE //! Not applicable for v6
// #define FLIP_MODE  //! Not applicable for v6
/** 2. Do you need an indicator? **/
#define USE_INDICATOR
/*********************/

/**
 * Derivate from DynamicBitSet
 **/
class MRBCBitSet_cuda : public DynamicBitset {
  // using Base = galois::DynamicBitSet;
  // @DynamicBitSet (protected)
  // using Base::bit_vector;
  // using Base::num_bits;
  // using Base::num_bits_capacity;

  #ifdef USE_INDICATOR
    //! indicate the index of bit to process
    size_t indicator;
  #endif

  // Member functions
  __device__ __inline__ size_t get_word(size_t pos) const { return pos < num_bits_capacity? 0 : pos / num_bits_capacity; }
  __device__ __inline__ size_t get_offset(size_t pos) const { return pos < num_bits_capacity? pos : pos % num_bits_capacity; }
  __device__ __inline__ uint64_t get_mask(size_t pos) const { return uint64_t(1) << get_offset(pos); }

  #if defined(REVERSE_MODE) || defined(FLIP_MODE)
    size_t reverse(size_t pos) {
      return pos == npos? npos : num_bits - pos - 1;
    }
  #endif

  #ifdef FLIP_MODE
    void flip_recursive(size_t pos) {
      size_t next = find_next(pos);
      if (next != npos)
        flip_recursive(next);
      // do the flip for pos
      uint64_t block = get_word(pos), mask = get_mask(pos);
      uint64_t rBlock = get_word(reverse(pos)), rMask = get_mask(reverse(pos));
      // flip if asymmetrical
      if (!(bit_vector[rBlock] & rMask)) {
        bit_vector[block].fetch_and(~mask);
        size_t r_old = bit_vector[rBlock];
        while (!bit_vector[rBlock].compare_exchange_weak(
          r_old, r_old | rMask, std::memory_order_relaxed));
      }
    }
  #endif

  __device__ static size_t integer_log2(uint64_t x) {
    size_t log = 0;
    while ( (x >> log) > 1) {
      log++;
    }
    return log;
  }

  __device__ size_t right_most_bit(uint64_t w) const {
    // assert(w >= 1);
    return integer_log2(w & -w);
  }

  __device__ size_t left_most_bit(uint64_t w) const {
      return integer_log2(w);
  }

  __device__ size_t find_from_block(size_t first, bool fore=true) const {
    size_t i;
    if (fore) {
      for (i = first; i < vec_size() && bit_vector[i] == 0; i++);
      if (i >= vec_size())
          return npos;
      return i * num_bits_capacity + right_most_bit(bit_vector[i]);
    }
    else {
      for (i = first; i > 0 && bit_vector[i] == 0; i--);
      if (i <= 0 && bit_vector[i] == 0)
        return npos;
      return i * num_bits_capacity + left_most_bit(bit_vector[i]);
    }
  }

public:
  typedef size_t iterator;
  typedef size_t reverse_iterator;

  //! sign for N/A
  static const size_t npos = std::numeric_limits<size_t>::max();

  #ifdef FLIP_MODE
    void flip() {
      size_t first = find_first();
      if (first != npos)
        flip_recursive(first);
    }
  #endif

  #ifdef USE_INDICATOR
  // Accessors
  // TODO: WESTON: USED
  __device__ size_t getIndicator() const { return indicator; }
  // TODO: WESTON: USED
  __device__ void setIndicator(size_t index) { indicator = index; }
  #endif

  // @DynamicBitSet
  #ifdef _GALOIS_DYNAMIC_BIT_SET_
  // using Base::size;
  #else
  __device__ __inline__ size_t size() const { return num_bits; }
  #endif

  //! Constructor which initializes to an empty bitset.
  __device__ MRBCBitSet_cuda() {
    // TODO: WESTON: deal with bitsets less than 64 bits?
    resize(NUM_SOURCES_PER_ROUND);
    #ifdef USE_INDICATOR
      indicator = npos;
    #endif
  }

  #ifdef _GALOIS_DYNAMIC_BIT_SET_
  // using Base::test;
  #else
  // @DynamicBitSet
  __device__ bool test(size_t index) const {
    size_t bit_index = get_word(index);
    uint64_t bit_mask = get_mask(index);
    return ((bit_vector[bit_index] & bit_mask) != 0);
  }
  #endif

  /**
   * Test and then set
   *
   * @returns test result before set
   */
  __device__ bool test_set(size_t pos, bool val=true) {
    bool const ret = test(pos);
    if (ret != val) {
      uint64_t old_val = bit_vector[get_word(pos)];
      if (val) {
        bit_vector[get_word(pos)] = (old_val | get_mask(pos));
      }
      else {
        bit_vector[get_word(pos)] = (old_val & ~get_mask(pos));
      }
    }
    return ret;
  }

  #ifdef _GALOIS_DYNAMIC_BIT_SET_
  // using Base::set;
  #else
  // @DynamicBitSet
  __device__ void set(size_t index) {
    size_t bit_index = get_word(index);
    uint64_t bit_mask = get_mask(index);
    if ((bit_vector[bit_index] & bit_mask) == 0) { // test and set
      size_t old_val = bit_vector[bit_index];
      bit_vector[bit_index] = old_val | bit_mask;
    }
  }
  #endif

  // DISABLED @DynamicBitSet
  __device__ void reset(size_t index) {
    size_t bit_index = get_word(index);
    uint64_t bit_mask = get_mask(index);
    bit_vector[bit_index] &= ~bit_mask;
    // @ Better implementation:
    // while (!bit_vector[bit_index].compare_exchange_weak(
    //   old_val, old_val & ~bit_mask, 
    //   std::memory_order_relaxed));
  }

  #ifdef _GALOIS_DYNAMIC_BIT_SET_
  // using Base::reset;
  // using Base::resize;
  #else
  // @DynamicBitSet
  // TODO: WESTON: USED
  __device__ void reset() { 
    for (size_t i = 0; i < vec_size(); i++) {
      bit_vector[i] = (uint64_t) 0;
    }
  }

  // @DynamicBitSet
  __device__ void resize(uint64_t n) {
    assert(num_bits_capacity == 64); // compatibility with other devices
    //bit_vector.resize((n + num_bits_capacity - 1) / num_bits_capacity);
    alloc(n);
    reset();
  }
  #endif

  __device__ bool none() {
    for (size_t i = 0; i < vec_size(); ++i)
      if (bit_vector[i])
        return false;
    return true;
  }

  #ifdef USE_INDICATOR
    /**
     * Set a bit with the side-effect updating indicator to the first.
     */
     // TODO: WESTON: USED
    __device__ void set_indicator(size_t pos) {
      #ifdef REVERSE_MODE
        set(reverse(pos));
      #else
        set(pos);
      #endif
      if (pos < indicator) {
        indicator = pos;
      }
    }

  // TODO: WESTON: USED
    __device__ bool test_set_indicator(size_t pos, bool val=true) {
      #ifdef REVERSE_MODE
        if (test_set(reverse(pos), val)) {
          if (pos == indicator) {
            forward_indicator();
          }
          return true;
        }
        else
          return false;
      #else
        if (test_set(pos, val)) {
          if (pos == indicator) {
            forward_indicator();
          }
          return true;
        }
        else
          return false;
      #endif
    }

    /**
     * Return true if indicator is npos
     */
    __device__ __inline__ bool nposInd() {
      return indicator == npos;
    }
  #endif
  /**
   * Returns: the lowest index i such as bit i is set, or npos if *this has no on bits.
   */
  __device__ size_t find_first() const {
    return find_from_block(0);
  }

  __device__ size_t find_last() const {
    return find_from_block(vec_size() - 1, false);
  }

  #ifdef REVERSE_MODE
    inline size_t begin() { return reverse(find_last()); }
  #else
    __device__ __inline__ size_t begin() { return find_first(); }
  #endif
  __device__ __inline__ size_t end() { return npos; }

  #if defined(REVERSE_MODE) || defined(FLIP_MODE)
    inline size_t rbegin() { return reverse(find_first()); }
  #else
    __device__ __inline__ size_t rbegin() { return find_last(); }
  #endif
  __device__ __inline__ size_t rend() { return npos; }

  /**
   * Returns: the lowest index i greater than pos such as bit i is set, or npos if no such index exists.
   */
  __device__ size_t find_next(size_t pos) const {
    if (pos == npos) {
      return find_first();
    }
    if (++pos >= size() || size() == 0) {
      return npos;
    }
    size_t curBlock = get_word(pos);
    auto curOffset = get_offset(pos);
    auto seg = bit_vector[curBlock];
    while (seg != bit_vector[curBlock])
      seg = bit_vector[curBlock];
    uint64_t res = seg >> curOffset;
    return res?
      pos + right_most_bit(res) : find_from_block(++curBlock);
  }

  __device__ size_t find_prev(size_t pos) const{
    if (pos >= size()) {
      return find_last();
    }
    // Return npos if no bit set
    if (pos-- == 0 || size() == 0) {
      return npos;
    }
    size_t curBlock = get_word(pos);
    uint64_t res = bit_vector[curBlock] & ((uint64_t(2) << get_offset(pos)) - 1);
    return res?
      curBlock * num_bits_capacity + left_most_bit(res) : 
      (curBlock?
        find_from_block(--curBlock, false) : npos);
  }

  /**
   * To move iterator to the previous set bit, and return the old value.
   */
  __device__ __inline__ size_t forward_iterate(size_t& i) {
    size_t old = i;
    #ifdef REVERSE_MODE
      i = reverse(find_prev(reverse(i)));
    #else
      i = find_next(i);
    #endif
    return old;
  }

  /**
   * To move iterator to the next set bit.
   */
  __device__ __inline__ size_t backward_iterate(size_t& i) {
    size_t old = i;
    #ifdef FLIP_MODE
      i = nposInd()? find_first() : find_next(i);
      return reverse(old);
    #else
      #ifdef REVERSE_MODE
      i = reverse(find_next(reverse(i)));
      #else
      i = find_prev(i);
      #endif
      return old;
    #endif
  }

  #ifdef USE_INDICATOR
    /**
     * To move indicator to the previous set bit, and return the old value.
     */
  // TODO: WESTON: USED
    __device__ size_t forward_indicator() {
      return forward_iterate(indicator);
    }

    /**
     * To move indicator to the next set bit.
     */
     // TODO: WESTON: USED
    __device__ size_t backward_indicator() {
      return backward_iterate(indicator);
    }
  #endif


  // TODO: WESTON: USED
  __device__ static void swap(MRBCBitSet_cuda* a, MRBCBitSet_cuda* b) {
    if (a != b) {
      size_t capacity_temp = a->num_bits_capacity;
      size_t num_bits_temp = a->num_bits;
      uint64_t* bit_vector_temp = a->bit_vector;
      size_t indicator_temp = a->indicator;
    
      a->num_bits_capacity = b->num_bits_capacity;
      a->num_bits = b->num_bits;
      a->bit_vector = b->bit_vector;
      a->indicator = b->indicator;
    
      b->num_bits_capacity = capacity_temp;
      b->num_bits = num_bits_temp;
      b->bit_vector = bit_vector_temp;
      b->indicator = indicator_temp;
    }
  
  }

};
