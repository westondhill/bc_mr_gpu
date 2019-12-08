#pragma once
// This is required at compile time for the GPU version of MRBC_tree.
// It needs to be <=64 and a factor of the number of graph nodes
#define NUM_SOURCES_PER_ROUND (6)
