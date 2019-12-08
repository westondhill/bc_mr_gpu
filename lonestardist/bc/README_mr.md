#GPU MRBC

The GPU implementation is single node only.
To run the GPU version, use the command:
./bc_mr <graph> -num_nodes=1 -pset="g"

The argument "numSourcesPerRound" is currently ignored and instead replaced by
"NUM_SOURCES_PER_ROUND" which is defined in bc_mr_parameters.h

This value needs <=64 and a factor of the number of nodes in the graph. It must be compiled to work.
