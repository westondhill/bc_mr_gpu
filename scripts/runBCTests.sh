#!/bin/bash

SOURCEDIR=~/Galois/
BUILDDIR=~/Galois/build/

#Make sure the modules are loaded
source $SOURCEDIR/scripts/my_iss_load_modules.sh
cd $BUILDDIR

# Build Galois in release mode
# cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_DIST_GALOIS=1 -DENABLE_HETERO_GALOIS=1 ..

# Make bc_level for comparison
# make -j bc_level

# Make bc_mr to get your changes
make -j bc_mr


cd $BUILDDIR/lonestardist/bc

# Run bc_mr
./bc_mr /net/ohm/export/iss/inputs/scalefree/rmat10.gr -num_nodes=1 > output_bc_mr.txt

# Run with bc_level
./bc_level /net/ohm/export/iss/inputs/scalefree/rmat10.gr -num_nodes=1 > output_bc_level.txt

# Write Comparison
echo "== bc_mr ==" > output.txt
grep -B 2 -A 1 '^BC sum is' output_bc_mr.txt >> output.txt
echo "== bc_level ==" >> output.txt
grep -B 3 '^BC sum is' output_bc_level.txt >> output.txt

cat output.txt

