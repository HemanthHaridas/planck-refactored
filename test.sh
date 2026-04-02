#!/bin/bash

for file in ./tests/inputs/casscf_tests/*.hfinp
do
	echo "$file"
	./build/hartree-fock ${file} > ${file%.hfinp}.restart.log
done
