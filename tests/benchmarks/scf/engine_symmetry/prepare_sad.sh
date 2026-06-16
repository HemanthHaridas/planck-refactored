#!/bin/bash

for file in *.hfinp
do
	sed "s/hcore/sad/g" $file > ${file%.hfinp}_sad.hfinp
done
