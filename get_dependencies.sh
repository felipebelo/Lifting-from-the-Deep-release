#!/bin/bash

echo 'Compiling dependencies...'
cd ./dependencies/
./upright_fast_setup build_ext --inplace
rm -rf build
cd ..
echo 'Done'
