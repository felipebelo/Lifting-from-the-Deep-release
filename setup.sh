#!/bin/bash

mkdir -p data/saved_sessions
cd data/saved_sessions

echo 'Downloading models...'
wget http://visual.cs.ucl.ac.uk/pubs/liftingFromTheDeep/res/init_session.tar.gz
wget http://visual.cs.ucl.ac.uk/pubs/liftingFromTheDeep/res/prob_model.tar.gz

echo 'Extracting models...'
tar -xvzf init_session.tar.gz
tar -xvzf prob_model.tar.gz
rm -rf init_session.tar.gz
rm -rf prob_model.tar.gz
cd ../..

echo 'Installing dependencies...'
pip install Cython
pip install scikit-image

echo 'Compiling external utilities...'
cd ./dependencies/
./setup build_ext --inplace
rm -rf build
cd ..

echo 'Done'
