#!/bin/bash
export PYTHONUTF8=1 NTHREADS=3
SEEDS=0,1 python run_atlas.py cross_vector > loga_cross.log 2>&1
SEEDS=0,1 FEATSET=nopath python run_atlas.py cross_vector > loga_cross_nopath.log 2>&1
SEEDS=0 python run_atlas.py loso > loga_loso.log 2>&1
echo ATLAS_ALL_DONE
