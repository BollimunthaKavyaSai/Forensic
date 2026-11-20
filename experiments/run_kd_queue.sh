#!/bin/bash
# two sequential queues of KronoDroid jobs (fast models), then FT-Transformer on family folds
export PYTHONUTF8=1 NTHREADS=3
( python run_kronodroid.py family all > logk_family_all.log 2>&1; python run_kronodroid.py temporal all > logk_temporal_all.log 2>&1; MAX_FOLDS=5 python run_kronodroid.py family static > logk_family_static.log 2>&1 ) &
( python run_kronodroid.py random all > logk_random_all.log 2>&1; python run_kronodroid.py temporal static > logk_temporal_static.log 2>&1 ) &
wait
NTHREADS=8 FT_ONLY=1 MAX_FOLDS=5 python run_kronodroid.py family all > logk_family_all_ft.log 2>&1
NTHREADS=8 FT_ONLY=1 MAX_FOLDS=5 python run_kronodroid.py random all > logk_random_all_ft.log 2>&1
echo KD_ALL_DONE
