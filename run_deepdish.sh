#!/usr/bin/env bash
# Run every experiment on deepdish.
#
#   nohup bash run_deepdish.sh > logs/all.log 2>&1 &     # starts everything in the background
#   tail -f logs/*.log                                   # watch progress
#
# DEVICE=cpu by default: the policy is tiny and acts once per frame, so a shared GPU only adds
# queueing delay. Use DEVICE=cuda GPU=<id> (see nvidia-smi) to run on a free card instead.
# If a Pong job dies, re-run this script: Pong resumes from its last checkpoint.
set -euo pipefail
cd "$(dirname "$0")"
export DEVICE="${DEVICE:-cpu}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p logs

# Pong is the long job (hours), so start it first; both parts run in parallel.
nohup python pg/train.py --game pong            > logs/pong_part1.log 2>&1 &
nohup python pg/train.py --game pong --baseline > logs/pong_part2.log 2>&1 &

# Fast parts: DP, CartPole, validation suites.
python dp/run_dp.py                           > logs/dp.log 2>&1
python dp/validate_dp.py                      > logs/dp_validation.log 2>&1
python pg/train.py --game cartpole            > logs/cartpole_part1.log 2>&1
python pg/train.py --game cartpole --baseline > logs/cartpole_part2.log 2>&1
python pg/validate_pg.py                      > logs/pg_validation.log 2>&1
echo "fast parts done: $(grep -h SUMMARY logs/*validation.log)"
echo "Pong keeps running in the background; when both finish run: python pg/plots.py"
