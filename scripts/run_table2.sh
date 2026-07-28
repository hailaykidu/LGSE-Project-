#!/bin/bash
# Table 2: five systems x two tasks x two languages x five seeds.
#
# The tokenizer/initialization strategy is the only variable. Backbone,
# LAPT procedure, fine-tuning hyperparameters, data splits and seeds are
# held fixed across systems, so a difference in F1 is attributable to
# initialization.
#
# usage: run_table2.sh <language> <task> <lapt-corpus> [systems...]

set -euo pipefail

LANGUAGE=${1:?usage: run_table2.sh <language> <task> <corpus> [systems...]}
TASK=${2:?}
CORPUS=${3:?}
shift 3
SYSTEMS=("${@:-xlmr lapt random_lapt focus_lapt lgse_lapt}")

SEEDS=$(python3 -c "import yaml;print(' '.join(map(str,yaml.safe_load(open('configs/base.yaml'))['evaluation']['seeds'])))")

case "$LANGUAGE-$TASK" in
  amharic-ner)   DATA=data/ner/amharic ;;
  tigrinya-ner)  DATA=data/ner/tigrinya ;;
  amharic-qa)    DATA=data/qa/amqa ;;
  tigrinya-qa)   DATA=data/qa/tigqa_squad ;;
  *) echo "no data dir for $LANGUAGE-$TASK" >&2; exit 1 ;;
esac

for SYSTEM in ${SYSTEMS[@]}; do
  for SEED in $SEEDS; do
    python3 src/training/run_experiment.py \
      --system "$SYSTEM" --task "$TASK" --language "$LANGUAGE" \
      --seed "$SEED" --data-dir "$DATA" --corpus "$CORPUS"
  done
done

python3 scripts/aggregate_results.py --task "$TASK" --language "$LANGUAGE"
