#!/bin/sh

OPTS=$(getopt \
    --longoptions n-samples:, \
    --options n: \
    -- "$@"
)
if [ $? -ne 0 ]; then exit 1; fi
eval set -- "$OPTS"

N_SAMPLES=160000
while true
do
  case "$1" in
    -n | --n-samples ) N_SAMPLES="$2"; shift; shift ;;
    -- ) shift; break ;;
  esac
done

mkdir -p jobs/logs

for INPUT in "$@"
do
  COMMAND="python scripts/generate.py ${INPUT} -n ${N_SAMPLES}"
  bash jobs/submit.sh jobs/job.sh "$COMMAND"
done
