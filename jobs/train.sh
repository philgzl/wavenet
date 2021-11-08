#!/bin/sh

OPTS=$(getopt \
    --longoptions ignore-checkpoint,workers,mixed-precision: \
    --options "" \
    -- "$@"
)
if [ $? -ne 0 ]; then exit 1; fi
eval set -- "$OPTS"

IGNORE_CHECKPOINT=false
WORKERS=8
MIXED_PRECISION=false
while true
do
  case "$1" in
    --ignore-checkpoint ) IGNORE_CHECKPOINT=true; shift ;;
    --workers ) WORKERS="$2"; shift; shift ;;
    --mixed-precision ) MIXED_PRECISION=true; shift ;;
    -- ) shift; break ;;
  esac
done

mkdir -p jobs/logs

for INPUT in "$@"
do
  COMMAND="python scripts/train.py ${INPUT} --cuda --workers ${WORKERS}"
  if [ "$IGNORE_CHECKPOINT" = true ]
  then
    COMMAND="${COMMAND} --ignore-checkpoint"
  fi
  if [ "$MIXED_PRECISION" = true ]
  then
    COMMAND="${COMMAND} --mixed-precision"
  fi
  bash jobs/submit.sh jobs/job.sh "$COMMAND"
done
