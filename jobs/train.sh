#!/bin/sh

OPTS=$(getopt \
    --longoptions ignore-checkpoint,workers: \
    --options "" \
    -- "$@"
)
if [ $? -ne 0 ]; then exit 1; fi
eval set -- "$OPTS"

IGNORE_CHECKPOINT=false
WORKERS=8
while true
do
  case "$1" in
    --ignore-checkpoint ) IGNORE_CHECKPOINT=true; shift ;;
    --workers ) WORKERS="$2"; shift; shift ;;
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
  bash jobs/submit.sh jobs/job.sh "$COMMAND"
done
