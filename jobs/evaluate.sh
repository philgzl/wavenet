#!/bin/sh

OPTS=$(getopt \
    --longoptions workers:, \
    --options "" \
    -- "$@"
)
if [ $? -ne 0 ]; then exit 1; fi
eval set -- "$OPTS"

WORKERS=8
while true
do
  case "$1" in
    --workers ) WORKERS="$2"; shift; shift ;;
    -- ) shift; break ;;
  esac
done

mkdir -p jobs/logs

COMMAND="python scripts/evaluate.py $1 $2 --workers ${WORKERS}"
bash jobs/submit.sh jobs/job.sh "$COMMAND"
