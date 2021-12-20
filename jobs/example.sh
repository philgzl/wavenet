#!/bin/sh

OPTS=$(getopt \
    --longoptions workers:,mixed-precision \
    --options "" \
    -- "$@"
)
if [ $? -ne 0 ]; then exit 1; fi
eval set -- "$OPTS"

WORKERS=8
MIXED_PRECISION=false
while true
do
  case "$1" in
    --workers ) WORKERS="$2"; shift; shift ;;
    --mixed-precision ) MIXED_PRECISION=true; shift ;;
    -- ) shift; break ;;
  esac
done

mkdir -p jobs/logs

COMMAND="python scripts/example.py $1 $2 $3 --cuda --workers ${WORKERS}"
if [ "$MIXED_PRECISION" = true ]
then
  COMMAND="${COMMAND} --mixed-precision"
fi
bash jobs/submit.sh jobs/job.sh "$COMMAND"
