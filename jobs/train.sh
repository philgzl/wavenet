#!/bin/sh

OPTS=`getopt -o f -l force -- "$@"`
if [ $? -ne 0 ]; then exit 1; fi
eval set -- "$OPTS"


FORCE=false
while true
do
  case "$1" in
    -f | --force ) FORCE=true; shift ;;
    -- ) shift; break ;;
  esac
done

if [ "$FORCE" = true ]
then
    COMMAND="python scripts/train.py -f"
else
    COMMAND="python scripts/train.py"
fi

mkdir -p jobs/logs
bash jobs/submit.sh jobs/job.sh "$COMMAND"
