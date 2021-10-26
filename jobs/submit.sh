#!/bin/sh
mkdir -p jobs/logs
bsub < $1
