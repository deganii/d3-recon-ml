#!/bin/sh

HOME=/home/ubuntu/d3
MODEL_NAME=$1
PROC_DIR=$HOME/src/processing
LOG_DIR=$HOME/logs/$MODEL_NAME

mkdir $HOME/logs
mkdir $HOME/logs/$MODEL_NAME

cd $PROC_DIR && nohup python train.py > $LOG_DIR/$MODEL_NAME.out 2> $LOG_DIR/$MODEL_NAME.err < /dev/null &