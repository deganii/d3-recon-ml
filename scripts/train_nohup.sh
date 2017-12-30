HOME = /home/ubuntu/d3

mkdir %HOME%/logs
mkdir %HOME%/logs/%MODEL_NAME%

LOG_DIR = %HOME%/logs/%MODEL_NAME%
nohup python %HOME%/src/processing/train.py > %LOG_DIR%/%MODEL_NAME%.out 2> %LOG_DIR%/%MODEL_NAME%.err < /dev/null &