#!/usr/bin/bash

num_iterations=$1 # this is for testing purposes
if [[ $# != 1 ]]; then
    echo "usage: ./scripts/run.sh <num_iterations>"
    exit
fi

python run.py --data_dir "stanfordSentimentTreebank" \
    --lr 0.8 --num_iterations $num_iterations --using_negative_sample\


python run.py --data_dir "stanfordSentimentTreebank" \
    --lr 0.8 --num_iterations $num_iterations 


