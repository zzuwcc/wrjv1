#!/bin/bash

python ./MACA/algorithm/ippo/Runner_detect.py --total_steps 200 --number 0 --map_name "zc_easy"

python ./MACA/algorithm/ippo/TestPolicy_detect.py --number 0 --map_name "zc_easy"

