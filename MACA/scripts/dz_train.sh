#!/bin/bash

python ./MACA/algorithm/ippo/Runner.py --total_steps 200 --number 0 --map_name "dz_easy"

python ./MACA/algorithm/ippo/TestPolicy.py --number 0 --map_name "dz_easy"

