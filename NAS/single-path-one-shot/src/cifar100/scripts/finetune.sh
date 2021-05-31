#!/bin/bash

python train_sandwich_rule.py --finetune --epochs 30
python train_sandwich_rule.py --finetune --epochs 30 --distill 