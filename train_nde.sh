#!/bin/bash

# This trains a neural conditional density estimator with a neural spline
# coupling flow.

# Settings are the same as in the paper. It will take several days to run, so
# you may want to decrease the number of epochs or the size of the network.

# Feel free to change the settings, but only the nde flow option will work at
# present.

python -m lfigw.gwpe train new nde \
    --data_dir waveforms/GW150914/ \
    --model_dir models/GW150914/ \
    --nbins 8 \
    --num_transform_blocks 10 \
    --nflows 15 \
    --batch_norm \
    --lr 0.0002 \
    --epochs 500 \
    --distance_prior_fn uniform_distance \
    --distance_prior 100.0 1000.0 \
    --hidden_dims 512 \
    --truncate_basis 100 \
    --activation elu \
    --lr_anneal_method cosine