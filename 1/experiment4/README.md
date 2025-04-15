# Experiment 4 - Dropout

## Parameters
- hidden_size: 128
- num_blocks: 3
- use_skip_connection: True
- use_batch_norm: True
- dropout_values_tested: [0.01, 0.1, 0.2, 0.5, 0.9]
- best_dropout: 0.2
- num_epochs: 10
- batch_size: 32
- learning_rate: 0.01
- seed: 42

## Results
- Best epoch: 9
- Train loss: 0.1911
- Test loss: 0.1776
- Train AUC: 0.9264
- Test AUC: 0.9314

