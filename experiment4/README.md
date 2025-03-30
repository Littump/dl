# Experiment 4 - Dropout

## Parameters
- hidden_size: 128
- num_blocks: 3
- use_skip_connection: True
- use_batch_norm: True
- dropout_values_tested: [0.01, 0.1, 0.2, 0.5, 0.9]
- best_dropout: 0.01
- num_epochs: 10
- batch_size: 32
- learning_rate: 0.01
- seed: 42

## Results
- Best epoch: 6
- Train loss: 0.1894
- Test loss: 0.1816
- Train AUC: 0.9277
- Test AUC: 0.9305
