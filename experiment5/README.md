# Experiment 5 - Weight Decay and Learning Rate

## Parameters
- hidden_size: 128
- num_blocks: 3
- use_skip_connection: True
- use_batch_norm: True
- dropout_p: 0.2
- weight_decay_values: [0.1, 0.01, 0.001]
- learning_rate_values: [0.01, 0.05, 0.1]
- best_weight_decay: 0.001
- best_learning_rate: 0.1
- num_epochs: 10
- batch_size: 32
- seed: 42

## Results
- Best epoch: 7
- Train loss: 0.1928
- Test loss: 0.1817
- Train AUC: 0.9241
- Test AUC: 0.9325

