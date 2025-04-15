import sys
import os
import torch
import torch.nn as nn
import itertools
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.data import get_data_loaders
from lib.models import Experiment5Model
from lib.train import train_model, set_seed
from lib.utils import get_device, save_experiment_summary

EXPERIMENT_NAME = "Experiment 5 - Weight Decay and Learning Rate"
HIDDEN_SIZE = 128
NUM_BLOCKS = 3
USE_SKIP_CONNECTION = True
USE_BATCH_NORM = True
DROPOUT_P = 0.2
WEIGHT_DECAY_VALUES = [0.1, 0.01, 0.001]
LEARNING_RATE_VALUES = [0.01, 0.05, 0.1]
NUM_EPOCHS = 10
BATCH_SIZE = 32
SEED = 42

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(CURRENT_DIR, "..", "data", "loan_train.csv")
TEST_PATH = os.path.join(CURRENT_DIR, "..", "data", "loan_test.csv")
PLOTS_DIR = os.path.join(CURRENT_DIR, "plots")

if __name__ == "__main__":
    set_seed(SEED)

    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader, input_dim = get_data_loaders(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        batch_size=BATCH_SIZE
    )

    results = {}
    best_combination = None
    best_auc = 0.0

    for weight_decay, lr in itertools.product(WEIGHT_DECAY_VALUES, LEARNING_RATE_VALUES):
        print(f"\n--- Training with weight_decay = {weight_decay}, lr = {lr} ---\n")

        model = Experiment5Model(input_dim=input_dim, dropout_p=DROPOUT_P)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        criterion = nn.BCELoss()

        experiment_name = f"{EXPERIMENT_NAME} (wd={weight_decay}, lr={lr})"
        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=NUM_EPOCHS,
            device=device,
            experiment_name=experiment_name,
            plots_dir=PLOTS_DIR
        )

        result_key = f"wd={weight_decay}, lr={lr}"
        results[result_key] = history

        best_epoch = torch.tensor(history['test_auc']).argmax().item()
        current_best_auc = history['test_auc'][best_epoch]

        if current_best_auc > best_auc:
            best_auc = current_best_auc
            best_combination = (weight_decay, lr)

    plt.figure(figsize=(10, 8))

    heatmap_data = np.zeros((len(WEIGHT_DECAY_VALUES), len(LEARNING_RATE_VALUES)))

    for i, wd in enumerate(WEIGHT_DECAY_VALUES):
        for j, lr in enumerate(LEARNING_RATE_VALUES):
            result_key = f"wd={wd}, lr={lr}"
            history = results[result_key]
            best_epoch = torch.tensor(history['test_auc']).argmax().item()
            heatmap_data[i, j] = history['test_auc'][best_epoch]

    plt.imshow(heatmap_data, cmap='viridis', interpolation='nearest')

    plt.colorbar(label='Best Test AUC')

    plt.xticks(np.arange(len(LEARNING_RATE_VALUES)), LEARNING_RATE_VALUES)
    plt.yticks(np.arange(len(WEIGHT_DECAY_VALUES)), WEIGHT_DECAY_VALUES)
    plt.xlabel('Learning Rate')
    plt.ylabel('Weight Decay')

    for i in range(len(WEIGHT_DECAY_VALUES)):
        for j in range(len(LEARNING_RATE_VALUES)):
            plt.text(j, i, f"{heatmap_data[i, j]:.3f}",
                     ha="center", va="center", color="w")

    plt.title('AUC for Different Weight Decay and Learning Rate Combinations')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "weight_decay_lr_heatmap.png"))
    plt.close()

    best_wd, best_lr = best_combination
    best_result_key = f"wd={best_wd}, lr={best_lr}"
    best_history = results[best_result_key]
    best_epoch = torch.tensor(best_history['test_auc']).argmax().item()
    best_metrics = {
        'best_epoch': best_epoch + 1,
        'best_train_loss': best_history['train_loss'][best_epoch],
        'best_test_loss': best_history['test_loss'][best_epoch],
        'best_train_auc': best_history['train_auc'][best_epoch],
        'best_test_auc': best_history['test_auc'][best_epoch]
    }

    params = {
        'hidden_size': HIDDEN_SIZE,
        'num_blocks': NUM_BLOCKS,
        'use_skip_connection': USE_SKIP_CONNECTION,
        'use_batch_norm': USE_BATCH_NORM,
        'dropout_p': DROPOUT_P,
        'weight_decay_values': WEIGHT_DECAY_VALUES,
        'learning_rate_values': LEARNING_RATE_VALUES,
        'best_weight_decay': best_wd,
        'best_learning_rate': best_lr,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'seed': SEED
    }

    save_experiment_summary(
        experiment_name=EXPERIMENT_NAME,
        params=params,
        metrics=best_metrics,
        save_path=os.path.join(CURRENT_DIR, "README.md")
    )
