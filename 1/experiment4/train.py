import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.data import get_data_loaders
from lib.models import Experiment4Model
from lib.train import train_model, set_seed
from lib.utils import get_device, save_experiment_summary, compare_experiments

EXPERIMENT_NAME = "Experiment 4 - Dropout"
HIDDEN_SIZE = 128
NUM_BLOCKS = 3
USE_SKIP_CONNECTION = True
USE_BATCH_NORM = True
DROPOUT_VALUES = [0.01, 0.1, 0.2, 0.5, 0.9]
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.01
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
    
    dropout_results = {}
    best_dropout = None
    best_auc = 0.0
    
    for dropout_p in DROPOUT_VALUES:
        print(f"\n--- Training with dropout_p = {dropout_p} ---\n")
        
        model = Experiment4Model(input_dim=input_dim, dropout_p=dropout_p)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCELoss()
        
        experiment_name = f"{EXPERIMENT_NAME} (p={dropout_p})"
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
        
        dropout_results[f"p={dropout_p}"] = history
        
        best_epoch = torch.tensor(history['test_auc']).argmax().item()
        current_best_auc = history['test_auc'][best_epoch]
        
        if current_best_auc > best_auc:
            best_auc = current_best_auc
            best_dropout = dropout_p
    
    compare_experiments(
        dropout_results,
        metric='test_auc',
        save_path=os.path.join(PLOTS_DIR, "dropout_comparison_auc.png")
    )
    
    compare_experiments(
        dropout_results,
        metric='test_loss',
        save_path=os.path.join(PLOTS_DIR, "dropout_comparison_loss.png")
    )
    
    best_history = dropout_results[f"p={best_dropout}"]
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
        'dropout_values_tested': DROPOUT_VALUES,
        'best_dropout': best_dropout,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'seed': SEED
    }

    save_experiment_summary(
        experiment_name=EXPERIMENT_NAME,
        params=params,
        metrics=best_metrics,
        save_path=os.path.join(CURRENT_DIR, "README.md")
    ) 