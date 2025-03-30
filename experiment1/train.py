import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.data import get_data_loaders
from lib.models import Experiment1Model
from lib.train import train_model, set_seed
from lib.utils import get_device, save_experiment_summary

EXPERIMENT_NAME = "Experiment 1 - Simple Model"
HIDDEN_SIZE = 32
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

    model = Experiment1Model(input_dim=input_dim)
    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=NUM_EPOCHS,
        device=device,
        experiment_name=EXPERIMENT_NAME,
        plots_dir=PLOTS_DIR
    )

    best_epoch = torch.tensor(history['test_auc']).argmax().item()
    best_metrics = {
        'best_epoch': best_epoch + 1,
        'best_train_loss': history['train_loss'][best_epoch],
        'best_test_loss': history['test_loss'][best_epoch],
        'best_train_auc': history['train_auc'][best_epoch],
        'best_test_auc': history['test_auc'][best_epoch]
    }

    params = {
        'hidden_size': HIDDEN_SIZE,
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
