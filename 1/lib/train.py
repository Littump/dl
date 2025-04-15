import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def train_model(model, train_loader, test_loader, optimizer, criterion,
                num_epochs=10, device='cpu', experiment_name='experiment',
                plots_dir='plots'):
    model.to(device)

    os.makedirs(plots_dir, exist_ok=True)

    history = {
        'train_loss': [],
        'test_loss': [],
        'train_auc': [],
        'test_auc': []
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Train)'):

            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                features, targets = batch
            else:
                print(f"Warning: Batch in unexpected format: {type(batch)}")
                continue

            features, targets = features.to(device), targets.to(device)

            outputs = model(features)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            try:
                batch_preds = outputs.detach().cpu().numpy()
                batch_targets = targets.detach().cpu().numpy()

                train_preds.extend(batch_preds)
                train_targets.extend(batch_targets)
            except Exception as e:
                print(f"Error processing batch for AUC: {e}")

            train_loss += loss.item() * features.size(0)

        if len(train_targets) > 0:
            train_loss /= len(train_targets)
            try:
                train_targets_arr = np.array(train_targets)
                train_preds_arr = np.array(train_preds)

                if train_targets_arr.ndim > 1:
                    train_targets_arr = train_targets_arr.ravel()
                if train_preds_arr.ndim > 1:
                    train_preds_arr = train_preds_arr.ravel()

                if len(np.unique(train_targets_arr)) < 2:
                    print("Warning: AUC requires at least two classes, but only one present in training data")
                    train_auc = float('nan')
                else:
                    train_auc = roc_auc_score(train_targets_arr, train_preds_arr)
            except Exception as e:
                print(f"Error calculating training AUC: {e}")
                train_auc = float('nan')
        else:
            train_loss = float('nan')
            train_auc = float('nan')

        model.eval()
        test_loss = 0
        test_preds = []
        test_targets = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Test)'):
                if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                    features, targets = batch
                else:
                    print(f"Warning: Batch in unexpected format: {type(batch)}")
                    continue

                features, targets = features.to(device), targets.to(device)

                outputs = model(features)
                loss = criterion(outputs, targets)

                test_preds.extend(outputs.detach().cpu().numpy())
                test_targets.extend(targets.detach().cpu().numpy())
                test_loss += loss.item() * features.size(0)

        if len(test_targets) > 0:
            test_loss /= len(test_targets)
            try:

                test_targets_arr = np.array(test_targets)
                test_preds_arr = np.array(test_preds)

                if test_targets_arr.ndim > 1:
                    test_targets_arr = test_targets_arr.ravel()
                if test_preds_arr.ndim > 1:
                    test_preds_arr = test_preds_arr.ravel()

                if len(np.unique(test_targets_arr)) < 2:
                    print("Warning: AUC requires at least two classes, but only one present in test data")
                    test_auc = float('nan')
                else:
                    test_auc = roc_auc_score(test_targets_arr, test_preds_arr)
            except Exception as e:
                print(f"Error calculating test AUC: {e}")
                test_auc = float('nan')
        else:
            test_loss = float('nan')
            test_auc = float('nan')

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_auc'].append(train_auc)
        history['test_auc'].append(test_auc)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}')
        print(f'  Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}')

    plot_training_history(history, experiment_name, plots_dir)

    return history


def plot_training_history(history, experiment_name, plots_dir):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['test_loss'], label='Test')
    plt.title(f'{experiment_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_auc'], label='Train')
    plt.plot(history['test_auc'], label='Test')
    plt.title(f'{experiment_name} - ROC AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{experiment_name}_history.png'))
    plt.close()

    valid_indices = [i for i, auc in enumerate(history['test_auc']) if not np.isnan(auc)]
    if valid_indices:
        best_epoch = max(valid_indices, key=lambda i: history['test_auc'][i])
    else:
        best_epoch = 0

    best_metrics = {
        'best_epoch': best_epoch + 1,
        'best_train_loss': history['train_loss'][best_epoch],
        'best_test_loss': history['test_loss'][best_epoch],
        'best_train_auc': history['train_auc'][best_epoch],
        'best_test_auc': history['test_auc'][best_epoch]
    }

    plt.figure(figsize=(10, 2))
    plt.axis('off')
    plt.table(
        cellText=[[best_metrics['best_epoch'],
                   f"{best_metrics['best_train_loss']:.4f}",
                   f"{best_metrics['best_test_loss']:.4f}",
                   f"{best_metrics['best_train_auc']:.4f}",
                   f"{best_metrics['best_test_auc']:.4f}"]],
        colLabels=['Best Epoch', 'Train Loss', 'Test Loss', 'Train AUC', 'Test AUC'],
        loc='center'
    )
    plt.title(f'{experiment_name} - Best Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{experiment_name}_best_metrics.png'))
    plt.close()

    return best_metrics


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
