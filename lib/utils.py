import torch
import matplotlib.pyplot as plt
import os


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_experiment_summary(experiment_name, params, metrics, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        f.write(f"# {experiment_name}\n\n")

        f.write("## Parameters\n")
        for key, value in params.items():
            f.write(f"- {key}: {value}\n")
        f.write("\n")

        f.write("## Results\n")
        f.write(f"- Best epoch: {metrics['best_epoch']}\n")
        f.write(f"- Train loss: {metrics['best_train_loss']:.4f}\n")
        f.write(f"- Test loss: {metrics['best_test_loss']:.4f}\n")
        f.write(f"- Train AUC: {metrics['best_train_auc']:.4f}\n")
        f.write(f"- Test AUC: {metrics['best_test_auc']:.4f}\n")
        f.write("\n")

    print(f"Experiment summary saved to {save_path}")


def compare_experiments(experiments_data, metric='test_auc', save_path=None):

    plt.figure(figsize=(10, 6))

    for experiment_name, history in experiments_data.items():
        plt.plot(history[metric], label=experiment_name)

    plt.title(f'Comparison of {metric}')
    plt.xlabel('Epoch')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.close()
