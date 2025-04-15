import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class LoanDataset(Dataset):
    def __init__(self, csv_file, transform=True, numerical_scaler=None, categorical_encoder=None):
        self.loan_data = pd.read_csv(csv_file)
        self.transform = transform

        id_col = 'id' if 'id' in self.loan_data.columns else None
        target_col = 'loan_status' if 'loan_status' in self.loan_data.columns else None

        cols_to_drop = [col for col in [id_col, target_col] if col is not None]
        self.X = self.loan_data.drop(columns=cols_to_drop, errors='ignore')

        if target_col:
            self.y = self.loan_data[target_col].values
        else:
            self.y = None

        self.cat_cols = self.X.select_dtypes(include=['object']).columns.tolist()
        self.num_cols = self.X.select_dtypes(include=['number']).columns.tolist()

        self.numerical_scaler = numerical_scaler or StandardScaler()
        self.categorical_encoder = categorical_encoder or OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        if self.transform:
            if numerical_scaler is None and categorical_encoder is None:
                self._fit_transformers()
            self._transform_features()

    def _fit_transformers(self):
        self.numerical_scaler.fit(self.X[self.num_cols])

        if self.cat_cols:
            self.categorical_encoder.fit(self.X[self.cat_cols])

    def _transform_features(self):
        if self.num_cols:
            num_features = self.numerical_scaler.transform(self.X[self.num_cols])
            self.num_features = torch.FloatTensor(num_features)
        else:
            self.num_features = torch.FloatTensor()

        if self.cat_cols:
            cat_features = self.categorical_encoder.transform(self.X[self.cat_cols])
            self.cat_features = torch.FloatTensor(cat_features)

            self.cat_dims = []
            for i, col in enumerate(self.cat_cols):
                unique_vals = self.X[col].nunique()
                self.cat_dims.append(unique_vals)
        else:
            self.cat_features = torch.FloatTensor()
            self.cat_dims = []

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.transform:
            if len(self.cat_cols) > 0 and len(self.num_cols) > 0:
                features = torch.cat([self.num_features[idx], self.cat_features[idx]], dim=0)
            elif len(self.cat_cols) > 0:
                features = self.cat_features[idx]
            else:
                features = self.num_features[idx]
        else:
            features = torch.FloatTensor(self.X.iloc[idx].values)

        if self.y is not None:
            label = torch.FloatTensor([self.y[idx]])
            return features, label
        else:
            return features


def get_data_loaders(train_path, test_path, batch_size=32, transform=True):
    print(f"Loading data from {train_path} and {test_path}")
    train_dataset = LoanDataset(train_path, transform=transform)

    test_dataset = LoanDataset(
        test_path,
        transform=transform,
        numerical_scaler=train_dataset.numerical_scaler,
        categorical_encoder=train_dataset.categorical_encoder
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    first_item = train_dataset[0]
    if isinstance(first_item, tuple):
        sample_features, _ = first_item
        print(f"Training dataset has labels. Features shape: {sample_features.shape}")
    else:
        sample_features = first_item
        print(f"Training dataset has no labels. Features shape: {sample_features.shape}")

    input_dim = sample_features.shape[0]

    return train_loader, test_loader, input_dim
