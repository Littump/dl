import torch.nn as nn


class SimpleBlock(nn.Module):
    def __init__(self, hidden_size, dropout_p=0.0, use_batch_norm=False):
        super(SimpleBlock, self).__init__()
        self.use_batch_norm = use_batch_norm

        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_size)

        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)

        if dropout_p > 0:
            self.dropout = nn.Dropout(dropout_p)
        else:
            self.dropout = None

    def forward(self, x):
        if self.use_batch_norm:
            x = self.batch_norm(x)

        x = self.fc1(x)
        x = self.relu(x)

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.fc2(x)

        return x


class SimpleBlockWithSkip(SimpleBlock):
    def forward(self, x):
        identity = x

        if self.use_batch_norm:
            x = self.batch_norm(x)

        x = self.fc1(x)
        x = self.relu(x)

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.fc2(x)

        x = x + identity

        return x


class BaseModel(nn.Module):
    def __init__(self, input_dim, hidden_size=32, num_blocks=1,
                 use_skip_connection=False, use_batch_norm=False, dropout_p=0.0):
        super(BaseModel, self).__init__()

        self.input_layer = nn.Linear(input_dim, hidden_size)

        block_class = SimpleBlockWithSkip if use_skip_connection else SimpleBlock

        self.blocks = nn.ModuleList([
            block_class(hidden_size, dropout_p, use_batch_norm) for _ in range(num_blocks)
        ])

        self.output_layer = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)

        for block in self.blocks:
            x = block(x)

        x = self.output_layer(x)
        x = self.sigmoid(x)

        return x


class Experiment1Model(BaseModel):

    def __init__(self, input_dim):
        super(Experiment1Model, self).__init__(
            input_dim=input_dim,
            hidden_size=32,
            num_blocks=1,
            use_skip_connection=False,
            use_batch_norm=False,
            dropout_p=0.0
        )


class Experiment2Model(BaseModel):

    def __init__(self, input_dim):
        super(Experiment2Model, self).__init__(
            input_dim=input_dim,
            hidden_size=128,
            num_blocks=3,
            use_skip_connection=False,
            use_batch_norm=False,
            dropout_p=0.0
        )


class Experiment3Model(BaseModel):

    def __init__(self, input_dim):
        super(Experiment3Model, self).__init__(
            input_dim=input_dim,
            hidden_size=128,
            num_blocks=3,
            use_skip_connection=True,
            use_batch_norm=True,
            dropout_p=0.0
        )


class Experiment4Model(BaseModel):

    def __init__(self, input_dim, dropout_p=0.1):
        super(Experiment4Model, self).__init__(
            input_dim=input_dim,
            hidden_size=128,
            num_blocks=3,
            use_skip_connection=True,
            use_batch_norm=True,
            dropout_p=dropout_p
        )


class Experiment5Model(Experiment4Model):

    def __init__(self, input_dim, dropout_p=0.1):
        super(Experiment5Model, self).__init__(
            input_dim=input_dim,
            dropout_p=dropout_p
        )
