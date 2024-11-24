import torch
import torchmetrics
from torch import nn
from torch.distributions import Uniform
from torch.optim import Adam
import lightning as L
from torchmetrics import ConfusionMatrix


class WordnPositionalSelfAttentionEmbeddings(L.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        network_width: int = 2,
        output_neurons: int = 4,
        context_length: int = 1000,
    ):
        super().__init__()
        min_thresh = -0.5
        max_thresh = 0.5

        # add positional encoding methods as a list
        # set weights with appropriate width and length
        self.vocab_size = vocab_size
        self.network_width = network_width
        self.l1 = nn.Sequential(
            nn.Linear(self.vocab_size, self.network_width),
            nn.ReLU(),
            nn.Linear(self.network_width, self.network_width // 2),
            nn.ReLU(),
            nn.Linear(self.network_width // 2, self.network_width // 4),
            nn.ReLU(),
            nn.Linear(self.network_width // 4, self.network_width // 4),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(self.network_width // 4 * context_length, output_neurons),
            nn.Softmax(dim=-1),
        )
        # self.fc1 = nn.Linear(self.vocab_size, self.network_width)
        # self.fc2 = nn.Linear(self.network_width, self.network_width // 2)
        # self.relu = nn.ReLU()
        # self.output_neurons = nn.Linear(self.network_width // 2, output_neurons)
        self.loss = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=output_neurons
        )
        self.valid_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=output_neurons
        )
        self.precision = torchmetrics.Precision(
            num_classes=output_neurons, task="multiclass"
        )
        self.recall = torchmetrics.Recall(num_classes=output_neurons, task="multiclass")
        self.precision_valid = torchmetrics.Precision(
            num_classes=output_neurons, task="multiclass"
        )
        self.recall_valid = torchmetrics.Recall(
            num_classes=output_neurons, task="multiclass"
        )

    def forward(self, input_tensor):
        op = self.l1(input_tensor)
        return op

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = self.loss(output_i, label_i)
        self.train_acc(output_i, label_i)
        self.precision(output_i, label_i)
        self.recall(output_i, label_i)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=False)
        self.log("train_prec", self.precision, on_step=True, on_epoch=False)
        self.log("train_rec", self.recall, on_step=True, on_epoch=False)
        self.logger.log_metrics({"loss": loss}, step=batch_idx)

        return loss

    def validation_step(self, batch, batch_idx):
        input_i, label_i = batch
        output = self(input_i)
        self.valid_acc(output, label_i)
        self.precision_valid(output, label_i)
        self.recall_valid(output, label_i)
        self.log("valid_acc", self.valid_acc, on_step=True, on_epoch=False)
        self.log("valid_prec", self.precision_valid, on_step=True, on_epoch=False)
        self.log("valid_rec", self.recall_valid, on_step=True, on_epoch=False)

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        self.log("train_prec", self.precision, on_step=False, on_epoch=True)
        self.log("train_rec", self.recall, on_step=False, on_epoch=True)
        self.train_acc.reset()
        self.precision.reset()
        self.recall.reset()

    def on_validation_epoch_end(self):

        self.log("valid_acc_epoch", self.valid_acc, on_step=False, on_epoch=True)
        self.log("valid_prec", self.precision_valid, on_step=False, on_epoch=True)
        self.log("valid_rec", self.recall_valid, on_step=False, on_epoch=True)
        self.valid_acc.reset()
        self.precision_valid.reset()
        self.recall_valid.reset()
