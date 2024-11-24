import torch
import torchmetrics
from torch import nn
from torch.distributions import Uniform
from torch.optim import Adam
import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import ConfusionMatrix


class WordnPositionalSelfAttentionEmbeddings(L.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        network_width: int = 2,
        output_neurons: int = 4,
        context_length: int = 1000,
        num_blocks: int = 2,
        shrinkage=2,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.network_width = network_width
        self.l1 = nn.Sequential(
            nn.Linear(self.vocab_size, self.network_width),
            nn.ReLU(),
            nn.Linear(self.network_width, self.network_width // shrinkage),
            nn.ReLU(),
        )
        self.output_neurons = output_neurons
        self.nn_blocks = nn.Sequential(
            *self.get_deep_blocks(num_blocks, self.network_width // shrinkage)
        )
        self.nn_blocks_shrinkage = nn.Sequential(
            *self.get_deep_blocks_shrinkage(num_blocks, shrinkage)
        )
        self.output_neurons = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(
                self.network_width // (shrinkage * (num_blocks + 1)) * context_length,
                output_neurons,
            ),
            nn.Softmax(dim=-1),
        )
        self.loss = nn.CrossEntropyLoss()
        self.val_loss = nn.CrossEntropyLoss()
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

    def get_deep_blocks_shrinkage(self, num_blocks, shrinkage):
        return_blocks = []
        for i in range(1, num_blocks + 1):
            if self.network_width // (i + 1) * shrinkage > self.output_neurons:
                print(
                    self.network_width // ((i) * shrinkage),
                    self.network_width // ((i + 1) * shrinkage),
                )
                return_blocks.append(
                    NNBlocks(
                        self.network_width // ((i) * shrinkage),
                        self.network_width // ((i + 1) * shrinkage),
                    )
                )
            else:
                return return_blocks
        return return_blocks

    def get_deep_blocks(self, num_blocks, neurons):
        return_blocks = []
        for i in range(1, num_blocks + 1):
            return_blocks.append(NNBlocks(neurons, neurons))
        return return_blocks

    def forward(self, input_tensor):
        op = self.l1(input_tensor)
        op = self.nn_blocks(op)
        op = self.nn_blocks_shrinkage(op)
        op = self.output_neurons(op)
        return op

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.1)
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, "min", patience=2),
            "name": "lor_scheduler",
            "interval": "epoch",
            "monitor": "train_loss",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

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
        self.log("train_loss", loss, on_step=True, on_epoch=False)

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
        val_loss = self.val_loss(output, label_i)
        self.log("val_loss", val_loss, on_step=True, on_epoch=False)

    def on_train_epoch_end(self):
        self.log("train_acc_epoch", self.train_acc, on_step=False, on_epoch=True)
        self.log("train_prec_epoch", self.precision, on_step=False, on_epoch=True)
        self.log("train_rec_epoch", self.recall, on_step=False, on_epoch=True)

        self.train_acc.reset()
        self.precision.reset()
        self.recall.reset()

    def on_validation_epoch_end(self):

        self.log("valid_acc_epoch", self.valid_acc, on_step=False, on_epoch=True)
        self.log("valid_prec_epoch", self.precision_valid, on_step=False, on_epoch=True)
        self.log("valid_rec_epoch", self.recall_valid, on_step=False, on_epoch=True)
        self.valid_acc.reset()
        self.precision_valid.reset()
        self.recall_valid.reset()


class NNBlocks(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(input_dims, output_dims),
            nn.Dropout(p=0.1),
            nn.ReLU(),
        )

    def forward(self, input_tensor):
        op = self.l1(input_tensor)
        return op
