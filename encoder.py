import torch
import torchmetrics
from torch import nn
from torch.optim import Adam
import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau


class WordnPositionalSelfAttentionEmbeddings(L.LightningModule):
    """
    A PyTorch Lightning module for word and positional self-attention embeddings.
    This model uses LSTMs and linear layers to process sequences and output predictions.

    Parameters
    ----------
    vocab_size : int, optional
        The size of the vocabulary (default is 2).
    network_width : int, optional
        The width of the network (default is 2).
    output_neurons : int, optional
        The number of output neurons for the final layer (default is 4).
    context_length : int, optional
        The length of the input context (default is 1000).
    num_blocks : int, optional
        The number of neural network blocks (default is 2).
    shrinkage : int, optional
        The shrinkage factor for reducing the dimensions of neural network layers (default is 2).
    embedding_size : int, optional
        The size of the embedding vectors (default is 512).
    hidden_size : int, optional
        The hidden size of the LSTM (default is 64).
    """

    def __init__(
        self,
        vocab_size: int = 2,
        network_width: int = 2,
        output_neurons: int = 4,
        context_length: int = 1000,
        num_blocks: int = 2,
        shrinkage=2,
        embedding_size: int = 512,
        hidden_size: int = 64,
    ):
        super().__init__()
        self.context_length = context_length
        self.embedding_size = embedding_size

        self.vocab_size = vocab_size
        self.network_width = network_width
        self.semantic = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_blocks,
            batch_first=True,
            bidirectional=True,
        )
        lstm_output_dim = hidden_size * 2
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.l1 = nn.Sequential(
            nn.Linear(lstm_output_dim, self.network_width),
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
            nn.Linear(
                self.network_width // (shrinkage * (num_blocks + 1)),
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
        """
        Constructs a sequence of neural network blocks with shrinkage.

        Parameters
        ----------
        num_blocks : int
            The number of blocks to create.
        shrinkage : int
            The shrinkage factor for dimension reduction.

        Returns
        -------
        List[nn.Module]
            A list of NNBlocks with progressively reduced dimensions.
        """
        return_blocks = []
        for i in range(1, num_blocks + 1):
            if self.network_width // (i + 1) * shrinkage > self.output_neurons:
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
        """
        Constructs a sequence of neural network blocks with fixed dimensions.

        Parameters
        ----------
        num_blocks : int
            The number of blocks to create.
        neurons : int
            The number of neurons in each block.

        Returns
        -------
        List[nn.Module]
            A list of NNBlocks with fixed dimensions.
        """
        return_blocks = []
        for i in range(1, num_blocks + 1):
            return_blocks.append(NNBlocks(neurons, neurons))
        return return_blocks

    def forward(self, input_tensor):
        """
        Forward pass for the model.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor of shape (batch_size, context_length).

        Returns
        -------
        torch.Tensor
            The output tensor of shape (batch_size, output_neurons).
        """
        op = self.semantic(input_tensor)
        lstm_out, (hidden, cell) = self.lstm(op)
        hidden_out = torch.cat((hidden[-2], hidden[-1]), dim=1)
        op = self.l1(hidden_out)
        op = self.nn_blocks(op)
        op = self.nn_blocks_shrinkage(op)
        op = self.output_neurons(op)
        return op

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns
        -------
        dict
            A dictionary containing the optimizer and scheduler configurations.
        """
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
        """
        A single training step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            A batch of input and target tensors.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        torch.Tensor
            The computed loss.
        """
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
        """
        A single validation step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            A batch of input and target tensors.
        batch_idx : int
            The index of the batch.
        """
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
        """
        Actions to perform at the end of a training epoch.
        """
        self.log("train_acc_epoch", self.train_acc, on_step=False, on_epoch=True)
        self.log("train_prec_epoch", self.precision, on_step=False, on_epoch=True)
        self.log("train_rec_epoch", self.recall, on_step=False, on_epoch=True)

        self.train_acc.reset()
        self.precision.reset()
        self.recall.reset()

    def on_validation_epoch_end(self):
        """
        Actions to perform at the end of a validation epoch.
        """
        self.log("valid_acc_epoch", self.valid_acc, on_step=False, on_epoch=True)
        self.log("valid_prec_epoch", self.precision_valid, on_step=False, on_epoch=True)
        self.log("valid_rec_epoch", self.recall_valid, on_step=False, on_epoch=True)
        self.valid_acc.reset()
        self.precision_valid.reset()
        self.recall_valid.reset()


class NNBlocks(nn.Module):
    """
    A neural network block consisting of a linear layer, dropout, and ReLU activation.

    Parameters
    ----------
    input_dims : int
        The number of input dimensions.
    output_dims : int
        The number of output dimensions.
    """

    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(input_dims, output_dims),
            nn.Dropout(p=0.1),
            nn.ReLU(),
        )

    def forward(self, input_tensor):
        """
        Forward pass for the block.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        op = self.l1(input_tensor)
        return op
