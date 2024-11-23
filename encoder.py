import torch
from torch import nn
from torch.distributions import Uniform
from torch.optim import Adam
import pytorch_lightning as L


class WordnPositionalSelfAttentionEmbeddings(L.LightningModule):
    def __init__(
        self, vocab_size: int, network_width: int = 2, output_neurons: int = 4
    ):
        super().__init__()
        min_thresh = -0.5
        max_thresh = 0.5

        # add positional encoding methods as a list
        self.positional_encoders = get_positional_encoders(network_width)
        # set weights with appropriate width and length
        self.vocab_size = vocab_size
        self.network_width = network_width
        self.fc1 = nn.Linear(self.vocab_size, self.network_width)
        self.fc2 = nn.Linear(self.network_width, self.network_width // 2)
        self.output_neurons = nn.Linear(self.network_width // 2, output_neurons)
        self.loss = nn.CrossEntropyLoss()

        pass

    def forward(self, input_tensor):
        inputs = []
        input_tensor = input_tensor[0]
        op = self.fc1(input_tensor)
        op = self.fc2(op)
        op = self.output_neurons(op)

        return torch.softmax(op)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = self.loss(output_i, label_i[0])
        return loss


def get_positional_encoders(network_width):
    encoders = [torch.sin, torch.cos]
    # based on the network width provide alternating sin and cos functions as encoders. Frequency management is done in forward method
    return [
        encoders[network_with_element % 2]
        for network_with_element in range(network_width)
    ]
