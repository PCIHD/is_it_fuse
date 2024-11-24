import torch
import lightning as L
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger

from dataset import Dataset
from encoder import WordnPositionalSelfAttentionEmbeddings
from utils import get_vocabulary

if __name__ == "__main__":

    df = pd.read_parquet("./data_foundation/curated/curated_dataset.parquet")
    df = df[df["text"].str.len() > 0]
    print(df.shape)
    df["text"] = df["text"].str.replace(r"[^A-Za-z\s]", "", regex=True)
    print(df[df["dataset_type"] == "train_data"].groupby("class_name").count())
    print(df[df["dataset_type"] == "test_data"].groupby("class_name").count())
    context_length = 600
    vocabulary = get_vocabulary(df, context_length)
    dataset = Dataset(
        df[df["dataset_type"] == "train_data"], vocabulary, context_length
    )
    val_dataset = Dataset(
        df[df["dataset_type"] == "test_data"], vocabulary, context_length
    )
    print(len(dataset))
    print(len(val_dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=4,
    )
    model = WordnPositionalSelfAttentionEmbeddings(
        vocab_size=len(dataset.vocabulary),
        network_width=8,
        num_blocks=1,
        context_length=context_length,
    )
    mlf_logger = MLFlowLogger(
        experiment_name="torch_classifier",
        tracking_uri="http://127.0.0.1:5000",
        log_model=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = L.Trainer(
        max_epochs=50,
        logger=mlf_logger,
        enable_progress_bar=True,
        log_every_n_steps=50,
        accelerator="mps",
        callbacks=[lr_monitor],
        accumulate_grad_batches=2,
    )
    trainer.fit(
        model=model, train_dataloaders=dataloader, val_dataloaders=val_dataloader
    )
