import torch
import lightning as L
import pandas as pd
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger


from dataset import Dataset
from encoder import WordnPositionalSelfAttentionEmbeddings
from utils import get_vocabulary

if __name__ == "__main__":

    df = pd.read_parquet("./data_foundation/curated/curated_dataset.parquet")
    df = df[df["text"].str.len() > 0]
    context_length = 500
    vocabulary = get_vocabulary(df, context_length)
    dataset = Dataset(
        df[df["dataset_type"] == "train_data"], vocabulary, context_length
    )
    val_dataset = Dataset(
        df[df["dataset_type"] == "test_data"], vocabulary, context_length
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=4, persistent_workers=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=16, num_workers=4, persistent_workers=True
    )
    model = WordnPositionalSelfAttentionEmbeddings(
        vocab_size=len(dataset.vocabulary),
        network_width=16,
        context_length=context_length,
    )
    mlf_logger = MLFlowLogger(
        experiment_name="torch_classifier",
        tracking_uri="http://127.0.0.1:5000",
        log_model=True,
    )

    trainer = L.Trainer(
        max_epochs=20,
        logger=mlf_logger,
        enable_progress_bar=True,
        log_every_n_steps=2,
        accelerator="mps",
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
    )
    trainer.fit(
        model=model, train_dataloaders=dataloader, val_dataloaders=val_dataloader
    )
