import models
import pytorch_lightning as L
from torch.nn import MSELoss
import torch
from pathlib import Path
class Model(L.LightningModule):
    def __init__(self, model_name, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        self.lr = float(model_kwargs['lr'])

        # Load model using its name
        selected_model = getattr(models, model_name)
        self.model = selected_model(**model_kwargs)

        # Loss function
        self.loss_module = MSELoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def forward(self, data):
        y_hat = self.model(data)
        return y_hat

    def training_step(self, batch, batch_idx):
        y_hat = self.forward(batch,)
        loss = self.loss_module(y_hat, batch.y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        loss = self.loss_module(y_hat, batch.y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        loss = self.loss_module(y_hat, batch.y)
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx):

        for i in range(batch.num_graphs):
            data= batch[i]
            y_hat = self.model(data)
            loss = self.loss_module(y_hat, data.y)
            save_path = Path(data.file_save_path)
            data.y_hat = y_hat

            if hasattr(data, "file_name"):
                save_path = Path(save_path / f"predict_{data.file_name}").with_suffix(".pt")
            else:
                save_path = Path(save_path / f"predict_{i}").with_suffix(".pt")
            torch.save(data, save_path)

