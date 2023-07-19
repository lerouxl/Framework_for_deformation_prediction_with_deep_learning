import configparser
from pathlib import Path
from dataset import DeformationDataset
from trainer.model import Model
from torch_geometric.loader import DataLoader
import pytorch_lightning as L
from torch_geometric import compile
from utils.visualise import visualise_deformation_from_graph
from utils.pt_to_vtk import convert_folder_of_pt
import datetime

now = datetime.datetime.now()
run_name = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
L.seed_everything(51, workers=True)

# Read the configuration
config = configparser.ConfigParser()
config.read(r"configuration.ini")

# Create dataset
dataset_path = Path(Path(config["DATA"]["dataset_path"]) / config["DATA"]["dataset_name"])
train_dataset = DeformationDataset(dataset_path / "train", config)
test_dataset = DeformationDataset(dataset_path / "test", config)
validation_dataset = DeformationDataset(dataset_path / "validation", config)

# Create dataloader:
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=int(config["AI"]["batch_size"]))
test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=int(config["AI"]["batch_size"]))
validation_dataloader = DataLoader(dataset=validation_dataset,
                                   batch_size=int(config["AI"]["batch_size"]))

# Load the AI model:
select_model = config["AI"]["model_name"]
model = Model(select_model=select_model, **dict(config["AI"]))

try:
    model = compile(model)
except:
    print("torch compile is not available, continue training without compiling the model.")

callback = [L.callbacks.ModelCheckpoint(dirpath=f"logs/{run_name}",
                                        monitor="val_loss",
                                        save_last=True)]
trainer = L.Trainer(accelerator="auto",
                    check_val_every_n_epoch=1,
                    devices="auto",
                    max_epochs=int(config["AI"]["max_epochs"]),
                    default_root_dir=f"logs/{run_name}"
                    )
trainer.fit(model,
            train_dataloader,
            validation_dataloader)

trainer.test(dataloaders=test_dataloader)

# Make a prediction
model.eval()
model.to("cpu")
graph = test_dataset.get(0)
y_hat = model(graph)

visualise_deformation_from_graph(graph.detach(), y_hat.detach())

trainer.predict(model, test_dataloader)
# Convert prediction into vtk files
convert_folder_of_pt(dataset_path / "test" / "processed", "predict_")