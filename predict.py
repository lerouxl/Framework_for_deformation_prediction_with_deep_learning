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
dataset_path = Path(r"data\cubes\test")
dataset = DeformationDataset(dataset_path, config)


# Create dataloader:
dataloader = DataLoader(dataset=dataset,
                              batch_size=int(config["AI"]["batch_size"]))

# Load the AI model:
select_model = config["AI"]["model_name"]
model = Model(select_model=select_model, **dict(config["AI"]))
model = model.load_from_checkpoint(r"logs\2023_7_19_11_29_28\lightning_logs\version_0\checkpoints\epoch=7908-step=7909.ckpt")

try:
    model = compile(model)
except:
    print("torch compile is not available, continue training without compiling the model.")


trainer = L.Trainer(accelerator="auto",
                    check_val_every_n_epoch=1,
                    devices="auto",
                    max_epochs=int(config["AI"]["max_epochs"]),
                    default_root_dir=f"logs/{run_name}"
                    )


# Make a prediction
model.eval()
model.to("cpu")
graph = dataset.get(0)
y_hat = model(graph)

visualise_deformation_from_graph(graph.detach(), y_hat.detach())

trainer.predict(model, dataloader)
# Convert prediction into vtk files
convert_folder_of_pt(dataset_path / "processed", "predict_")