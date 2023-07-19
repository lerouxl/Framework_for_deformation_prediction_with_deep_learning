from torch_geometric.data import Dataset, Data
from pathlib import Path
import torch
import pyvista as pv
import configparser
import numpy as np
from torch_geometric.transforms import FaceToEdge
from utils.visualise import visualise_deformation_from_graph


class DeformationDataset(Dataset):
    def __init__(self, root, configuration=None, transform=None, pre_transform=None, pre_filter=None):
        """
        Dataset of deformed mesh with homogeneous graph. Use vtk files as raw data.
        The label is expected to be in the deformation_label_name vector.
        Features are expected to be in the features_name vector.
        """
        if configuration is None:
            self.config = configparser.ConfigParser()
            self.config.read(r"configuration.ini")
        elif isinstance(configuration, configparser.ConfigParser):
            self.config = configuration
        else:
            raise TypeError("Unsupported configuration type: {} instead of ConfigParser".format(type(configuration)))

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """ List all vtk files available in the root directory.
        """
        return [str(p.name) for p in Path(self.root / "raw").glob("*.vtk")]

    @property
    def processed_file_names(self):
        """ List all pytorch files available in the root directory.
        A list of files in the processed_dir which needs to be found in order to skip the processing.
        Those files are created by the process functions and are graph that can be used for training and evaluation.
        """
        return [str(p.with_suffix(".pt").name) for p in Path(self.root / "raw").glob("*.vtk")]

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # Where the processed file will be saved
            file_save_path = Path(self.processed_dir) / f'{idx}.pt'

            # Read data from `raw_path`
            mesh = pv.read(raw_path)

            # Get features
            x = mesh[self.config["DATA"]["features_name"]]
            x = torch.tensor(x)

            # Get the label
            y = mesh[self.config["DATA"]["deformation_label_name"]]
            y = torch.tensor(y)

            # Get the position
            pos = mesh.points
            pos = torch.tensor(np.array(pos))

            # Extract the edges from vtk
            if mesh.is_all_triangles:
                faces = mesh.faces  # Each face are organised as n0, p0_0, p0_1, ..., p0_n, with n_0 the number of point
                # Faces should be reshaped into the pytorch geometric format, with the number of point
                # we are expecting to have triangular faces
                faces = faces.reshape(-1, 4)[:, 1:]
                # Reshape into [[n0, p0_0, p0_1, p0_2],...] and remove n_0 to have [n,3 shape]
                faces = torch.tensor(faces.T)  # [3,n] shape compatible with pytorch geometric
            else:
                raise AssertionError(f"Mesh {raw_path.stem} have non triangles faces. Please provide triangular mesh. "
                                     f"You can try to correct with mesh.triangulate() to re-mesh the pyvista mesh.")

            # Create an homogeneous graph
            data = Data(x=x,  # Vertex features
                        y=y,  # Vertex label
                        face=faces,
                        pos=pos,  # vertex position
                        edge_attr=None,  # Edges features
                        file_name=Path(raw_path).stem,
                        file_save_path=file_save_path.parent
                        )
            # generate edges from the faces data
            transform = FaceToEdge(remove_faces=False)
            data = transform(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, file_save_path)
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(Path(self.processed_dir) / f'{idx}.pt')
        return data


if __name__ == "__main__":
    # Load configuration
    config = configparser.ConfigParser()
    config.read(r"configuration.ini")

    dataset_path = Path(config["DATA"]["dataset_path"]) / config["DATA"]["dataset_name"] / "train"
    dataset = DeformationDataset(dataset_path)
    data = dataset[0]

    # Display graph for confirmation
    visualise_deformation_from_graph(data)
