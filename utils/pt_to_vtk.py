import torch
import pyvista as pv
from pathlib import Path
import numpy as np
from tqdm import tqdm


def pt_to_vtk(path: Path):
    """ Take a .pt file and convert it to a vtk file.
    """
    path = Path(path)
    graph = torch.load(path)
    mesh = graph_to_vtk(graph)
    mesh.save(path.with_suffix('.vtk'))


def graph_to_vtk(graph):
    """
    Convert a graph to a vtk object
    """
    vertex = graph.pos.detach().cpu().numpy().copy()
    faces = graph.face.detach().cpu().numpy().copy().T
    faces = np.insert(faces, 0, 3, axis=1).flatten()
    mesh = pv.PolyData(var_inp=vertex, faces=faces)

    mesh["features"] = graph.x.detach().cpu().numpy().copy()
    mesh["label"] = graph.y.detach().cpu().numpy().copy()
    mesh.set_active_scalars("label")

    if hasattr(graph, "y_hat"):
        mesh["prediction"] = graph.y_hat.detach().cpu().numpy().copy()
        mesh.set_active_scalars("prediction")

    return mesh


def convert_folder_of_pt(path: Path, filter: str = None):
    """
    Convert all the pt files of a folder into vtk files

    :param path: Path to the folder to convert
    :param filter: str that every transformed files name must contain
    """
    path = Path(path)

    if filter is None:
        filter = ".pt"
    else:
        filter = f"*{filter}*.pt"

    files = list(path.glob(filter))

    for file in tqdm(files, desc="Transforming pt files to vtk files"):
        pt_to_vtk(file)
