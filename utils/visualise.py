from pathlib import Path, PurePath
import pyvista as pv
from torch_geometric.data import Data
import torch
import numpy as np
from .pt_to_vtk import graph_to_vtk

def visualise_deformation_from_file(mesh_path: Path, deformation_name: str, scale_deformation: int = 1) -> None:
    """ Display a mesh and its deformed mesh.

    :param mesh_path: Path to the vtk mesh with extension.
    :param deformation_name: Name of the deformation vector.
    :param scale_deformation: Deformation scaling magnitude. Defaults to 1
    :return: None
    """
    # Type check of the input variables
    if type(mesh_path) is str:
        mesh_path = Path(mesh_path)

    assert isinstance(mesh_path, PurePath), f"mesh_path should be a Path from pathlib not a {type(mesh_path)}"

    assert isinstance(deformation_name, str), f"total_sample should be an str not a {type(deformation_name)}"

    # Load undeformed mesh
    mesh = pv.read(mesh_path)

    # Create a plotter
    pl = pv.Plotter()

    # Add the undeformed mesh
    pl.add_mesh(mesh, color="white", opacity=0.5)

    # Add the deformed pump bracket with the mode shape
    warp = mesh.warp_by_vector(deformation_name, factor=scale_deformation)
    pl.add_mesh(warp, show_scalar_bar=True, ambient=0.2, opacity=0.5)
    pl.add_axes()
    pl.enable_anti_aliasing('fxaa')
    pl.show()


def visualise_deformation_from_pt_file(path: str):
    """Load a pt graph file and display it"""
    graph = torch.load(path)
    visualise_deformation_from_graph(graph=graph)


def visualise_deformation_from_graph(graph: Data, prediction: torch.Tensor = None, scale_deformation: int = 1):
    """Recreate pyvista object from pytorch graph.
    """
    mesh = graph_to_vtk(graph)
    """
    vertex = graph.pos.numpy().copy()
    faces = graph.face.numpy().copy().T
    faces = np.insert(faces, 0, 3, axis=1).flatten()
    mesh = pv.PolyData(var_inp=vertex, faces=faces)

    mesh["features"] = graph.x.numpy().copy()
    mesh["label"] = graph.y.numpy().copy()
    mesh.set_active_scalars("label")"""

    # Create a plotter
    plot = pv.Plotter()
    # Add the deformed pump bracket with the mode shape
    label = mesh.warp_by_vector("label", factor=scale_deformation)

    if prediction is not None:
        # Add the label (white) and predicted deformed mesh
        mesh["prediction"] = prediction
        prediction = mesh.warp_by_vector("prediction", factor=scale_deformation)
        prediction.set_active_scalars("prediction")
        plot.add_mesh(prediction, show_scalar_bar=True, ambient=0.2, opacity=0.5)
        plot.add_mesh(label, color="white", ambient=0.2, opacity=0.5)
    else:
        # Add the undeformed mesh and the label deformed mesh
        plot.add_mesh(mesh, color="white", opacity=0.5)
        plot.add_mesh(label, show_scalar_bar=True, ambient=0.2, opacity=0.5)

    plot.add_axes()
    plot.enable_anti_aliasing('fxaa')
    plot.show()

    return mesh


if __name__ == "__main__":
    print("Visualise a vtk file with the label deformation")
    visualise_deformation_from_file(Path(r"data/cubes/train/raw/0.vtk"), "deformation (mm)")

    print("Visualise a pt file")
    visualise_deformation_from_pt_file(Path(r"data/cubes/train/processed/0.pt"))

    print("Visualise a deformation and label")
    graph = torch.load(Path(r"data/cubes/train/processed/0.pt"))
    deformation = torch.zeros_like(graph.x)
    deformation[:, 0] = torch.cos(graph.x[:, -1])
    visualise_deformation_from_graph(graph, deformation, 1)
