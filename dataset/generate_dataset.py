from pathlib import Path, PurePath
from pyvista import Cube
from random import uniform
from tqdm import tqdm, trange
import configparser
from numpy import savetxt, vstack, zeros_like
import pyvista as pv


def generate_dataset(directory: Path, total_sample: int) -> None:
    """Create a simple dataset from scratch.

    Create a simple dataset at the "directory" path given.
    Three folders are created:
        - train : representing 80% of the dataset
        - test : representing 15% of the dataset
        - validation : representing 5% of the dataset

    :param directory: Path of the dataset ( dataset_path / dataset_name)
    :param total_sample: Total number of sample to generate
    :return: None
    """

    # Type check of the input variables
    if type(directory) is str:
        directory = Path(directory)

    assert isinstance(directory, PurePath), f"directory should be a Path from pathlib not a {type(directory)}"

    assert isinstance(total_sample, int), f"total_sample should be an int not a {type(total_sample)}"

    # Set the number of samples for the train, test and validation
    validation_sample = int(0.15 * total_sample)
    test_sample = int(0.05 * total_sample)
    train_sample = total_sample - validation_sample - test_sample

    print(f"A dataset will be generated in {str(directory)} with:")
    print(f"\t - train sample : {train_sample}")
    print(f"\t - test sample : {test_sample}")
    print(f"\t - validation sample : {validation_sample}")

    # Create datasets folders
    (directory / "train" / "raw").mkdir(parents=True, exist_ok=True)
    (directory / "test" / "raw").mkdir(parents=True, exist_ok=True)
    (directory / "validation" / "raw").mkdir(parents=True, exist_ok=True)

    # Generate the train dataset
    for i in trange(train_sample, desc="Generating training data"):
        simple_cube_and_def(directory / "train" / "raw" / str(i))

    # Generate the test dataset
    for i in trange(test_sample, desc="Generating testing data"):
        simple_cube_and_def(directory / "test" / "raw" / str(i))

    # Generate the validation dataset
    for i in trange(validation_sample, desc="Generating validation data"):
        simple_cube_and_def(directory / "test" / "raw" / str(i))


def simple_cube_and_def(save_path: Path) -> None:
    """Generate a simple cube with a deformation file.

    Generate a cube of random dimension and the deformation of each vertex.
    Cube bottom surface will be on the XY plan (Z=0) and its center is on the Z axis.
    Cube side is generated from an uniform distribution in the range [CUBE][min_side] to [CUBE][max_side] rounded to 3.
    The cube is saved as save_path.vtk that can be opened with Paraview.
    Vertex deformation label are saved in [CUBE][deformation_label_name] from configuration.ini.

    :param save_path: Path where to save files, should not have extension ( e.g. dataset_path/dataset_name/train/1 )
    :return: None
    """

    # Type check of the input variables
    if type(save_path) is str:
        save_path = Path(save_path)

    assert isinstance(save_path, PurePath), f"save_path should be a Path from pathlib not a {type(save_path)}"

    # Load configuration
    config = configparser.ConfigParser()
    config.read(r"configuration.ini")

    # Generate a cube
    cube_dimension = uniform(float(config["CUBE"]["min_side"]),
                             float(config["CUBE"]["max_side"]))
    cube_dimension = round(cube_dimension, 3)

    cube_center_z = cube_dimension / 2

    mesh = Cube(center=(0.0, 0.0, cube_center_z),
                x_length=cube_dimension,
                y_length=cube_dimension,
                z_length=cube_dimension,
                clean=True).triangulate()

    # Remesh it to increase the number of vertex
    mesh = mesh.subdivide(nsub=5)

    # Generate deformation
    # This deformation is a schrikage of the part dependant of the z axis
    vertex_coor = mesh.points
    z = vertex_coor[:, 2]
    xy = vertex_coor[:, :2].T
    deformation_xy = xy * z * float(config["CUBE"]["deformation_scale"])

    # Deformation vector
    deformation = - vstack((deformation_xy, zeros_like(z))).T
    mesh[config["DATA"]["deformation_label_name"]] = deformation

    # Feature vector
    # The vertex coordinates are used as features
    mesh[config["DATA"]["features_name"]] = mesh.points

    # Save file and deformation
    obj_path = save_path.with_suffix(".vtk")
    mesh.save(obj_path)


if __name__ == "__main__":
    # Load configuration
    config = configparser.ConfigParser()
    config.read(r"configuration.ini")

    dataset_path = Path(config["DATA"]["dataset_path"]) / config["DATA"]["dataset_name"]
    dataset_sample = int(config["DATA"]["total_sample"])
    generate_dataset(dataset_path, dataset_sample)
