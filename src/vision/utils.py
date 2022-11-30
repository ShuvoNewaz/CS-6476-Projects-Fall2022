from typing import Sequence, Union, Callable

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader

from vision.part3_pointnet import PointNet
from vision.part4_analysis import get_confusion_matrix, get_critical_indices
from vision.part5_positional_encoding import PointNetPosEncoding


def verify(function: Callable) -> str:
    """Will indicate with a print statement whether assertions passed or failed
    within function argument call.
    Args:
        function: Python function object
    Returns:
        string that is colored red or green when printed, indicating success
    """
    try:
        function()
        return '\x1b[32m"Correct"\x1b[0m'
    except AssertionError:
        return '\x1b[31m"Wrong"\x1b[0m'


def plot_confusion_matrix(confusion_matrix: np.ndarray, class_labels: Sequence[str]) -> None:
    """
    Plots the confusion matrix
    
    Args:
    -   confusion_matrix: a (num_classes, num_classes) numpy array representing the confusion matrix
    -   class_labels: A list containing the class labels at the index of their label_number
                      e.g. if the labels are {"Cat": 0, "Monkey": 2, "Dog": 1}, the input value
                      should be ["Cat", "Dog", "Monkey"]
    """
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)

    num_classes = len(class_labels)

    ax.imshow(confusion_matrix, cmap="Blues")

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Ground-Truth label")
    ax.set_title("Confusion Matrix")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(num_classes):
        for j in range(num_classes):
            _ = ax.text(
                j,
                i,
                f"{confusion_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
            )

    plt.show()


def generate_and_plot_confusion_matrix(
    model: Union[PointNet, PointNetPosEncoding], 
    loader: DataLoader, 
    num_classes: int, 
    device: str='cpu'
) -> None:
    """
    Runs the entire confusion matrix pipeline for convenience
    
    Args:
    -   model: Model to generate confusion matrix data for
    -   dataset: The ImageLoader dataset that corresponds to training or validation data
    -   num_classes: The number of classes we have
    -   device: which device to run on
    """

    class_to_idx = loader.dataset.class_dict
    class_labels = sorted(class_to_idx, key=lambda item: class_to_idx[item])
    confusion_matrix = get_confusion_matrix(model, loader, num_classes, True, device)

    plot_confusion_matrix(confusion_matrix, class_labels)


def read_points_file(file_path: str) -> np.ndarray:
    """
    Reads points from the given file
    
    Args:
    -   file_path: The path to the file to read from
    
    Output:
    -   numpy array with shape (n, 3) where n is the number of points in the file
    """
    x = []
    y = []
    z = []
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines[1:]:
            coordinates = line.strip().split(' ')
            x.append([float(coordinates[0])])
            y.append([float(coordinates[1])])
            z.append([float(coordinates[2])])

    points = np.hstack((np.array(x), np.array(y), np.array(z)))
    return points


def plot_point_cloud(pts: np.ndarray, library='plotly') -> None:
    """
    Plots point cloud using the specified library

    Args:
    -   pts: numpy array of shape (n, 3) where n is the number of points to plot
    -   library: either plotly or matplotlib specifying which library to use to plot the points
    """
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    if library=='plotly':
        fig = px.scatter_3d(x=x, y=y, z=z)
        fig.show()

    elif library=='matplotlib':
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.2, edgecolors='none')
        ax.grid(False)

        ax.set_xlim3d(-4, 4)
        ax.set_ylim3d(-4, 4)
        ax.set_zlim3d(-4, 4)
        plt.show()


def plot_critical_points(pts: np.ndarray, crit_indices: np.ndarray) -> None:
    """
    Plots the given points and high lights the critical point indices 
    in a different color
    
    Args:
    -   pts: numpy array of shape (n, 3) where n is the number of points to plot
    -   crit_indices: array of shape (c,) where c is the number of critical points to highlight
    """
    color = ["red" if i in crit_indices else "blue" for i in range(0, pts.shape[0])]
    
    fig = make_subplots(
        rows=1, 
        cols=2, 
        subplot_titles=("All Points", "Critical Points"),
        specs=[[{'type': 'surface'}, {'type': 'surface'}]]
    )

    fig.add_trace(
        trace=go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], 
            line=go.scatter3d.Line(color='rgba(0,0,0,0)'), 
            marker=go.scatter3d.Marker(color=color)
        ),
        row=1,col=1
    )

    fig.add_trace(
        trace=go.Scatter3d(
            x=pts[crit_indices, 0], y=pts[crit_indices, 1], z=pts[crit_indices, 2], 
            line=go.scatter3d.Line(color='rgba(0,0,0,0)'), 
            marker=go.scatter3d.Marker(color='red')
        ),
        row=1,col=2
    )

    fig.show()


def plot_crit_points_from_file(model: Union[PointNet, PointNetPosEncoding], file_path: str, pad_size: int) -> None:
    """
    Plots critical points as determined by the given model for the point
    cloud stored in the given file
    
    Args:
    -   model: model to use to calculate the critical points
    -   file_path: path to file containing point cloud
    -   pad_size: number of points model is expecting in each point cloud
    """
    pts = torch.tensor(read_points_file(file_path)).float()
    pts_full = torch.zeros((pad_size, 3))
    pts_full[:pts.shape[0], :] = pts
    pts_full[pts.shape[0]:, :] = pts_full[0]
    crit_indices = get_critical_indices(model, pts_full)
    plot_critical_points(pts_full, crit_indices)


def plot_cloud_from_file(file_path: str, library='plotly') -> None:
    """
    Plots point cloud in the given file using the specified library
    
    Args:
    -   file_path: path to file that contains point cloud to plot
    -   library: either plotly or matplotlib specifying which library to use to plot the points
    """
    pts = read_points_file(file_path)
    plot_point_cloud(pts, library)