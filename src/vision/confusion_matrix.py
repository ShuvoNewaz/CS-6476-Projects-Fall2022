from typing import Sequence, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from vision.image_loader import ImageLoader
from torch import nn
from torch.utils.data import DataLoader, Subset


def generate_confusion_data(
    model: nn.Module,
    dataset: ImageLoader,
    use_cuda: bool = False,
) -> Tuple[Sequence[int], Sequence[int], Sequence[str]]:
    """
    Get the accuracy on the val/train dataset

    Args:
    -   model: Model to generate confusion matrix data for
    -   dataset: The ImageLoader dataset that corresponds to training or val data
    -   use_cuda: Whether to evaluate on CPU or GPU

    Returns:
    -   targets: a numpy array of shape (N) containing the targets indices
    -   preds: a numpy array of shape (N) containing the predicted indices
    -   class_labels: A list containing the class labels at the index of their label_number
                      e.g. if the labels are {"Cat": 0, "Monkey": 2, "Dog": 1},
                           the return value should be ["Cat", "Dog", "Monkey"]
    """

    batch_size = 32
    cuda = use_cuda and torch.cuda.is_available()
    dataloader_args = {"num_workers": 1, "pin_memory": True} if cuda else {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **dataloader_args)

    preds = np.zeros(len(dataset)).astype(np.int32)
    targets = np.zeros(len(dataset)).astype(np.int32)
    label_to_idx = dataset.get_classes()
    class_labels = [""] * len(label_to_idx)

    model.eval()
    ##########################################################################
    # Student code begins here
    ##########################################################################

    raise NotImplementedError(
        "`generate_confusion_data` function in "
        + "`confusion_matrix.py` needs to be implemented"
    )

    ##########################################################################
    # Student code ends here
    ##########################################################################
    model.train()

    return targets.cpu().numpy(), preds.cpu().numpy(), class_labels


def generate_confusion_matrix(
    targets: np.ndarray, preds: np.ndarray, num_classes: int, normalize=True
) -> np.ndarray:
    """Generate the actual confusion matrix values

    The confusion matrix is a num_classes x num_classes matrix that shows the
    number of classifications made to a predicted class, given a ground truth class

    If the classifications are:
        ground_truths: [1, 0, 1, 2, 0, 1, 0, 2, 2]
        predicted:     [1, 1, 0, 2, 0, 1, 1, 2, 0]

    Then the confusion matrix is:
        [1 2 0],
        [1 1 0],
        [1 0 2],

    Each ground_truth value corresponds to a row,
    and each predicted value is a column

    A confusion matrix can be normalized by dividing all the entries of
    each ground_truth prior by the number of actual instances of the ground truth
    in the dataset.

    Args:
    -   targets: a numpy array of shape (N) containing the targets indices
    -   preds: a numpy array of shape (N) containing the predicted indices
    -   num_classes: Number of classes in the confusion matrix
    -   normalize: Whether to normalize the confusion matrix or not
    Returns:
    -   confusion_matrix: a (num_classes, num_classes) numpy array
                          representing the confusion matrix
    """

    confusion_matrix = np.zeros((num_classes, num_classes))

    for target, prediction in zip(targets, preds):
        ##########################################################################
        # Student code begins here
        ##########################################################################
    
        raise NotImplementedError(
            "`generate_confusion_matrix` function in "
            + "`confusion_matrix.py` needs to be implemented"
        )

        ##########################################################################
        # Student code ends here
        ##########################################################################

    if normalize:
        ##########################################################################
        # Student code begins here
        ##########################################################################
    
        raise NotImplementedError(
            "`generate_confusion_matrix` function in "
            + "`confusion_matrix.py` needs to be implemented"
        )
    
        ##########################################################################
        # Student code ends here
        ##########################################################################
    return confusion_matrix


def plot_confusion_matrix(
    confusion_matrix: np.ndarray, class_labels: Sequence[str]
) -> None:
    """Plots the confusion matrix

    Args:
    -   confusion_matrix: a (num_classes, num_classes) numpy array
                          representing the confusion matrix
    -   class_labels: A list containing the class labels at the index of their label_number
                      e.g. if the labels are {"Cat": 0, "Monkey": 2, "Dog": 1},
                           the return value should be ["Cat", "Dog", "Monkey"]
                      The length of class_labels should be num_classes
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
    model: nn.Module, dataset: ImageLoader, use_cuda: bool = False
) -> None:
    """Runs the entire confusion matrix pipeline for convenience

    Args:
    -   model: Model to generate confusion matrix data for
    -   dataset: The ImageLoader dataset that corresponds to training or validation data
    -   use_cuda: Whether to evaluate on CPU or GPU
    """

    targets, predictions, class_labels = generate_confusion_data(
        model, dataset, use_cuda=use_cuda
    )

    confusion_matrix = generate_confusion_matrix(
        np.array(targets, dtype=np.int32),
        np.array(predictions, np.int32),
        len(class_labels),
    )

    plot_confusion_matrix(confusion_matrix, class_labels)


def get_pred_images_for_target(
    model: nn.Module,
    dataset: ImageLoader,
    predicted_class: int,
    target_class: int,
    use_cuda: bool = False,
) -> Sequence[str]:
    """Returns a list of image paths that correspond to a particular prediction
    for a given target class

    Args:
    -   model: Model to generate confusion matrix data for
    -   dataset: The ImageLoader dataset that corresponds to training or validation data
    -   predicted_class: The class predicted by the model
    -   target_class: The actual class of the image
    -   use_cuda: Whether to evaluate on CPU or GPU

    Returns:
    -   valid_image_paths: Image paths that are classified as <predicted_class>
                           but actually belong to <target_class>
    """
    model.eval()
    dataset_list = dataset.dataset
    indices = []
    image_paths = []
    for i, (image_path, class_label) in enumerate(dataset_list):
        if class_label == target_class:
            indices.append(i)
            image_paths.append(image_path)
    subset = Subset(dataset, indices)
    dataloader_args = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    loader = DataLoader(subset, batch_size=32, shuffle=False, **dataloader_args)
    preds = []
    for i, (inp, _) in enumerate(loader):
        if use_cuda:
            inp = inp.cuda()
        logits = model(inp)
        p = torch.argmax(logits, dim=1)
        preds.append(p)
    predictions = torch.cat(preds, dim=0).cpu().tolist()
    valid_image_paths = [
        image_paths[i] for i, p in enumerate(predictions) if p == predicted_class
    ]
    model.train()
    return valid_image_paths
