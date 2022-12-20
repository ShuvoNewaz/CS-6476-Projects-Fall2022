from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from vision.part2_baseline import Baseline
from vision.part3_pointnet import PointNet
from vision.part5_positional_encoding import PointNetPosEncoding


def get_critical_indices(model: Union[PointNet, PointNetPosEncoding], pts: torch.Tensor) -> np.ndarray:
    '''
    Finds the indices of the cirtical points in the given point cloud. A
    critical point is a point that contributes to the global feature (i.e
    a point whose calculated feature has a maximal value in at least one 
    of its dimensions)
    
    Hint:
    1) Use the encodings returned by your model
    2) Make sure you aren't double-counting points since points may
       contribute to the global feature in more than one dimension

    Inputs:
        model: The trained model
        pts: (model.pad_size, 3) tensor point cloud representing an object

    Returns:
        crit_indices: (N,) numpy array, where N is the number of critical pts

    '''
    crit_indices = None

    ############################################################################
    # Student code begin
    ############################################################################

    if len(pts.shape) == 2:
            pts = torch.unsqueeze(pts, 0)
    encodings = model(pts)[1]
    crit_indices = torch.argmax(encodings, dim=1)
    crit_indices = torch.unique(crit_indices)

    # raise NotImplementedError(
    #     "`get_critical_indices` function in "
    #     + "`part4_pointnet.py` needs to be implemented"
    # )

    ############################################################################
    # Student code end
    ############################################################################

    return crit_indices

    
def get_confusion_matrix(
    model: Union[Baseline, PointNet, PointNetPosEncoding], 
    loader: DataLoader, 
    num_classes: int,
    normalize: bool=True, 
    device='cpu'
) -> np.ndarray:
    '''
    Builds a confusion matrix for the given models predictions
    on the given dataset. 
    
    Recall that each ground truth label corresponds to a row in
    the matrix and each predicted value corresponds to a column.

    A confusion matrix can be normalized by dividing entries for
    each ground truch prior by the number of actual isntances the
    ground truth appears in the dataset. (Think about what this means
    in terms of rows and columns in the matrix) 

    Hint:
    1) Generate list of prediction, ground-truth pairs
    2) For each pair, increment the correct cell in the matrix
    3) Keep track of how many instances you see of each ground truth label
       as you go and use this to normalize 

    Args: 
    -   model: The model to use to generate predictions
    -   loader: The dataset to use when generating predictions
    -   num_classes: The number of classes in the dataset
    -   normalize: Whether or not to normalize the matrix
    -   device: If 'cuda' then run on GPU. Run on CPU by default

    Output:
    -   confusion_matrix: a numpy array with shape (num_classes, num_classes)
                          representing the confusion matrix
    '''

    model.eval()
    confusion_matrix = None

    ############################################################################
    # Student code begin
    ############################################################################

    for i, pt in enumerate(loader):
        one_hot_prediction = model(pt[0])[0]
        if i == 0:
            one_hot_predictions = one_hot_prediction
            targets = pt[1]
        else:
            one_hot_predictions = torch.concat((one_hot_predictions, one_hot_prediction))
            targets = torch.concat((targets, pt[1]))
    predictions = torch.argmax(one_hot_predictions, dim=1)
    confusion_matrix = np.zeros((num_classes, num_classes))
    for target, prediction in zip(targets, predictions):
        confusion_matrix[target, prediction] += 1
    if normalize:
         confusion_matrix /= np.sum(confusion_matrix, axis=1, keepdims=True)

    # raise NotImplementedError(
    #     "`get_confusion_matrix` function in "
    #     + "`part4_pointnet.py` needs to be implemented"
    # )

    ############################################################################
    # Student code end
    ############################################################################

    model.train()

    return confusion_matrix
