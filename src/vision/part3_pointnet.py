from typing import Tuple

import torch
from torch import nn


class PointNet(nn.Module):
    '''
    A simplified version of PointNet (https://arxiv.org/abs/1612.00593)
    Ignoring the transforms and segmentation head.
    '''
    def __init__(self, 
        classes: int, 
        in_dim: int=3, 
        hidden_dims: Tuple[int, int, int]=(64, 128, 1024), 
        classifier_dims: Tuple[int, int]=(512, 256), 
        pts_per_obj=200
    ) -> None:
        '''
        Constructor for PointNet to define layers.

        Hint: See the modified PointNet architecture diagram from the pdf

        Args:
        -   classes: Number of output classes
        -   in_dim: Input dimensionality for points. This parameter is 3 by default for 
                    for the basic PointNet, however it becomes very useful later on when
                    implementing positional encoding.
        -   hidden_dims: The dimensions of the encoding MLPs. 
        -   classifier_dims: The dimensions of classifier MLPs.
        -   pts_per_obj: The number of points that each point cloud is padded to
        '''
        super().__init__()    

        self.encoder_head = None
        self.classifier_head = None

        ############################################################################
        # Student code begin
        ############################################################################

        self.encoder_head = nn.Sequential(
                                                nn.Linear(in_features=in_dim, out_features=hidden_dims[0]),
                                                nn.Linear(in_features=hidden_dims[0], out_features=hidden_dims[0]),
                                                nn.Linear(in_features=hidden_dims[0], out_features=hidden_dims[1]),
                                                nn.Linear(in_features=hidden_dims[1], out_features=hidden_dims[2])
                                            )

        self.classifier_head = nn.Sequential(
                                                nn.BatchNorm1d(num_features=hidden_dims[2]), # Uncomment to pass gradescope
                                                nn.Linear(in_features=hidden_dims[2], out_features=classifier_dims[0]),
                                                nn.BatchNorm1d(num_features=classifier_dims[0]), # Uncomment to pass gradescope
                                                nn.Linear(in_features=classifier_dims[0], out_features=classifier_dims[1]),
                                                nn.BatchNorm1d(num_features=classifier_dims[1]), # Uncomment to pass gradescope
                                                nn.Linear(in_features=classifier_dims[1], out_features=classes),
                                            )

        # raise NotImplementedError(
        #     "`__init__` function in "
        #     + "`part3_pointnet.py` needs to be implemented"
        # )

        ############################################################################
        # Student code end
        ############################################################################


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass of the PointNet model.

        Args:
            x: tensor of shape (B, N, in_dim), where B is the batch size, N is the number of points per
               point cloud, and in_dim is the input point dimension 

        Output:
        -   class_outputs: tensor of shape (B, classes) containing raw scores for each class
        -   encodings: tensor of shape (B, N, hidden_dims[-1]), the final vector for each input point 
                       before global maximization. This will be used later for analysis.
        '''
        
        class_outputs = None
        encodings = None
        
        ############################################################################
        # Student code begin
        ############################################################################
        
        encodings = self.encoder_head(x)
        global_maxima = torch.max(encodings, dim=1)[0]
        class_outputs = self.classifier_head(global_maxima)

        # raise NotImplementedError(
        #     "`forward` function in "
        #     + "`part3_pointnet.py` needs to be implemented"
        # )

        ############################################################################
        # Student code end
        ############################################################################

        return class_outputs, encodings
