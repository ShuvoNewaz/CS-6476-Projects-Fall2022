from typing import Tuple

import torch
from vision.part3_pointnet import PointNet
from torch import nn


class PointNetPosEncoding(nn.Module):

    def __init__(
        self, 
        classes: int, 
        hidden_dims: Tuple[int, int, int]=(64, 128, 1024), 
        classifier_dims: Tuple[int, int]=(512, 256), 
        encoding_dim=10, 
        pts_per_obj=200
    ) -> None:
        '''
        Constructor for PointNet with positional encoding. The main difference between our 
        original PointNet model and this one the addition of a positional encoding using
        the points of our input point clouds.

        There are a couple ways we can use positional encoding. One is to simply replace
        the input point cloud with its positional encoding. Another option is to
        append the positional encoding output to your original point cloud. Either is 
        viable and comes with its own advantages and drawbacks. We recommend trying
        both options and seeing which one does better! 

        Hint: 
        1) Think about how to reuse your PointNet implementation from earlier

        Args:
        -   classes: Number of output classes
        -   hidden_dims: The dimensions of the encoding MLPs. 
        -   classifier_dims: The dimensions of classifier MLPs. 
        -   encoding_dim: The number of frequencies to use with positional encoding
        -   pts_per_obj: The number of points that each point cloud is padded to
        '''
        super().__init__()

        self.encoding_dim = None
        self.point_net = None
            
        ############################################################################
        # Student code begin
        ############################################################################

        self.encoding_dim = encoding_dim
        self.point_net = PointNet(      
                                        in_dim=6*encoding_dim,
                                        classes=classes, 
                                        hidden_dims=(64, 128, 1024), 
                                        classifier_dims=(512, 256), 
                                        pts_per_obj=pts_per_obj
                                    )

        # raise NotImplementedError(
        #     "`__init__` function in "
        #     + "`part5_positional_encoding.py` needs to be implemented"
        # )

        ############################################################################
        # Student code end
        ############################################################################


    def positional_encoding(self, xyz: torch.Tensor) -> torch.Tensor:
        '''
        Takes a given xyz input tensor and applies a sine and cosine positional encoding 
        as done in NeRF (https://arxiv.org/abs/2003.08934, section 5.1). The idea is that
        this helps deal with spectral bias, resulting in better performance with high 
        frequency features. (Also see the project pdf for more information on this)

        Args:
        -   xyz: tensor of shape (B, pts_per_obj, 3), where B is the batch size
                 and pts_per_obj is the number of points per sample,
        Output:
        -   enc: tensor of shape (B, N, 6L), where L is the encoding dimension (encoding_dim
            is passed into the constructor for PointNetPosEncoding) which is the number of
            frequencies to use when calculating the encoding.
        '''
        
        enc = None
        
        ############################################################################
        # Student code begin
        ############################################################################

        x, y, z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]
        B, N = xyz.shape[:2]
        enc = torch.zeros((B, N, 6*self.encoding_dim))

        for i in range(6):
            for j in range(self.encoding_dim):
                if i in [0, 1]:
                    enc[:, :, 2*j] = torch.sin(2**j * torch.pi * x)
                    enc[:, :, 2*j + 1] = torch.cos(2**j * torch.pi * x)
                if i in [2, 3]:
                    enc[:, :, 2*j + 2*self.encoding_dim] = torch.sin(2**j * torch.pi * y)
                    enc[:, :, 2*j + 1 + 2*self.encoding_dim] = torch.cos(2**j * torch.pi * y)
                if i in [4, 5]:
                    enc[:, :, 2*j + 4*self.encoding_dim] = torch.sin(2**j * torch.pi * z)
                    enc[:, :, 2*j + 1 + 4*self.encoding_dim] = torch.cos(2**j * torch.pi * z)

        # raise NotImplementedError(
        #     "`positional_encoding` function in "
        #     + "`part5_positional_encoding.py` needs to be implemented"
        # )

        ############################################################################
        # Student code end
        ############################################################################
        
        return enc
        

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass of the PointNet model.

        Hint:
        1) Use the positional_encoding method you implemented
        2) Decide how you want to use the positional encoding
            a) Replace the original point cloud with its positional encoding
            b) Append the positional encoding to the original point cloud 
        3) Use your original PointNet architecture

        Args:
        -   x: tensor of shape (B, pts_per_obj, 3), where B is the batch size and 
               pts_per_obj is the number of points per point cloud

        Outputs:
        -   class_outputs: tensor of shape (B, classes) containing raw scores for each class
        -   encodings: tensor of shape (B, N, hidden_dims[-1]), the final vector for each input point 
                       before global maximization. This will be used later for analysis.
        '''
        class_outputs = None
        encodings = None

        ############################################################################
        # Student code begin
        ############################################################################

        enc = self.positional_encoding(x)
        encodings = self.point_net.encoder_head(enc)
        global_maxima = torch.max(encodings, dim=1)[0]
        class_outputs = self.point_net.classifier_head(global_maxima)

        # raise NotImplementedError(
        #     "`forward` function in "
        #     + "`part5_positional_encoding.py` needs to be implemented"
        # )

        ############################################################################
        # Student code end
        ############################################################################

        return class_outputs, encodings

