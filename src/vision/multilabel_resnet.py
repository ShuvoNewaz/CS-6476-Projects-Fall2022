import torch
import torch.nn as nn
from torchvision.models import resnet18


class MultilabelResNet18(nn.Module):
    def __init__(self):
        """Initialize network layers.

        Note: Do not forget to freeze the layers of ResNet except the last one
        Note: Consider which activation function to use
        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        Download pretrained resnet using pytorch's API (Hint: see the import statements)
        """
        super().__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = None
        self.activation = None

        ############################################################################
        # Student code begin
        ############################################################################

        my_resnet = resnet18(pretrained=True)
        for param in my_resnet.parameters():
            param.requires_grad = False

        num_features = my_resnet.fc.in_features
        my_resnet.fc = nn.Linear(num_features, 7)
        self.my_resnet = my_resnet

        # self.conv_layers = nn.Sequential(*(list(my_resnet.children())[:-1]))
        # self.fc_layers = nn.Linear(num_features, 7)

        self.conv_layers = nn.Sequential(my_resnet.layer1, my_resnet.layer2, my_resnet.layer3, my_resnet.layer4)
        self.fc_layers = my_resnet.fc

        self.loss_criterion = nn.BCELoss(reduction='mean')
        self.activation = nn.Sigmoid()

        # raise NotImplementedError(
        #     "`__init__` function in "
        #     + "`multi_resnet.py` needs to be implemented"
        # )

        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass with the net, duplicating grayscale channel to 3-channel.

        Args:
            x: tensor of shape (N,C,H,W) representing input batch of images
        Returns:
            y: tensor of shape (N,num_classes) representing the output (raw scores) of the net
                Note: we set num_classes=15
        """
        model_output = None
        x = x.repeat(1, 3, 1, 1)  # as ResNet accepts 3-channel color images
        ############################################################################
        # Student code begin
        ############################################################################

        model_output = self.my_resnet(x)
        model_output = self.activation(model_output)
        
        # raise NotImplementedError(
        #     "`forward` function in "
        #     + "`multi_resnet.py` needs to be implemented"
        # )

        ############################################################################
        # Student code end
        ############################################################################
        return model_output
