import torch
import torch.nn as nn

class SimpleNetFinal(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means
        """
        super().__init__()

        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()
        self.loss_criterion = None

        ############################################################################
        # Student code begin
        ############################################################################

        self.conv_layers = nn.Sequential(
                                            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
                                            nn.BatchNorm2d(10),
                                            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(in_channels=10, out_channels=15, kernel_size=5, stride=1, padding=0),
                                            nn.BatchNorm2d(15),
                                            nn.MaxPool2d(kernel_size=3, stride=3),
                                            nn.ReLU(),
                                            nn.Dropout(p=0.5),
                                            nn.Conv2d(in_channels=15, out_channels=20, kernel_size=5, stride=1, padding=0),
                                            nn.MaxPool2d(kernel_size=3, stride=3),
                                            nn.ReLU(),
                                        )
        self.fc_layers = nn.Sequential(
                                        nn.Flatten(),
                                        nn.Linear(in_features=500, out_features=200),
                                        nn.ReLU(),
                                        nn.Linear(in_features=200, out_features=15)
                                        )

        # self.conv_layers = nn.Sequential(
        #                                     nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0),
        #                                     nn.MaxPool2d(kernel_size=3, stride=3),
        #                                     nn.ReLU(),
        #                                     # nn.Dropout(p=0.5),
        #                                     nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0),
        #                                     nn.MaxPool2d(kernel_size=3, stride=3),
        #                                     nn.ReLU()
        #                                 )
        # self.fc_layers = nn.Sequential(
        #                                 nn.Flatten(),
        #                                 nn.Linear(in_features=500, out_features=200),
        #                                 nn.ReLU(),
        #                                 nn.Linear(in_features=200, out_features=15)
        #                                 )

        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')

        # raise NotImplementedError(
        #     "`__init__` function in "
        #     + "`simple_net_final.py` needs to be implemented"
        # )

        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the net

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        """
        model_output = None
        ############################################################################
        # Student code begin
        ############################################################################

        model_output = self.conv_layers(x)
        model_output = self.fc_layers(model_output)
        
        # raise NotImplementedError(
        #     "`forward` function in "
        #     + "`simple_net_final.py` needs to be implemented"
        # )
        
        ############################################################################
        # Student code end
        ############################################################################

        return model_output
