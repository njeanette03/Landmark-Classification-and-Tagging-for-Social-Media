import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        self.model = nn.Sequential(
            # define layers
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # convolutional layer 1
            nn.ReLU(), # activation
            nn.MaxPool2d(2, 2), # max pooling 
            nn.BatchNorm2d(32), # batch normalization
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # convolutional layer 2
            nn.ReLU(), # activation
            nn.MaxPool2d(2, 2), # max pooling 
            nn.BatchNorm2d(64), # batch normalization
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # convolutional layer 3
            nn.ReLU(), # activation
            nn.MaxPool2d(2, 2), # max pooling 
            nn.BatchNorm2d(128), # batch normalization
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # convolutional layer 4
            nn.ReLU(), # activation
            nn.MaxPool2d(2, 2), # max pooling 
            nn.BatchNorm2d(256), # batch normalization
            
            # flattening
            nn.Flatten(),
            nn.Dropout(p=dropout), # dropout
            nn.Linear(14*14*256, 512),
            nn.BatchNorm1d(512), # batch normalization
            nn.ReLU(), # activation
            nn.Linear(512, num_classes) # output layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return self.model(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
