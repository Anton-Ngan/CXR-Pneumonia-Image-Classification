from torch import nn

class PneumoniaModel(nn.Module):

  def __init__(self,
               input_shape: int,
               hidden_units: int,
               output_shape: int) -> None:
    super().__init__()
    self.cnn = nn.Sequential(

        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=4,
                  padding=2
                  ),
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units),

        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=4,
                  padding=2
                  ),
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units),                  
        nn.MaxPool2d(kernel_size=3),

        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=4,
                  padding=2
                  ),
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units),

        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=4,
                  padding=2
                  ),
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units),
        nn.MaxPool2d(kernel_size=3),

        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=4,
                  padding=2
                  ),
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units),

        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=4,
                  padding=2
                  ),
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units),
        nn.MaxPool2d(kernel_size=3),

        nn.Dropout(p=0.5),
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*9*9,
                  out_features=output_shape)
    )
  
  def forward(self, x):
    return self.cnn(x)