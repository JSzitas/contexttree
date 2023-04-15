from torch import nn

class ContextualNetNetwork(nn.Module):
    """Contextual Network for the Context Tree algorithm.
    """

    def __init__(self,
                 input_size: int = 100,
                 layer_dims: [int] = [120, 120, 120],
                 dropout_rate: float = 0.2,
                 n_tree: int = 50) -> self:  # noqa: F821
        """Initialize Contextual Tree Neural Network.
        Args:
         - input_size (int): An integer specifying the input dimension.
         - layer_dims ([int]): A list containing the linear layer sizes - implictly,
          the number of elements of this list determines the number of hidden layers.
         - dropout_rate (float): A floating giving the dropout fraction (always applied
           after the linear layer, same for all layers).
         - n_tree (int): The number of trees to train for - this determines output size.

        Returns:
         self
        """
        assert len(layer_dims) > 0, "The neural network must have hidden layers," +\
                                    " but you provided a 0 length list."


        super(nn.Module, self).__init__()
        # initialize net object 
        net = nn.Sequential()
        # the size of the first layer is partially determined by input size 
        net.add(nn.Linear(input_size, layer_dims[0]))
        net.add(nn.Dropout(p=dropout_rate))
        net.add(nn.ReLU())
        # iterate over remaining list elements 
        for i in range(1, len(layer_dims)-1):
            net.add(nn.Linear(layer_dims[i-1], layer_dims[i]))
            net.add(nn.Dropout(p=dropout_rate))
            net.add(nn.ReLU())
        # final layer 
        net.add(nn.Linear(layer_dims[len(layer_dims)-1], n_tree))
        # set class attribute
        self.net = net
        return self

    def forward(self, x):
        return self.net(x)
