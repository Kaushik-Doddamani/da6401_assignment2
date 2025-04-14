import torch
import torch.nn as nn

class MyCNN(nn.Module):
    """
    A modular CNN architecture:
      - 5 x (Conv -> Activation -> MaxPool)
      - 1 Dense (fully connected) layer of n neurons
      - 1 Output layer of 10 neurons

    Accepts images of size (3 x H x W).
    After 5 max-pool operations (each halves H and W),
    final feature map size is (m x (H/32) x (W/32)) if H and W are multiples of 32.
    """


    def __init__(self,
                 in_channels=3,
                 num_filters=16,     # m
                 kernel_size=3,      # k
                 activation_fn=nn.ReLU,
                 dense_neurons=128,  # n
                 image_height=224,   # default
                 image_width=224     # default
                 ):
        """
        :param in_channels:   Number of input channels (3 for RGB images)
        :param num_filters:   m = number of filters in each Conv layer
        :param kernel_size:   k = kernel size of each Conv filter (k x k)
        :param activation_fn: Pytorch activation class, e.g., nn.ReLU
        :param dense_neurons: n = number of neurons in the fully connected layer
        :param image_height:  The height of the input image (assumed multiple of 32)
        :param image_width:   The width of the input image (assumed multiple of 32)
        """
        super(MyCNN, self).__init__()

        # We assume 'same' padding, i.e., output of conv has same spatial size
        # Then each MaxPool(2x2) halves the H and W each time.
        padding = kernel_size // 2
        Act = activation_fn  # for readability

        #-------------------------
        # 1) Block 1
        #   Conv(in_channels->m), Act, MaxPool(2x2)
        #-------------------------
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        self.act1 = Act()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        #-------------------------
        # 2) Block 2
        #   Conv(m->m), Act, MaxPool(2x2)
        #-------------------------
        self.conv2 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        self.act2 = Act()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #-------------------------
        # 3) Block 3
        #   Conv(m->m), Act, MaxPool(2x2)
        #-------------------------
        self.conv3 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        self.act3 = Act()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        #-------------------------
        # 4) Block 4
        #   Conv(m->m), Act, MaxPool(2x2)
        #-------------------------
        self.conv4 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        self.act4 = Act()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        #-------------------------
        # 5) Block 5
        #   Conv(m->m), Act, MaxPool(2x2)
        #-------------------------
        self.conv5 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        self.act5 = Act()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        #--------------------------------------
        # Compute final feature map dimension.
        # Each pool halves H and W => H/32, W/32
        #--------------------------------------
        reduced_height = image_height // 32
        reduced_width  = image_width // 32
        self.flatten_dim = num_filters * reduced_height * reduced_width

        # Dense layer
        self.fc1 = nn.Linear(self.flatten_dim, dense_neurons)
        self.act_fc1 = Act()

        # Output layer: 10 neurons
        self.output = nn.Linear(dense_neurons, 10)

    def forward(self, x):
        # x: (batch_size, 3, H, W)

        # Block 1
        x = self.conv1(x)    # (batch_size, m, H, W)
        x = self.act1(x)     # (batch_size, m, H, W)
        x = self.pool1(x)    # (batch_size, m, H/2, W/2)

        # Block 2
        x = self.conv2(x)    # (batch_size, m, H/2, W/2)
        x = self.act2(x)     # (batch_size, m, H/2, W/2)
        x = self.pool2(x)    # (batch_size, m, H/4, W/4)

        # Block 3
        x = self.conv3(x)    # (batch_size, m, H/4, W/4)
        x = self.act3(x)     # (batch_size, m, H/4, W/4)
        x = self.pool3(x)    # (batch_size, m, H/8, W/8)

        # Block 4
        x = self.conv4(x)    # (batch_size, m, H/8, W/8)
        x = self.act4(x)     # (batch_size, m, H/8, W/8)
        x = self.pool4(x)    # (batch_size, m, H/16, W/16)

        # Block 5
        x = self.conv5(x)    # (batch_size, m, H/16, W/16)
        x = self.act5(x)     # (batch_size, m, H/16, W/16)
        x = self.pool5(x)    # (batch_size, m, H/32, W/32)

        # Flatten
        x = x.view(x.size(0), -1)  # (batch_size, m * (H/32) * (W/32))

        # Dense
        x = self.fc1(x)            # (batch_size, dense_neurons)
        x = self.act_fc1(x)        # (batch_size, dense_neurons)

        # Output: 10 classes
        x = self.output(x)         # (batch_size, 10)

        return x