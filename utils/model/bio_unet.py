from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ConvTranspose2d, ReLU, MaxPool2d, Sigmoid, Parameter
from torch import tensor, cat


class BiONet(Module):

    def __init__(self,
                 num_classes: int = 1,
                 iterations: int = 2,
                 multiplier: float = 1.0,
                 num_layers: int = 3,
                 integrate: bool = False):

        super(BiONet, self).__init__()
        # Parameters
        self.iterations = iterations
        self.multiplier = multiplier
        self.num_layers = num_layers
        self.integrate = integrate
        self.batch_norm_momentum = 0.01
        # Generate channel parameters, the channels start from the first Encoder output to the semantic vector
        self.filters_list = [int(64 * (2 ** i) * self.multiplier) for i in range(self.num_layers + 1)]
        # Pre-processing convolutional block that does not participate in the loop, finally outputting 32*256*256
        self.pre_transform_conv_block = Sequential(
            Conv2d(2, self.filters_list[0], kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),  # Generate f[1]*512*512
            ReLU(),
            BatchNorm2d(self.filters_list[0], momentum=self.batch_norm_momentum),
            Conv2d(self.filters_list[0], self.filters_list[0], kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            # Generate f[1]*512*512
            ReLU(),
            BatchNorm2d(self.filters_list[0], momentum=self.batch_norm_momentum),
            Conv2d(self.filters_list[0], self.filters_list[0], kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            # Generate f[1]*512*512
            ReLU(),
            BatchNorm2d(self.filters_list[0], momentum=self.batch_norm_momentum),
            MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        )
        self.reuse_convs = []  # Reusable convolution kernels for encoders, each encoder corresponds to a tuple (3 convolution kernels, excluding ReLU)
        self.encoders = []  # List of encoders. Since part of the encoder does not participate in the loop, each encoder is a tuple (two CONV Sequentials, DOWN)
        self.reuse_deconvs = []  # Reusable convolution and deconvolution kernels for decoders, each decoder corresponds to a tuple (3 convolution kernels, excluding ReLU)
        self.decoders = []  # List of decoders. Since part of the decoder does not participate in the loop, each decoder is a tuple (two CONV Sequentials, UP)
        for iteration in range(self.iterations):
            for layer in range(self.num_layers):

                # Create the encoder part. Although some code can be combined, it is separated for clarity (and the constructor doesn't require efficiency).
                # Constants related to the layer
                in_channel = self.filters_list[layer] * 2  # Double the input channels as there is data from the output part
                mid_channel = self.filters_list[layer]
                out_channel = self.filters_list[layer + 1]
                # Create encoder model
                if iteration == 0:
                    # Create and add reusable convolution kernels
                    # Only the last convolution kernel is responsible for increasing the channels
                    conv1 = Conv2d(in_channel, mid_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
                    conv2 = Conv2d(mid_channel, mid_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
                    conv3 = Conv2d(mid_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
                    self.reuse_convs.append((conv1, conv2, conv3))
                # Create encoder
                # Construct two CONVs first
                convs = Sequential(
                    self.reuse_convs[layer][0],
                    ReLU(),
                    BatchNorm2d(mid_channel, momentum=self.batch_norm_momentum),
                    self.reuse_convs[layer][1],
                    ReLU(),
                    BatchNorm2d(mid_channel, momentum=self.batch_norm_momentum)
                )
                # Construct DOWN
                down = Sequential(
                    self.reuse_convs[layer][2],
                    ReLU(),
                    BatchNorm2d(out_channel, momentum=self.batch_norm_momentum),
                    MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
                )
                self.add_module("iteration{0}_layer{1}_encoder_convs".format(iteration, layer), convs)
                self.add_module("iteration{0}_layer{1}_encoder_down".format(iteration, layer), down)
                self.encoders.append((convs, down))
                
                # Create the decoder part, following the encoder structure
                # Constants related to the layer. Note that this part does not need mid_channel since the first convolution kernel already increases the dimension.
                in_channel = self.filters_list[self.num_layers - layer] + self.filters_list[self.num_layers - 1 - layer]
                out_channel = self.filters_list[self.num_layers - 1 - layer]
                # Create decoder model
                if iteration == 0:
                    # Create and add reusable convolution kernels
                    # Increase the number of channels from the first convolution kernel
                    conv1 = Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
                    conv2 = Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
                    conv3 = ConvTranspose2d(out_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2),
                                            output_padding=(1, 1))  # This part differs from TensorFlow, it requires using output_padding for complete shape
                    self.reuse_deconvs.append((conv1, conv2, conv3))
                # Create decoder
                # Construct two CONVs first
                convs = Sequential(
                    self.reuse_deconvs[layer][0],
                    ReLU(),
                    BatchNorm2d(out_channel, momentum=self.batch_norm_momentum),
                    self.reuse_deconvs[layer][1],
                    ReLU(),
                    BatchNorm2d(out_channel, momentum=self.batch_norm_momentum)
                )
                # Construct UP
                up = Sequential(
                    self.reuse_deconvs[layer][2],
                    ReLU(),
                    BatchNorm2d(out_channel, momentum=self.batch_norm_momentum)
                )
                self.add_module("iteration{0}_layer{1}_decoder_convs".format(iteration, layer), convs)
                self.add_module("iteration{0}_layer{1}_decoder_up".format(iteration, layer), up)
                self.decoders.append((convs, up))
        # Create the middle layer
        self.middles = Sequential(
            Conv2d(self.filters_list[-1], self.filters_list[-1], kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            ReLU(),
            BatchNorm2d(self.filters_list[-1], momentum=self.batch_norm_momentum),
            Conv2d(self.filters_list[-1], self.filters_list[-1], kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            ReLU(),
            BatchNorm2d(self.filters_list[-1], momentum=self.batch_norm_momentum),
            ConvTranspose2d(self.filters_list[-1], self.filters_list[-1], kernel_size=(3, 3), padding=(1, 1),
                            stride=(2, 2), output_padding=(1, 1)),
            ReLU(),
            BatchNorm2d(self.filters_list[-1], momentum=self.batch_norm_momentum)
        )
        self.post_transform_conv_block = Sequential(
            Conv2d(self.filters_list[0] * self.iterations, self.filters_list[0], kernel_size=(3, 3), padding=(1, 1),
                   stride=(1, 1)) if self.integrate else Conv2d(self.filters_list[0],
                                                                self.filters_list[0], kernel_size=(3, 3),
                                                                padding=(1, 1), stride=(1, 1)),
            ReLU(),
            BatchNorm2d(self.filters_list[0], momentum=self.batch_norm_momentum),
            Conv2d(self.filters_list[0], self.filters_list[0], kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            ReLU(),
            BatchNorm2d(self.filters_list[0], momentum=self.batch_norm_momentum),
            Conv2d(self.filters_list[0], 2, kernel_size=(1, 1), stride=(1, 1)),
            Sigmoid(),
        )

    def forward(self, x: tensor) -> tensor:
        enc = [None for i in range(self.num_layers)]
        dec = [None for i in range(self.num_layers)]
        all_output = [None for i in range(self.iterations)]
        x = self.pre_transform_conv_block(x)
        e_i = 0
        d_i = 0
        for iteration in range(self.iterations):
            for layer in range(self.num_layers):
                if layer == 0:
                    x_in = x
                x_in = self.encoders[e_i][0](cat([x_in, x_in if dec[-1 - layer] is None else dec[-1 - layer]], dim=1))
                enc[layer] = x_in
                x_in = self.encoders[e_i][1](x_in)
                e_i = e_i + 1
            x_in = self.middles(x_in)
            for layer in range(self.num_layers):
                x_in = self.decoders[d_i][0](cat([x_in, enc[-1 - layer]], dim=1))
                dec[layer] = x_in
                x_in = self.decoders[d_i][1](x_in)
                d_i = d_i + 1
            all_output[iteration] = x_in
        if self.integrate:
            x_in = cat(all_output, dim=1)
        x_in = self.post_transform_conv_block(x_in)
        return x_in
