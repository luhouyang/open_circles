"""
CNN Segmentation Models
LeNet-1
AlexNet
VGG16
ResNet

author: Lu Hou Yang
GitHub: https://github.com/luhouyang/open_circles.git
date: 23 March 2025
"""

# NOTE: To ensure a comparison that is accurate to the timeline and
#       techonlogy/technique available there are some constrains
# 1. The contracting (down, feature extraction) module will have scaled parameter count (similar to U-Net)
# 2. The expansive (up, segmentation) module will follow the U-Net architecture

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# Reading LeNet-1 (paper): https://scholar.google.com/citations?view_op=view_citation&hl=en&user=WLN3QrAAAAAJ&citation_for_view=WLN3QrAAAAAJ:u-x6o8ySG0sC
# Original: 5,126 | U-Net: 137,256 | Modified: 157,762
class LeNet1(nn.Module):
    # yapf: disable
    def __init__(self, in_channels, num_classes):
        super(LeNet1, self).__init__()

        # Reading Tanh: https://towardsdatascience.com/activation-functions-in-neural-networks-how-to-choose-the-right-one-cb20414c04e5/
        self.tanh = nn.Tanh()

        # Original
        # Conv2d    (in, 12, 5, padding='same', bias=True)
        # Conv2d    (12, 12, 5, padding='same', bias=True)
        # Conv2d    (12, 12, 5, padding='same', bias=True)
        # Conv2d    (12, num_classes, 5, padding='same', bias=True)

        self.down_conv1 = nn.Conv2d(in_channels, 32, 5, padding='same', bias=True)
        self.down_conv2 = nn.Conv2d(32, 96, 5, padding='same', bias=True)

        self.up_conv1 = nn.Conv2d(96, 32, 5, padding='same', bias=True)
        self.up_conv2 = nn.Conv2d(32, num_classes, 5, padding='same', bias=True)

    def forward(self, x):
        x = self.tanh(self.down_conv1(x))
        x = self.tanh(self.down_conv2(x))
        x = self.tanh(self.up_conv1(x))
        x = self.tanh(self.up_conv2(x))

        return x
    # yapf: enable


def lenet1_weight_initializer(model):
    prev = 1

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            num_weights = len(layer.weight)
            fan_in = num_weights * prev
            std = (24.0 / fan_in) / 2.0
            print(fan_in)

            init.normal_(layer.weight, mean=0, std=std)
            if layer.bias is not None:
                init.zeros_(layer.bias)

            prev = num_weights


# yapf: disable
# Reading AlexNet (paper): https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwie-_vy_d6LAxW_zDgGHQIBO9gQFnoECAgQAQ&url=https%3A%2F%2Fproceedings.neurips.cc%2Fpaper%2F4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf&usg=AOvVaw26V5YkBm0FS972qI4eBNgu&opi=89978449
# Original: 60,000,000 | U-Net: 137,256 | Modified: 271,846
# CNN with ReLU, Overlapping MaxPooling, Dropout
class LocalResponseNormalization(nn.Module):

    def __init__(self, k=2, n=5, alpha=10e-4, beta=0.75):
        super(LocalResponseNormalization, self).__init__()

        self.k = k
        self.n = n
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        norm_factor = torch.pow(self.k + self.alpha * (torch.sum(torch.pow(x, 2))), self.beta)
        b = x / norm_factor
        return b


class AlexNet(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(AlexNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout1d(0.5)
        self.lr_norm = LocalResponseNormalization()

        # Original
        # Conv2d    (in, 48, 11, stride=4, padding=2, bias=True)    * 2
        # Local Response Normalization | k=2, n=5, α=10e-4, β=0.75
        # MaxPool2d (3, stride=2)
        # ReLU
        # Conv2d    (96, 128, 5, padding=1, bias=True)              * 2
        # Local Response Normalization | k=2, n=5, α=10e-4, β=0.75
        # MaxPool2d (3, stride=2)
        # ReLU
        # Conv2d    (256, 192, 3, padding=1, bias-True)             * 2
        # Conv2d    (192, 192, 3, padding=1, bias=True)             * 2

        # self.conv1_1 = nn.Conv2d(in_channels, 48,  11, stride=4, padding=2, bias=True)
        # self.conv1_2 = nn.Conv2d(in_channels, 48,  11, stride=4, padding=2, bias=True)

        # self.conv2_1 = nn.Conv2d(48, 128, 5, padding=2, bias=True)
        # self.conv2_2 = nn.Conv2d(48, 128, 5, padding=2, bias=True)

        # # accept input from both layer 2
        # self.conv3_1 = nn.Conv2d(256, 192, 3, padding=1, bias=True)
        # self.conv3_2 = nn.Conv2d(256, 192, 3, padding=1, bias=True)

        # self.conv4_1 = nn.Conv2d(192, 192, 3, padding=1, bias=True)
        # self.conv4_2 = nn.Conv2d(192, 192, 3, padding=1, bias=True)

        # self.conv5_1 = nn.Conv2d(192, 128, 3, padding=1, bias=True)
        # self.conv5_2 = nn.Conv2d(192, 128, 3, padding=1, bias=True)

        # self.fc1 = nn.Linear(6*6*256, 4096, bias=True)
        # self.fc2 = nn.Linear(4096, 4096, bias=True)
        # self.out = nn.Linear(4096, 1000, bias=True)

        self.conv1_1 = nn.Conv2d(in_channels, 12,  11, stride=4, padding=2, bias=True)
        self.conv1_2 = nn.Conv2d(in_channels, 12,  11, stride=4, padding=2, bias=True)

        self.conv2_1 = nn.Conv2d(12, 32, 5, padding=2, bias=True)
        self.conv2_2 = nn.Conv2d(12, 32, 5, padding=2, bias=True)

        # accept input from both layer 2
        self.conv3_1 = nn.Conv2d(64, 48, 3, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(64, 48, 3, padding=1, bias=True)

        self.conv4_1 = nn.Conv2d(48, 48, 3, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(48, 48, 3, padding=1, bias=True)

        self.conv5_1 = nn.Conv2d(48, 32, 3, padding=1, bias=True)
        self.conv5_2 = nn.Conv2d(48, 32, 3, padding=1, bias=True)

        # upsample
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)

        self.conv6_1 = nn.Conv2d(32, 48, 3, padding=1, bias=True)
        self.conv6_2 = nn.Conv2d(32, 48, 3, padding=1, bias=True)

        self.conv7_1 = nn.Conv2d(48, 48, 3, padding=1, bias=True)
        self.conv7_2 = nn.Conv2d(48, 48, 3, padding=1, bias=True)

        # chunk and split to 2 lower layers
        self.conv8_1 = nn.Conv2d(48, 32, 3, padding=1, bias=True)
        self.conv8_2 = nn.Conv2d(48, 32, 3, padding=1, bias=True)

        self.dropout = nn.Dropout2d(0.5)

        self.conv9 = nn.Conv2d(64, 32, 3, padding=1, bias=True)

        self.conv10 = nn.Conv2d(32, 12, 3, padding=1, bias=True)

        self.out = nn.Conv2d(12, num_classes, 3, padding=1, bias=True)

    def forward(self, x):
        # # split 1
        # x1 = self.relu(self.maxpool(self.lr_norm(self.conv1_1(x))))
        # x2 = self.relu(self.maxpool(self.lr_norm(self.conv1_2(x))))

        # x1 = self.relu(self.maxpool(self.lr_norm(self.conv2_1(x1))))
        # x2 = self.relu(self.maxpool(self.lr_norm(self.conv2_2(x2))))

        # # merge, then pass into 2 different 3rd layers
        # x = torch.cat([x1, x2], dim=1)
        # x1 = self.relu(self.conv3_1(x))
        # x2 = self.relu(self.conv3_2(x))

        # # split 2
        # x1 = self.relu(self.conv4_1(x1))
        # x2 = self.relu(self.conv4_2(x2))

        # x1 = self.relu(self.maxpool(self.lr_norm(self.conv5_1(x1))))
        # x2 = self.relu(self.maxpool(self.lr_norm(self.conv5_2(x2))))

        # # merge and flatten
        # x = torch.cat([x1, x2], dim=1)
        # x = self.flatten(x)

        # # fully connected layer
        # x = self.dropout(self.relu(self.fc1(x)))
        # x = self.dropout(self.relu(self.fc2(x)))
        # x = self.relu(self.out(x))

        # split 1
        x1 = self.relu(self.maxpool(self.lr_norm(self.conv1_1(x))))
        x2 = self.relu(self.maxpool(self.lr_norm(self.conv1_2(x))))

        x1 = self.relu(self.maxpool(self.lr_norm(self.conv2_1(x1))))
        x2 = self.relu(self.maxpool(self.lr_norm(self.conv2_2(x2))))

        # merge, then pass into 2 different 3rd layers
        x = torch.cat([x1, x2], dim=1)
        x1 = self.relu(self.conv3_1(x))
        x2 = self.relu(self.conv3_2(x))

        # split 2
        x1 = self.relu(self.conv4_1(x1))
        x2 = self.relu(self.conv4_2(x2))

        x1 = self.relu(self.maxpool(self.lr_norm(self.conv5_1(x1))))
        x2 = self.relu(self.maxpool(self.lr_norm(self.conv5_2(x2))))

        # upsample
        x1 = self.relu(self.upsample4(self.lr_norm(self.conv6_1(x1))))
        x2 = self.relu(self.upsample4(self.lr_norm(self.conv6_2(x2))))

        x1 = self.relu(self.upsample4(self.conv7_1(x1)))
        x2 = self.relu(self.upsample4(self.conv7_2(x2)))

        # forward pass then merge
        x1 = self.relu(self.upsample2(self.conv8_1(x1)))
        x2 = self.relu(self.upsample2(self.conv8_2(x2)))
        x = torch.cat([x1, x2], dim=1)

        x = self.dropout(self.relu(self.upsample(self.conv9(x))))
        x = self.dropout(self.relu(self.conv10(x)))
        x = self.out(x)
        return x
# yapf: enable


# We initialized the weights in each layer from a zero-mean Gaussian distribution with standard deviation 0.01.
# We initialized the neuron biases in the second, fourth, and fifth convolutional layers,
# as well as in the fully-connected hidden layers, with the constant 1. This initialization accelerates
# the early stages of learning by providing the ReLUs with positive inputs. We initialized the neuron
# biases in the remaining layers with the constant 0.
def alexnet_weight_initializer(model):
    one_bias_layers = [2, 4, 5]
    layer_number = 1

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            init.normal_(layer.weight, mean=0, std=0.01)
            if layer.bias is not None:
                if layer_number in one_bias_layers:
                    init.ones_(layer.bias)
                else:
                    init.zeros_(layer.bias)

                layer_number += 1


# Reading VGG (paper): https://arxiv.org/abs/1409.1556
# Original (C): 133,638,952 | U-Net: 137,256 | Modified: 299,418
class VGG16(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(VGG16, self).__init__()

        # Original
        # Conv2d    (in, 64, 3, padding=1)
        # Conv2d    (64, 64, 3, padding=1)
        # MaxPool2d (2)  224 -> 112

        # Conv2d    (64, 128, 3, padding=1)
        # Conv2d    (128, 128, 3, padding=1)
        # MaxPool2d (2)  112 -> 56

        # Conv2d    (128, 256, 3, padding=1)
        # Conv2d    (256, 256, 3, padding=1)
        # Conv2d    (256, 256, 1, padding=0)
        # MaxPool2d (2)  56 -> 28

        # Conv2d    (256, 512, 3, padding=1)
        # Conv2d    (512, 512, 3, padding=1)
        # Conv2d    (512, 512, 1, padding=0)
        # MaxPool2d (2)  28 -> 14

        # Conv2d    (512, 512, 3, padding=1)
        # Conv2d    (512, 512, 3, padding=1)
        # Conv2d    (512, 512, 1, padding=0)
        # MaxPool2d (2)  14 -> 7

        # Flatten
        # Linear    (7*7*512, 4096)
        # Dropout   (0.5)
        # Linear    (4096, 4096)
        # Dropout   (0.5)
        # Linear    (4096, 1000)
        # SoftMax

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        # )

        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(64, 128, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 128, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        # )

        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(128, 256, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 1, padding=0),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        # )

        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(256, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, 1, padding=0),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        # )

        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, 1, padding=0),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        # )

        # self.flatten = nn.Flatten()

        # self.fc = nn.Sequential(
        #     nn.Linear(7 * 7 * 512, 4096),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(4096, 1000),
        #     nn.Softmax(),
        # )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 56, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(56, 56, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(56, 56, 1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(56, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='bilinear',
                                    align_corners=True)

        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 56, 1, padding=0),
            nn.ReLU(),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(56, 56, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(56, 56, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(56, 32, 1, padding=0),
            nn.ReLU(),
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 1, padding=0),
            nn.ReLU(),
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.5),
        )

        self.conv10 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, num_classes, 3, padding=1),
        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.flatten(x)
        # x = self.fc(x)
        # return x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.conv6(self.upsample(x))
        x = self.conv7(self.upsample(x))
        x = self.conv8(self.upsample(x))
        x = self.conv9(self.upsample(x))
        x = self.conv10(self.upsample(x))
        return x


# Reading Xavier Glorot & Yoshua Bengio | Weight Initialization (paper): https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
def vgg16_weight_initializer(model):
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            gain = init.calculate_gain('relu')

            init.xavier_uniform_(layer.weight, gain=gain)

            if layer.bias is not None:
                init.zeros_(layer.bias)

        if isinstance(layer, nn.ReLU):
            layer.inplace = True


# Reading ResNet (paper): https://arxiv.org/abs/1512.03385
# Original: | U-Net: 137,256 | Modified:
class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def forward(self, x):
        pass


def resnet_weight_initialize(model):
    pass


import matplotlib.pyplot as plt


def visualize_flattened_tensor(
    tensor: torch.Tensor,
    title: str = "Flattened Tensor Visualization",
    bins: int = 50,
):
    """
    Visualizes a flattened PyTorch tensor.

    This function is AI generated. 

    Args:
        tensor: A PyTorch tensor.
        title: The title of the plot.
    """
    # Flatten the tensor
    flattened_tensor = tensor.flatten().detach().numpy()

    # Create a histogram
    plt.figure(figsize=(10, 5))
    plt.hist(
        flattened_tensor, bins=bins,
        edgecolor='black')  # Added edgecolor for better separation of bars
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(axis='y',
             alpha=0.75)  # Added a y-axis grid for better readability
    plt.show()


def visualize_initialized_weights(model):
    num_plots = 0
    for i, layer in enumerate(model.modules()):
        if i != 0 and (isinstance(layer, nn.Conv2d)
                       or isinstance(layer, nn.Linear)):
            num_plots += 1

    import math
    plot_w = 4
    plot_h = math.ceil(num_plots / plot_w)

    fig, axes = plt.subplots(plot_h, plot_w, figsize=(plot_w * 4, plot_h * 3))
    axes = np.atleast_2d(axes)

    # Plot histogram for the weights of a specific layer
    def plot_weights_histogram(layer_weights, layer_name, ax):
        ax.hist(layer_weights.cpu().detach().numpy().flatten(), bins=50)
        ax.set_title(f"Weight Distribution for {layer_name}")

    layer_number = 0
    for i, layer in enumerate(model.modules()):
        if i != 0 and (isinstance(layer, nn.Conv2d)
                       or isinstance(layer, nn.Linear)):
            plot_weights_histogram(
                layer.weight,
                f"{layer._get_name()} | {layer_number + 1}",
                axes[(layer_number // plot_w), (layer_number % plot_w)],
            )
            layer_number += 1

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    ### a bit about weigh initializations
    ### Reading (Medium): https://medium.com/data-scientists-diary/how-to-initialize-weights-in-pytorch-e912308459d4

    sample_input = torch.randn([1, 3, 224, 224])
    # sample_input = torch.randint(-100, 100, [1, 3, 224, 224])

    # model = LeNet1(in_channels=3, num_classes=2)
    # model.apply(lenet1_weight_initializer)

    # model = AlexNet(in_channels=3, num_classes=2)
    # model.apply(alexnet_weight_initializer)
    # model = LocalResponseNormalization()

    # model = VGG16(in_channels=3, num_classes=2)
    # model.apply(vgg16_weight_initializer)

    model = ResNet()
    model.apply(resnet_weight_initialize)

    output = model(sample_input)

    print(output.shape)
    visualize_flattened_tensor(sample_input)
    visualize_flattened_tensor(output)

    import torchinfo
    torchinfo.summary(model)

    visualize_initialized_weights(model)
