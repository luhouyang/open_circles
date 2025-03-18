# Open Circles

Open Circles : Bridging Minds Beyond Borders is an international university collaboration connecting students globally through IT and computer science. This immersive experience features workshops, talks, and activities that inspire, educate, and foster cross-cultural connections, creating a global network of future tech leaders.

## OC Archives

*Add all past OC projects here, swap the following h2 tag with latest OC documentation*

- [OC 1.0](/OC01/)

- [OC 2.0](/)

## OC 1.0 - Computer Vision from Convolutional to Transformers

**Setup & Run**

1. Clone this repo

    ```console
    $ git clone --depth 1 https://github.com/luhouyang/open_circles.git
    ```

1. Create a Python environment (replace with your own directory)

    ```console
    $ python3.12 -m venv D:\User\test
    ```

1. Change your Python interpreter (steps if using VS Code). At the search bar type `>Python: Select Interpreter` -> select `Enter interpreter path` -> paste the path used to create env *i.e.* `D:\User\test`

1. Restart your IDE

1. Activate your environment at the terminal (replace with your own directory)

    ```console
    $ D:/User/test/Scripts/activate.bat
    ```

    You should see `(test) C:\Users\...\open_circles>` in the terminal

1. Install packages (for pytorch please change according to your device capabilities)

    **IMPORTANT! BEFORE INSTALLING CHECK THAT YOUR ENVIRNOMENT IS ACTIVE i.e. (your_env_name) C:\\...**

    ```console
    cd OC01
    pip install -r requirements.txt
    ```

    *alternatively use pip to auto resolve the packages*

    ```
    pip install polars pillow pycocotools
    ```

    Check the [PyTorch Website](https://pytorch.org)

    ```console
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ```

**Reading Materials**

![History of CNN Timeline](/OC01/media/history_of_CNN.png)

- [Hisory of CNN](https://towardsdatascience.com/the-history-of-convolutional-neural-networks-for-image-classification-1989-today-5ea8a5c5fe20/)

- LeNet-1 | [Original CNN Paper](https://www.academia.edu/download/47948178/lecun-89e.pdf)

- LeNet-5 | [paper](https://www.researchgate.net/publication/2985446_Gradient-Based_Learning_Applied_to_Document_Recognition)

- AlexNet | [paper](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwie-_vy_d6LAxW_zDgGHQIBO9gQFnoECAgQAQ&url=https%3A%2F%2Fproceedings.neurips.cc%2Fpaper%2F4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf&usg=AOvVaw26V5YkBm0FS972qI4eBNgu&opi=89978449) | [PyTorch](https://pytorch.org/hub/pytorch_vision_alexnet/)

- GoogLeNet | [paper](https://arxiv.org/abs/1409.4842) | [PyTorch](https://pytorch.org/hub/pytorch_vision_googlenet/)

- VGGNet | [paper](https://arxiv.org/abs/1409.1556) | [PyTorch](https://pytorch.org/hub/pytorch_vision_vgg/)

- ResNet | [paper](https://arxiv.org/abs/1512.03385) | [PyTorch](https://pytorch.org/hub/pytorch_vision_resnet/)

- DenseNet | [paper](https://arxiv.org/abs/1608.06993) | [PyTorch](https://pytorch.org/hub/pytorch_vision_densenet/)

- MobileNetV1 | [paper](https://arxiv.org/abs/1704.04861) | MobileNetV2 | [paper](https://arxiv.org/abs/1801.04381) | [PyTorch](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/) | MobileNetV3 | [paper](https://arxiv.org/abs/1905.02244) | [PyTorch](https://pytorch.org/vision/main/models/mobilenetv3.html) | MobileNetV4 | [paper](https://arxiv.org/abs/2404.10518) | [Hugging Face](https://huggingface.co/collections/timm/mobilenetv4-pretrained-weights-6669c22cda4db4244def9637)

- ConvNeXt | [paper](https://arxiv.org/abs/2201.03545) | [PyTorch](https://pytorch.org/vision/main/models/convnext.html)

## Future Additions TODO

- Add documentation for Linux, MacOS, etc. platforms. (job for future maintainers)
- Move setup documentation to respective folders