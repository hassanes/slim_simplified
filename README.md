# slim_simplified

Simple interface to fine-tune Tensorflow model using slim. This repository is a fork form [Tensorflow slim][tensorflow slim]. The purpose is for making it more simple and more easy to use.

This repository some of the code from [Tensorflow slim][tensorflow slim] and a modify code from the author.

### Run a file

In order to run a code to finetune or transfer learning with CNN models, simply just run `finetune_custom_dataset.py` in a `scripts` folder, then follow the instruction on the screen.

```Shell
python ./scripts/finetune_custom_dataset.py
```

This script is support 7 CNN models, there are :

- InceptionV3
- InceptionV4
- Resnet Inception
- ResnetV1
- ResnetV2
- VGG16
- MobileNet

### Prerequisite to train a model with transfer-CNN

- Python 3.6
- Install all python package requirement in requirements.txt using pip.

```Shell
pip install -r requirements.txt
```

- Nvidia CUDA toolkit version 9.0, see [CUDA Toolkit Archive][cuda toolkit archive].
- Nvidia cuDNN SDK version >= 7.2, see [cuDNN Download Page][cudnn download].
- GPU contains a more than 4 gigabytes of memory is recommend for a faster performance.

### Todos

- [ ] Freeze model graph for application usage.
- [ ] NasNet model.

[tensorflow slim]: https://github.com/tensorflow/models/tree/master/research/slim
[tensorflow website]: https://www.tensorflow.org/
[cuda toolkit archive]: https://developer.nvidia.com/cuda-toolkit-archive
[cudnn download]: https://developer.nvidia.com/cudnn
