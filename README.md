# slim_simplified
Simple interface to fine-tune Tensorflow model using slim. This repository is a fork form [Tensorflow slim][Tensorflow slim]. The purpose is for making it more simple and more easy to use.

This repository some of the code from [Tensorflow slim][Tensorflow slim] and a modify code from the author especially Shell script and Python script.

### Run a file

In order to run a code to finetune or transfer learning with CNN models, simply just run **finetune_custom_dataset.sh** in a **scripts** folder.

```Shell
sh ./scripts/finetune_custom_dataset.sh
```

 This script are support 7 CNN models, there are :
 * InceptionV3
 * InceptionV4
 * Resnet Inception
 * ResnetV1
 * ResnetV2
 * VGG16
 * MobileNet


### Prerequisite to train a model with transfer-CNN

* [Tensorflow][Tensorflow website] must be installed.
* This code contains Shell script, UNIX-like operating system (Linux distros, OSX, BSD, etc..) is required to run a Shell scripts.
* GPU contains a more than 4 gigabytes of memory is recommend for a faster performance.

[Tensorflow slim]: https://github.com/tensorflow/models/tree/master/research/slim
[Tensorflow website]: https://www.tensorflow.org/
