import glob
import os
import sys
import urllib.request
import tarfile


PRETRAINED_CHECKPOINT_DIR = r".\out\checkpoints"
TRAIN_DIR = r".\out\custom-models"


def main():
    datasetDir = input("Enter path to your dataset : ")
    allImages = len(glob.glob(r"%s/*/*.jpg" % (datasetDir)))
    numCategory = countNumDir(glob.glob(r"%s/*" % (datasetDir)))

    trainPercentage = int(
        input("Specify train percentage (reminder test) : "))
    trainData = (allImages * trainPercentage)/100
    testData = (allImages - trainData)
    print("Your path to dataset is %s, contains %s classes" %
          (datasetDir, numCategory))
    print("Total %s images %d train %d test " %
          (allImages, trainData, testData))
    print("""Which model you want to train ?
    1.Inception V3
    2.Inception V4
    3.ResNet V1 152
    4.ResNet V2 152
    5.Inception-ResNet-v2
    6.VGG 19
    7.MobileNet v1 1.0_224""")
    trainModel = int(input("Enter number : "))
    modelData = addModelData(trainModel)
    numberOfStep = int(input("Please specify number of step : "))
    batchSize = int(input("Please specify batch size : "))
    print("You choose %s, %s steps of training, batch size is %s." %
          (modelData["modelName"], numberOfStep, batchSize))
    downloadModel(modelData)
    trainDir = getTrainDir(modelData, datasetDir)

    os.system("python download_and_convert_data.py \
	--dataset_name=custom \
	--dataset_dir=%s \
	--num_validation=%d \
	--num_shards=%d" % (datasetDir, testData, numCategory))

    os.system("python train_image_classifier.py \
    --train_dir=%s \
    --dataset_name=custom \
    --dataset_split_name=train \
    --dataset_dir=%s \
    --train_num=%d \
    --validation_num=%d \
    --num_of_classes=%d \
    --model_name=%s\
    --checkpoint_path=%s \
    --checkpoint_exclude_scopes=%s \
    --trainable_scopes=%s \
    --max_number_of_steps=%d\
    --batch_size=%s \
    --learning_rate=0.01 \
    --learning_rate_decay_type=fixed \
    --save_interval_secs=60 \
    --save_summaries_secs=60 \
    --log_every_n_steps=100 \
    --optimizer=rmsprop \
    --weight_decay=0.00004" % (trainDir, datasetDir, trainData,
                               testData, numCategory, modelData["modelName"],
                               (PRETRAINED_CHECKPOINT_DIR + "\\" +
                                modelData["modelFileName"][0].replace(".index", "")),
                               modelData["trainableScope"], modelData["trainableScope"], numberOfStep, batchSize))

    os.system("python eval_image_classifier.py \
    --checkpoint_path=%s \
    --eval_dir=%s \
    --dataset_name=custom \
    --train_num=%d \
    --validation_num=%d \
    --num_of_classes=%d \
    --dataset_split_name=validation \
    --dataset_dir=%s \
    --model_name=%s" % (trainDir, trainDir, trainData, testData, numCategory, datasetDir, modelData["modelName"]))


def countNumDir(globList):
    numCategory = 0
    for item in globList:
        if(os.path.isdir(item)):
            numCategory += 1
    return numCategory


def addModelData(trainModel):
    modelData = {}
    if(trainModel == 1):
        modelData["modelUrl"] = "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
        modelData["modelFileName"] = ["inception_v3.ckpt"]
        modelData["zipFileName"] = "inception_v3_2016_08_28.tar.gz"
        modelData["modelName"] = "inception_v3"
        modelData["trainableScope"] = "InceptionV3/Logits,InceptionV3/AuxLogits"
    if(trainModel == 2):
        modelData["modelUrl"] = "http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz"
        modelData["modelFileName"] = ["inception_v4.ckpt"]
        modelData["zipFileName"] = "inception_v4_2016_09_09.tar.gz"
        modelData["modelName"] = "inception_v4"
        modelData["trainableScope"] = "InceptionV4/Logits,InceptionV4/AuxLogits"
    if(trainModel == 3):
        modelData["modelUrl"] = "http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz"
        modelData["modelFileName"] = ["resnet_v1_152.ckpt"]
        modelData["zipFileName"] = "resnet_v1_152_2016_08_28.tar.gz"
        modelData["modelName"] = "resnet_v1_152"
        modelData["trainableScope"] = "resnet_v1_152/logits"
    if(trainModel == 4):
        modelData["modelUrl"] = "http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz"
        modelData["modelFileName"] = ["resnet_v2_152.ckpt"]
        modelData["zipFileName"] = "resnet_v2_152_2017_04_14.tar.gz"
        modelData["modelName"] = "resnet_v2_152"
        modelData["trainableScope"] = "resnet_v2_152/logits"
    if(trainModel == 5):
        modelData["modelUrl"] = "http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz"
        modelData["modelFileName"] = ["inception_resnet_v2_2016_08_30.ckpt"]
        modelData["zipFileName"] = "inception_resnet_v2_2016_08_30.tar.gz"
        modelData["modelName"] = "inception_resnet_v2"
        modelData["trainableScope"] = "InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits"
    if(trainModel == 6):
        modelData["modelUrl"] = "http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz"
        modelData["modelFileName"] = ["vgg_19.ckpt"]
        modelData["zipFileName"] = "vgg_19_2016_08_28.tar.gz"
        modelData["modelName"] = "vgg_19"
        modelData["trainableScope"] = "vgg_19/fc8"
    if(trainModel == 7):
        modelData["modelUrl"] = "http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz"
        modelData["modelFileName"] = ["mobilenet_v1_1.0_224.ckpt.index",
                                      "mobilenet_v1_1.0_224.ckpt.meta", "mobilenet_v1_1.0_224.ckpt.data-00000-of-00001"]
        modelData["zipFileName"] = "mobilenet_v1_1.0_224_2017_06_14.tar.gz"
        modelData["modelName"] = "mobilenet_v1"
        modelData["trainableScope"] = "MobilenetV1/Logits"
    return modelData


def downloadModel(modelData):
    if not (os.path.exists(PRETRAINED_CHECKPOINT_DIR)):
        os.makedirs(PRETRAINED_CHECKPOINT_DIR)
    if not (os.path.exists(r"%s\%s" % (PRETRAINED_CHECKPOINT_DIR, modelData["modelFileName"]))):
        isPretrainModelExist = checkPretrainModelExist(modelData)
        if not isPretrainModelExist:
            urllib.request.urlretrieve(
                modelData["modelUrl"], modelData["zipFileName"], reporthook)
            tar = tarfile.open(modelData["zipFileName"])
            tar.extractall()
            tar.close()
            for item in modelData["modelFileName"]:
                os.rename(item, r"%s\%s" % (PRETRAINED_CHECKPOINT_DIR, item))
            os.remove(modelData["zipFileName"])
        else:
            print("Pretrain model file already exist. Exist without re-downloading them.")


def getTrainDir(modelData, datasetDir):
    trainDir = r"%s\%s_%s" % (
        TRAIN_DIR, modelData["modelName"], datasetDir.split("\\")[-1])
    if not (os.path.exists(trainDir)):
        os.makedirs(trainDir)
    return trainDir


def checkPretrainModelExist(modelData):
    isPretrainModelExist = False
    for item in modelData["modelFileName"]:
        if(os.path.exists(r"%s\%s" % (PRETRAINED_CHECKPOINT_DIR, item))):
            isPretrainModelExist = True
    return isPretrainModelExist


def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r Downloading %5.1f%% %*.1f MB / %.1f MB" % (
            percent, len(str(totalsize)), (readsofar/1048576), (totalsize/1048576))
        sys.stderr.write(s)
        if readsofar >= totalsize:
            sys.stderr.write("\n")
    else:
        sys.stderr.write("read %d\n" % (readsofar,))


main()
