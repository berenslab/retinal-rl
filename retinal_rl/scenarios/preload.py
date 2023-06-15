import os
import shutil

from num2words import num2words
import os.path as osp

from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100


### Loading Datasets ###


def preload_apples():
    # check if scenarios/resources/textures/apples exists
    if not osp.exists("scenarios/resources/textures/apples"):
        # if not, create it
        os.makedirs("scenarios/resources/textures/apples")
        # download apple images
        rappth = "scenarios/resources/base/apples/red_apple.png"
        bappth = "scenarios/resources/base/apples/blue_apple.png"

        # copy red apple to scenarios/resources/textures/apples/nourishment-{1-5} and blue apples to poisson-{1-5}
        for i in range(1,6):
            os.makedirs("scenarios/resources/textures/apples/nourishment-" + str(i))
            shutil.copyfile(rappth, "scenarios/resources/textures/apples/nourishment-" + str(i) + "/apple.png")
            os.makedirs("scenarios/resources/textures/apples/poison-" + str(i))
            shutil.copyfile(bappth, "scenarios/resources/textures/apples/poison-" + str(i) + "/apple.png")

def preload_mnist():
    # check if scenarios/resources/textures/mnist exists
    if not osp.exists("scenarios/resources/textures/mnist"):
    # if not, create it
        os.makedirs("scenarios/resources/textures/mnist")
    # download mnist images
        mnist = MNIST("scenarios/resources/textures/mnist", download=True)
    # save mnist images as pngs organized by word label
        for i in range(10):
            os.makedirs("scenarios/resources/textures/mnist/" + num2words(i))
        for i in range(len(mnist)):
            mnist[i][0].save("scenarios/resources/textures/mnist/" + num2words(mnist[i][1]) + "/" + str(i) + ".png")

        # remove all downloaded data except for the pngs
        shutil.rmtree("scenarios/resources/textures/mnist/MNIST", ignore_errors=True)

    else:
        print("mnist dir exists, files not downloaded)")

def preload_cifar10():
    # check if scenarios/resources/textures/cifar-10 exists
    if not osp.exists("scenarios/resources/textures/cifar-10"):
        # if not, create it
        os.makedirs("scenarios/resources/textures/cifar-10")
        # download cifar images
        cifar = CIFAR10("scenarios/resources/textures/cifar-10", download=True)
        # save cifar images as pngs organized by label name
        for i in range(10):
            os.makedirs("scenarios/resources/textures/cifar-10/" + cifar.classes[i])
        for i in range(len(cifar)):
            cifar[i][0].save("scenarios/resources/textures/cifar-10/" + cifar.classes[cifar[i][1]] + "/" + str(i) + ".png")

        # remove all downloaded data except for the pngs
        os.remove("scenarios/resources/textures/cifar-10/cifar-10-python.tar.gz")
        shutil.rmtree("scenarios/resources/textures/cifar-10/cifar-10-batches-py", ignore_errors=True)
    else:
        print("cifar-10 dir exists, files not downloaded")

def preload_cifar100():
    # check if scenarios/resources/textures/cifar-100 exists
    if not osp.exists("scenarios/resources/textures/cifar-100"):
        # if not, create it
        os.makedirs("scenarios/resources/textures/cifar-100")
        # download cifar images
        cifar = CIFAR100("scenarios/resources/textures/cifar-100", download=True)
        # save cifar images as pngs organized by label name
        for i in range(100):
            os.makedirs("scenarios/resources/textures/cifar-100/" + cifar.classes[i])
        for i in range(len(cifar)):
            cifar[i][0].save("scenarios/resources/textures/cifar-100/" + cifar.classes[cifar[i][1]] + "/" + str(i) + ".png")

        # remove all downloaded data except for the pngs
        os.remove("scenarios/resources/textures/cifar-100/cifar-100-python.tar.gz")
        shutil.rmtree("scenarios/resources/textures/cifar-100/cifar-100-python", ignore_errors=True)
    else:
        print("cifar-100 dir exists, files not downloaded")
