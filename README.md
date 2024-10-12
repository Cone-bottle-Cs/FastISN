# FastISN
An inversible neural network for video steganographic which only contains 4 convolution layers.

## Performance
more than 30fps on RTX3060, but the performance of video after decoding is worse than other neural network steganography method.  
haven't many change after JPEG compressing (could not worse more, I would say)

## Description of the file
**train.py** Inspired by the aricle of [ISN](https://www.shaopinglu.net/index.files/CVPR21__Image_Steganography.pdf), we consider LIPIS as loss function and Adamax as optimizer. To accelerate the training process, we utilized CIFAR10 as dataset with batch_size=128, and substitute JPEG compressed process by GaussianBlur in training process. From the experiment we did, the model can still have good performance in larger image after such subsititution.
**encoedr.py and decoedr.py** Simply for encode and decode the secret video and cover video using the *model.pth*, but the encoder.py still have some bugs.

## Further work
OBS puglin
