# FastISN
An invertible neural network for video steganography, consisting of only 4 convolutional layers.

## Performance
Achieves more than 30fps on an RTX 3060. However, the decoded video's quality is inferior compared to other neural network-based steganography methods.  
JPEG compression does not significantly degrade performance (as it couldn't get much worse, frankly speaking).

## File Descriptions
- **train.py**: Inspired by the paper [ISN](https://www.shaopinglu.net/index.files/CVPR21__Image_Steganography.pdf), we use [LIPIS](https://pypi.org/project/lpips/) as loss function and Adamax as optimizer. To accelerate the training process, CIFAR10 is used as the dataset with `batch_size=128`, and we replace the JPEG compression process with Gaussian blur during training. From our experiments, this substitution still achieves similar performance on larger images.  
- **encoedr.py and decoedr.py**: These scripts encode and decode secret and cover videos using the **model.pth** and **ffmpeg** commands (so ffmpeg must be installed in your environment).
- **how to use**:  
Run the command `python encoder.py --cover cover.mp4 --secret secret.mp4` to generate a steganographic video.  
Run the command `python decoder.py --source Encode.mp4` to decode the video.  
For more details on available parameters, refer to the code.

## Further work
Quantization and OBS puglin