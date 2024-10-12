import torch
import torch.nn as nn
from torchvision import transforms,io
import numpy as np

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InvertibleBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.conv1=nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.activation=nn.ReLU()

    def forward(self,x1,x2):
        y1=x1+self.conv1(self.activation(x2))
        y2=x2+self.conv2(self.activation(y1))
        return y1,y2

    def inverse(self,y1,y2):
        x2=y2 - self.conv2(self.activation(y1))
        x1=y1 - self.conv1(self.activation(x2))
        return x1,x2

class InvertibleNN(nn.Module):
    def __init__(self,channels,num_blocks=2):
        super().__init__()
        self.blocks=nn.ModuleList([InvertibleBlock(channels) for _ in range(num_blocks)])

    def forward(self,x1,x2):
        for block in self.blocks:
            x1,x2=block(x1,x2)
        return torch.sigmoid(x1),torch.sigmoid(x2)

    def inverse(self,y1,y2):
        for block in reversed(self.blocks):
            y1,y2=block.inverse(y1,y2)
        return torch.sigmoid(y1),torch.sigmoid(y2)

class INN():
    def __init__(self,size:tuple=(3,1080,1920),path:str="model.pth",key:torch.tensor=None):
        self.model=InvertibleNN(channels=3,num_blocks=2)
        self.model.to(device)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

        if key==None:
            key=torch.zeros(size)+0.5
        self.key=key.to(device)
        pass

    def Encoder(self,x1,x2):
        y1,y2=self.model(x1,x2)
        return y1

    def Decoder(self,y1):
        target_y2=self.key
        x1,x2=self.model.inverse(y1,target_y2)
        return x2

def loader(video_path,batch_size:int=128,output_size:tuple=(1920,1080)):
    h,w=output_size
    transform=transforms.Compose([
        transforms.Resize((h,w)),
        transforms.CenterCrop((h,w))
    ])
    frames,_,_=io.read_video(video_path).to(device)
    frames=frames.permute(0,3,1,2)#[B,C,H,W]
    frames=transform(frames)
    num_batches=frames.size(0) // batch_size
    batches=[]
    for i in range(num_batches):
        batch=frames[i * batch_size:(i+1) * batch_size]
        batches.append(batch)
    return batches

def encode_audio(audio: np.ndarray,frame: np.ndarray,redundant: int=5):
    """
    将音频片段嵌入到视频帧的底部，使用空间冗余策略，既将包含音频信息的行重复多次。
    
    参数：
    - audio: 音频片段，dtype=int16，形状为[N,]。
    - frame: 视频帧，dtype=np.uint8，形状为[height,width,channels]。
    - redundant: 嵌入的行数（默认为3行）。
    
    返回：
    - frame_with_audio: 嵌入音频的帧。
    """
    # 获取frame的宽度、高度和通道数
    height,width,channels=frame.shape
    max_size=width * channels
    
    # 检查audio的长度是否超过可以嵌入的长度
    if len(audio) > max_size:
        # 如果audio的长度超过可用空间，截取前max_size个样本
        audio=audio[:max_size]
    else:
        # 如果audio的长度不足，用零填充到所需长度
        audio=np.pad(audio,(0,max(0,max_size - len(audio))),mode='constant')
    audio=audio.reshape(width,channels)#盘成[1920,3]
    
    # 将int16的音频样本值转换为uint8
    audio_uint8=(audio//256+128).astype(np.uint8)
    
    # 将音频样本值嵌入到frame的底部
    frame[height - redundant:height,:,:]=np.tile(audio_uint8,(redundant,1,1))

    return frame

def decode_audio(frame: np.ndarray,length:int=3200,redundant: int=5) -> np.ndarray:
    """
    从视频帧的底部解码音频信息，使用空间冗余策略（多行取均值）。
    
    参数：
    - frame: 视频帧，dtype=np.uint8，形状为[height,width,channels]。
    - redundant: 冗余行数，表示嵌入音频信息的底部重复行数（默认为3行）。
    
    返回：
    - audio: 解码出的音频片段，dtype=int16。
    """
    # 获取 frame 的宽度、高度和通道数
    height,width,channels=frame.shape
    
    # 提取底部的冗余行：直接切片，取出倒数 redundant 行
    audio_data=frame[height - redundant:height,:,:]
    
    # 计算冗余行的均值，指定在第一个维度（行）上取均值
    audio_mean=np.mean(audio_data,axis=0)

    # 展平数据为一维音频
    audio_mean=audio_mean.flatten()
    
    # 将 uint8 的音频数据转换回 int16
    audio_int16=((audio_mean - 128) * 256).astype(np.int16)
    
    return audio_int16[:length]