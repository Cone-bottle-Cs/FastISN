import matplotlib.pyplot as plt
import cv2
import torch
from torchvision import transforms
from time import time
import numpy as np
from pydub import AudioSegment
import subprocess
from util import INN,encode_audio

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化视频捕捉
video1=cv2.VideoCapture('Stellaris.mp4')
video2=cv2.VideoCapture('secret.mp4')
inn=INN(path="model.pth")

# 获取视频信息
target_fps=30
size=(1920, 1080)
fourcc=cv2.VideoWriter_fourcc(*'mp4v')
output=cv2.VideoWriter('tmp.mp4', fourcc, target_fps, size)

# 读取音频
audioS=AudioSegment.from_file('secret.mp4')
frame_duration_ms=1000 / target_fps
audio_segments=[]

success1, frame1=video1.read()
success2, frame2=video2.read()
T=[]
frame_count=0

while success1 and success2:
    # 调整帧大小
    frame1=cv2.resize(frame1, size)
    frame1=cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2=cv2.resize(frame2, size)
    frame2=cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    # 转换为tensor并处理
    frame1_tensor=transforms.ToTensor()(frame1).to(device)
    frame2_tensor=transforms.ToTensor()(frame2).to(device)
    
    t=time()
    y1=inn.Encoder(frame1_tensor, frame2_tensor)
    y1=y1.permute(1, 2, 0).cpu().detach().numpy()
    y1=(y1 * 255.0).astype(np.uint8)
    T.append(1 / (time() - t))
    
    y1=cv2.cvtColor(y1, cv2.COLOR_RGB2BGR)

    # 处理对应的音频
    start_time=frame_count * frame_duration_ms
    end_time=start_time + frame_duration_ms
    audio_segment=audioS[start_time:end_time]
    audio=np.array(audio_segment.get_array_of_samples())
    y1=encode_audio(audio,y1)
    frame_count += 1
    output.write(y1)

    # 读取下一帧
    success1, frame1=video1.read()
    success2, frame2=video2.read()

# 释放资源
video1.release()
video2.release()
output.release()

# 绘制FPS图
plt.plot(T)
plt.savefig("encodeFPS.png")
print(f"average FPS: {sum(T)/len(T):.2f}")
print(f"audio length: {audio.shape[0]}")