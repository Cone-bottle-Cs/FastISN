import matplotlib.pyplot as plt
import cv2
import torch
from torchvision import transforms
from time import time
import numpy as np
from pydub import AudioSegment
import subprocess
from util import INN,decode_audio

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#获得视频的格式
video1=cv2.VideoCapture('BBdown.mp4')
decodeVideoPath="BBdecode.mp4"
inn=INN(path="model.pth")

#获得码率及尺寸
fps=30
size=(1920,1080)
fNUMS1=video1.get(cv2.CAP_PROP_FRAME_COUNT)#这个应该是总帧长，两分钟30帧应该就是3600帧
fourcc=cv2.VideoWriter_fourcc(*'mp4v')
output=cv2.VideoWriter('tmp.mp4',fourcc,fps,size,True)

#读帧
success1, frame1=video1.read()
T=[]
audio_segments=[]
while success1:
    frame1=cv2.resize(frame1,size)
    audio_segments.append(decode_audio(frame1))
    frame1=cv2.cvtColor(frame1,cv2.COLOR_BGR2RGB)
    frame1=transforms.ToTensor()(frame1).to(device)
    t=time()
    x2=inn.Decoder(frame1)
    x2=x2.permute(1,2,0).cpu().detach().numpy()
    x2=(x2*255.0).astype(np.uint8)
    T.append(1/(time()-t))
    x2=cv2.cvtColor(x2,cv2.COLOR_RGB2BGR)
    output.write(x2)

    success1, frame1=video1.read()#获取下一帧

video1.release()
output.release()

# 导出处理后的音频
final_audio = AudioSegment(
        data=np.concatenate(audio_segments).tobytes(),
        sample_width=2,
        frame_rate=48000,
        channels=2
    )
final_audio.export('tmp.wav', format='wav')

# 合并处理后的音频与新视频
command = f"ffmpeg -i tmp.mp4 -i tmp.wav -c:v copy -c:a aac -strict experimental {decodeVideoPath}"
subprocess.call(command, shell=True)

plt.plot(T)
plt.savefig("decodeFPS.png")
print(f"average FPS: {sum(T)/len(T):.2f}")