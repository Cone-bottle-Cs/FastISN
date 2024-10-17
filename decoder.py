import cv2
import torch
from torchvision import transforms
from time import time
import numpy as np
from pydub import AudioSegment
import subprocess
from util import INN,decode_audio
import argparse

parser=argparse.ArgumentParser(description='Decode video file.')
parser.add_argument('--source',type=str,default='BBdown.mp4',help='Path to the video file to decode.')
parser.add_argument('--save',type=str,default='BBdecode.mp4',help='Path to save the video file.')
parser.add_argument('--model',type=str,default='model.pth',help='Path of the INN model.')
parser.add_argument('--size',type=tuple,default=(3,1080,1920),help='Shape of frames in [C,H,W].')
parser.add_argument('--device',type=str,default="cpu",help='cuda or cpu')
parser.add_argument('--fps',type=int,default=30,help='frames per second')
parser.add_argument('--redundant',type=int,default=9,help='the number of rows to encode audio')
parser.add_argument('--length',type=int,default=3200,help='the length of audio in each frame,which you can find after encoding')
args=parser.parse_args()


#Initialization
'''
video1=cv2.VideoCapture('BBdown.mp4')
decodeVideoPath="BBdecode.mp4"
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
inn=INN(size=(3,1080,1920),path="model.pth",device=device)
fps=30
size=(1920,1080)
'''
video1=cv2.VideoCapture(args.source)
decodeVideoPath=args.save
device=torch.device(args.device)
inn=INN(size=args.size,path=args.model,device=device)
fps=args.fps
size=(args.size[2],args.size[1])#(W,H)

fourcc=cv2.VideoWriter_fourcc(*'mp4v')
output=cv2.VideoWriter('tmp.mp4',fourcc,fps,size,True)
success1,frame1=video1.read()
T=[]
audio_segments=[]
while success1:
    frame1=cv2.resize(frame1,size)
    audio_segments.append(decode_audio(frame1,length=args.length,redundant=args.redundant))
    frame1=cv2.cvtColor(frame1,cv2.COLOR_BGR2RGB)
    frame1=transforms.ToTensor()(frame1).to(device)
    t=time()
    x2=inn.Decoder(frame1)
    x2=x2.permute(1,2,0).cpu().detach().numpy()
    x2=(x2*255.0).astype(np.uint8)
    T.append(1/(time()-t))
    x2=cv2.cvtColor(x2,cv2.COLOR_RGB2BGR)
    output.write(x2)

    success1,frame1=video1.read()#next frame

video1.release()
output.release()

#export audio
final_audio=AudioSegment(
        data=np.concatenate(audio_segments).tobytes(),
        sample_width=2,
        frame_rate=48000,
        channels=2
    )
final_audio.export('tmp.wav',format='wav')

#merge audio and visuals by ffmpeg
command=f"ffmpeg -i tmp.mp4 -i tmp.wav -c:v copy -c:a aac -strict experimental {decodeVideoPath}"
subprocess.call(command,shell=True)

print(f"average FPS: {sum(T)/len(T):.2f}")