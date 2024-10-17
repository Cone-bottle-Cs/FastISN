import cv2
import torch
from torchvision import transforms
from time import time
import numpy as np
from pydub import AudioSegment
import subprocess
from util import INN,encode_audio
import argparse

parser=argparse.ArgumentParser(description='Decode video file.')
parser.add_argument('--cover',type=str,default='Stellaris.mp4',help='Path to the cover video file.')
parser.add_argument('--secret',type=str,default='secret.mp4',help='Path to the secret video file.')
parser.add_argument('--save',type=str,default='Encode.mp4',help='Path to save the video file.')
parser.add_argument('--model',type=str,default='model.pth',help='Path of the INN model.')
parser.add_argument('--size',type=tuple,default=(3,1080,1920),help='Shape of frames in [C,H,W].')
parser.add_argument('--device',type=str,default="cpu",help='cuda or cpu')
parser.add_argument('--fps',type=int,default=30,help='frames per second')
parser.add_argument('--redundant',type=int,default=9,help='the number of rows to encode audio')
args=parser.parse_args()

#Initialization
'''
video1=cv2.VideoCapture('Stellaris.mp4')
video2=cv2.VideoCapture('secret.mp4')
decodeVideoPath="Encode.mp4"
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
inn=INN(size=(3,1080,1920),path="model.pth",device=device)
fps=30
size=(1920,1080)
'''
video1=cv2.VideoCapture(args.cover)
video2=cv2.VideoCapture(args.secret)
decodeVideoPath=args.save
device=torch.device(args.device)
inn=INN(size=args.size,path=args.model,device=device)
fps=args.fps
size=(args.size[2],args.size[1])#(W,H)

#save the audio of container video
audioC=AudioSegment.from_file('Stellaris.mp4')
audioC.export("tmp.wav",format="wav")

#audio of secret video
audioS=AudioSegment.from_file('secret.mp4')
frame_duration_ms=1000/fps

fourcc=cv2.VideoWriter_fourcc(*'mp4v')
output=cv2.VideoWriter('tmp.mp4',fourcc,fps,size)
success1,frame1=video1.read()
success2,frame2=video2.read()
T=[]
frame_count=0
while success1 and success2:
    frame1=cv2.resize(frame1,size)
    frame1=cv2.cvtColor(frame1,cv2.COLOR_BGR2RGB)
    frame2=cv2.resize(frame2,size)
    frame2=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
    frame1_tensor=transforms.ToTensor()(frame1).to(device)
    frame2_tensor=transforms.ToTensor()(frame2).to(device)
    
    t=time()
    y1=inn.Encoder(frame1_tensor,frame2_tensor)
    y1=y1.permute(1,2,0).cpu().detach().numpy()
    y1=(y1*255.0).astype(np.uint8)
    T.append(1/(time() - t))
    
    y1=cv2.cvtColor(y1,cv2.COLOR_RGB2BGR)

    #encode audio
    start_time=frame_count*frame_duration_ms
    end_time=start_time+frame_duration_ms
    audio_segment=audioS[start_time:end_time]
    audio=np.array(audio_segment.get_array_of_samples())
    y1=encode_audio(audio,y1,redundant=args.redundant)
    frame_count+=1
    output.write(y1)

    success1,frame1=video1.read()#next frame
    success2,frame2=video2.read()

#release
video1.release()
video2.release()
output.release()

#merge audio and visuals by ffmpeg
command=f"ffmpeg -i tmp.mp4 -i tmp.wav -c:v copy -c:a aac -strict experimental {decodeVideoPath}"
subprocess.call(command,shell=True)

print(f"average FPS: {sum(T)/len(T):.2f}")
print(f"audio length: {audio.shape[0]}")