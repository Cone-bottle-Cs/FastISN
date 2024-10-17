import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from time  import time
import itertools
import lpips

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_seed: int=42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

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

class PSNRLoss(nn.Module):
    def __init__(self):
        super(PSNRLoss,self).__init__()

    def forward(self,img1,img2):
        mse=torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20*torch.log10(img2.max()/torch.sqrt(mse))

def train_model(model,lr,dataset,key,num_epochs=10,batch_size=64,A:list=[32,1,1,0,0],compress:bool=False,lossList:list=[],psnr_y1List:list=[],psnr_x2List:list=[]):
    model.train()
    a1,a2,a3=A
    loss_fn_vgg=lpips.LPIPS(net='alex')
    loss_fn_vgg.cuda()
    compute_psnr=PSNRLoss()
    optimizer=optim.Adamax(model.parameters(),lr=lr,betas=(0.9,0.999))
    data_loader=DataLoader(dataset,batch_size=batch_size,shuffle=True)
    Blur=transforms.GaussianBlur(7,sigma=(0.5,2.0))

    target_y2=key.repeat(batch_size,1,1,1).to(device)

    t=time()
    for epoch in range(num_epochs):
        running_loss=0.0
        x2Loss=0.0
        y1Loss=0.0

        for batch_idx,(x1,_) in enumerate(data_loader):
            x1=x1.to(device)
            if x1.shape[0]<batch_size:
                break
            x2,_=next(iter(data_loader))
            x2=x2.to(device)

            y1,y2=model(x1,x2)
            loss_y1=loss_fn_vgg(y1,x1)
            psnr_y1=compute_psnr(y1,x1)
            loss_y2=loss_fn_vgg(y2,target_y2)

            if compress:
                #y1=JPEG(y1)
                y1=Blur(y1)

            _,decoded_x2=model.inverse(y1,target_y2)
            loss_x2=loss_fn_vgg(x2,decoded_x2)
            psnr_x2=compute_psnr(decoded_x2,x2)

            loss=a1*loss_y1+a2*loss_y2+a3*loss_x2
            loss=loss.mean()

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss+= loss.item()
            x2Loss+=loss_x2
            y1Loss+=loss_y1

        lossList.append(running_loss/len(data_loader))
        psnr_y1List.append(psnr_y1.item())
        psnr_x2List.append(psnr_x2.item())
        print(f"Epoch [{epoch+1}/{num_epochs}],Loss: {running_loss/len(data_loader):.4f},y1Loss: {y1Loss.mean()/len(data_loader):.4f},x2Loss: {x2Loss.mean()/len(data_loader):.4f},psnr_y1:{psnr_y1:.2f}dB,psnr_x2:{psnr_x2:.2f}dB,time:{int(time()-t)}s/{int((time()-t)*(num_epochs)/(epoch+1))}s")

        fig,ax1=plt.subplots()
        ax1.plot(lossList,label="Loss",color='blue')
        ax1.set_ylabel('Loss')
        ax1.tick_params(axis='y')
        ax2=ax1.twinx()
        ax2.plot(psnr_y1List,label="PSNR y1",color='red')
        ax2.plot(psnr_x2List,label="PSNR x2",color='green')
        ax2.set_ylabel('PSNR (dB)')
        ax2.tick_params(axis='y')
        fig.legend(loc="upper right",bbox_to_anchor=(1,1),bbox_transform=ax1.transAxes)
        plt.savefig("Loss.png")
        plt.clf(),plt.cla(),plt.close(fig)

        torch.save(model.state_dict(),'model.pth')
    #plt.ioff()
    return lossList,psnr_y1List,psnr_x2List

def test_model(model,key,container=None,cover_image=None,compress:bool=False):
    model.eval()
    compute_psnr=PSNRLoss()
    if container is None:
      dataset=datasets.CIFAR10(root='/data',train=True,transform=transform,download=True)
      data_loader=DataLoader(dataset,batch_size=1,shuffle=False)
      x1,_=next(iter(data_loader))
      x1=x1.to(device)
    else:
      x1=container.unsqueeze(0).to(device)
    x2=cover_image.unsqueeze(0).to(device)
    target_y2=key.repeat(1,1,1,1).to(device)

    with torch.no_grad():
        y1,y2=model(x1,x2)

        if compress:
            decoded_x1,decoded_x2=model.inverse(JPEG(y1),target_y2)
        else:
            decoded_x1,decoded_x2=model.inverse(y1,target_y2)

    diff=torch.abs(decoded_x2 - x2).mean().item()
    fig,axs=plt.subplots(2,2,figsize=(4,4))

    axs[0][0].imshow(x1.squeeze().permute(1,2,0).cpu().numpy())
    axs[0][0].set_title('Original Image (x1)')
    axs[0][0].axis('off')

    axs[0][1].imshow(x2.squeeze().permute(1,2,0).cpu().numpy())
    axs[0][1].set_title('Stego Content (x2)')
    axs[0][1].axis('off')

    print(y1.min(),y1.max())
    axs[1][0].imshow(y1.squeeze().permute(1,2,0).cpu().numpy())
    axs[1][0].set_title('Stego Image (y1)')
    axs[1][0].text(0,0,f'{compute_psnr(x1,y1):.2f}dB',va='bottom',ha='center')
    axs[1][0].axis('off')

    axs[1][1].imshow(decoded_x2.squeeze().permute(1,2,0).cpu().numpy())
    axs[1][1].set_title('Decoded Stego Content')
    axs[1][1].text(0,0,f'{compute_psnr(x2,decoded_x2):.2f}dB',va='bottom',ha='center')
    axs[1][1].axis('off')

    plt.show()

    return diff
def JPEG(im):
    """
    输入：PyTorch 张量 (batchSize,3,H,W)，表示一批图片
    输出：JPEG 压缩后的图片张量 (batchSize,3,H,W)
    """
    Q=torch.tensor([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
    ],requires_grad=True,dtype=torch.float32).to(im.device)

    DCT=np.zeros((8,8,8,8),dtype=np.float32)
    for x,y,u,v in itertools.product(range(8),repeat=4):
        DCT[x,y,u,v]=np.cos((2*x+1)*u*np.pi/16)*np.cos(
            (2*y+1)*v*np.pi/16)
    DCT= torch.from_numpy(DCT).float().to(im.device)

    IDCT=np.zeros((8,8,8,8),dtype=np.float32)
    for x,y,u,v in itertools.product(range(8),repeat=4):
        IDCT[x,y,u,v]=np.cos((2*u+1)*x*np.pi/16)*np.cos(
            (2*v+1)*y*np.pi/16)
    IDCT=torch.from_numpy(IDCT).float().to(im.device)

    alpha=np.array([1./np.sqrt(2)]+[1]*7)
    scale=torch.from_numpy(np.outer(alpha,alpha)*0.25).float().to(im.device)

    def dct_2d(im):
        result=scale*torch.tensordot(im - 128,DCT,dims=2)
        result.view(im.shape)
        return result

    def idct_2d(im):
        result =scale*torch.tensordot(im,IDCT,dims=2)+128
        result.view(im.shape)
        return result

    batch_size,channels,h,w=im.shape

    pad_h=(8 - (h % 8)) if h % 8 != 0 else 0
    pad_w=(8 - (w % 8)) if w % 8 != 0 else 0
    im_padded=torch.nn.functional.pad(im,(0,pad_w,0,pad_h))
    h+=pad_h
    w+=pad_w

    patches=im_padded.view(batch_size,channels,h // 8,8,w // 8,8)
    patches=patches.permute(0,1,2,4,3,5)  # 调整维度为 (batch_size,channels,num_blocks_h,num_blocks_w,8,8)
    blocks=patches.contiguous().view(batch_size,channels,-1,8,8)

    # 进行 DCT 变换、量化和反量化
    dct_blocks=dct_2d(blocks*255.0)  # 投射到0~255，然后对所有块进行 DCT

    quantized_blocks=torch.round(dct_blocks/Q)*Q  # 量化并反量化

    idct_blocks=idct_2d(quantized_blocks)/255.0 # 对所有块进行 IDCT，然后投射回0~1

    permuted_tensor=idct_blocks.view(batch_size,channels,h//8,w//8,8,8).permute(0,1,2,4,3,5)
    compressed_image=permuted_tensor.contiguous().view(batch_size,channels,h,w)

    return compressed_image

if __name__ == '__main__':
    lr=0.0002
    batch_size=128

    transform=transforms.Compose([transforms.ToTensor()])
    dataset=datasets.CIFAR10(root='/data',train=True,transform=transform,download=True)
    model=InvertibleNN(channels=3,num_blocks=2)
    model.to(device)

    key=torch.zeros((3,32,32))+0.5
    model.load_state_dict(torch.load('model.pth'))
    train_model(model,lr,dataset,key,num_epochs=100,batch_size=batch_size,A=[16,1,20,0,0.1],compress=True)

    container=Image.open("im1.png").convert("RGB").resize((1920,1080))
    container=transforms.ToTensor()(container)
    cover_image=Image.open("im2.png").convert("RGB").resize((1920,1080))
    cover_image=transforms.ToTensor()(cover_image)
    key=torch.zeros((3,1080,1920))+0.5
    test_model(model,key,container=container,cover_image=cover_image)
    test_model(model,key,container=container,cover_image=cover_image,compress=True)