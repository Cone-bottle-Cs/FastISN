import torch
import torch.quantization
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn

#device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=torch.device("cpu")

class InvertibleBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.conv1=nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.activation=nn.ReLU()
        self.FF=nn.quantized.FloatFunctional()

    def forward(self,x1,x2):
        #y1=x1+self.conv1(self.activation(x2))
        y1=self.FF.add(x1,self.conv1(self.activation(x2)))
        #y2=x2+self.conv2(self.activation(y1))
        y2=self.FF.add(x2,self.conv2(self.activation(y1)))
        return y1,y2

    def inverse(self,y1,y2):
        #x2=y2-self.conv2(self.activation(y1))
        x2=self.FF.sub(y2,self.conv2(self.activation(y1)))  # 替换为torch.sub来支持量化
        #x1=y1-self.conv1(self.activation(x2))
        x1=self.FF.sub(y1,self.conv1(self.activation(x2)))  # 同样用torch.sub
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

# 加载预训练模型
model=InvertibleNN(channels=3,num_blocks=2)
model.load_state_dict(torch.load('model.pth'))
model.to(device)

# 进行量化
# 设置模型为评估模式
model.eval()

# 量化配置：量化卷积和激活函数
model.qconfig=torch.quantization.get_default_qconfig('fbgemm')

# 准备模型进行量化
model_prepared=torch.quantization.prepare(model)

# 执行量化，不需要重新训练
model_quantized=torch.quantization.convert(model_prepared)
model_quantized.to(device)
model_quantized.eval()
torch.save(model_quantized.state_dict(),'modelQ.pth')

# 测试量化模型
# 假设 im1 和 im2 是你的输入图像，大小和形状都需要符合模型的输入要求
im1=Image.open("im1.png").convert("RGB").resize((1920,1080))
im1=transforms.ToTensor()(im1).unsqueeze(0).to(device)
im1=torch.quantize_per_tensor(im1,scale=1.0/255,zero_point=0,dtype=torch.quint8)
im2=Image.open("im2.png").convert("RGB").resize((1920,1080))
im2=transforms.ToTensor()(im2).unsqueeze(0).to(device)
im2=torch.quantize_per_tensor(im2,scale=1.0/255,zero_point=0,dtype=torch.quint8)
key=torch.zeros((3,1080,1920))+0.5
key=key.repeat(1,1,1,1).to(device)
key=torch.quantize_per_tensor(key,scale=1.0/255,zero_point=0,dtype=torch.quint8)

with torch.no_grad():
    y1,_=model_quantized(im1,im2)
    _,x2=model_quantized(y1,key)

plt.subplot(2,2,1),plt.imshow(im1.squeeze().permute(1,2,0).dequantize().cpu().numpy())
plt.subplot(2,2,2),plt.imshow(im2.squeeze().permute(1,2,0).dequantize().cpu().numpy())
plt.subplot(2,2,3),plt.imshow(y1.squeeze().permute(1,2,0).dequantize().cpu().numpy())
plt.subplot(2,2,4),plt.imshow(x2.squeeze().permute(1,2,0).dequantize().cpu().numpy())
plt.show()