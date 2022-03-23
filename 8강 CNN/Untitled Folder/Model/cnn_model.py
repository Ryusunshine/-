import torch
import torch.nn as nn


class ConvolutionBlock(nn.Module):

    def __init__(self, in_channels, out_channels): #convolution block를 한개씩 지날때마다 이미지의 사이즈는 반으로 줄어들고 channel의 수는 2배씩 늘어남
        self.in_channels = in_channels
        self.out_channels = out_channels

        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=1), #kernel size(필터) = (3,3), 필터사이즈 (3,3)쓰고 패딩 하나쓰면 입출력의 사이즈는 똑같음
            nn.ReLU(),
            nn.BatchNorm2d(out_channels), #BatchNorm1d는 출력의 크기를 넣어줬었는데 BatchNorm2d는 output channel의 숫자를 넣어줘야함.
            nn.Conv2d(out_channels, out_channels, (3, 3), stride=2, padding=1), #여기서는 stride를 써서 사이즈가 반으로 줄어들음
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x): #지금은 grayscale 이라서 in_Channel(입력하는 채널의 수)는 1이다. 만약 컬러이미지엿다면 3이다.
        # |x| = (batch_size, in_channels, h, w)

        y = self.layers(x)
        # |y| = (batch_size, out_channels, h, w)

        return y


class ConvolutionalClassifier(nn.Module):

    def __init__(self, output_size): #이미지분류모델이기때문에 입력되는 차원(28,28)은 고정이므로 ouptput_size만 받는다
        self.output_size = output_size

        super().__init__()

        self.blocks = nn.Sequential( # |x| = (n, 1, 28, 28) n은 batch_size, 1은 gray scale
            ConvolutionBlock(1, 32), # (n, 32, 14, 14)
            ConvolutionBlock(32, 64), # (n, 64, 7, 7)
            ConvolutionBlock(64, 128), # (n, 128, 4, 4)
            ConvolutionBlock(128, 256), # (n, 256, 2, 2)
            ConvolutionBlock(256, 512), # (n, 512, 1, 1) = (bs, 512) 512차원의 벡터가 배치사이즈만큼 잇는거임
        )
        self.layers = nn.Sequential(
            nn.Linear(512, 50), #512차원을 받아서 50차원으로 넘겨주고 
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, output_size), #50차원을 output_size는 10개인 클래스로 출력한다.
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        assert x.dim() > 2 #flatten을 하고들어올까봐 assert. 애는 flatten되어있으면 안됨.
#원래 정상이면(bs, 28, 28)인데 이게 flatten되면 (bs, 784)임.이게 기존의 fully-connected model로 만들었을때의 차원인데 이렇게 들어오면 작동안함.오류남. 그래서 (bs, 1, 28, 28)이 들어오지 않게 2보다 커야한다. 

        if x.dim() == 3: #만약에 채널수가 없는 dimension이 3개인경우(bs, 28, 28)
            # |x| = (batch_size, h, w)
            x = x.view(-1, 1, x.size(-2), x.size(-1)) #채널의 수는 gray scale이니깐 1이 들어가고 남은 batch_size는 -1로 저절로. x.size(-2), x.size(-1)는 원래 수 그대로.
        # |x| = (batch_size, 1, h, w)

        z = self.blocks(x) #그걸 blocks에 통과해주면 z는 (512,1,1)이 나온다.
        # |z| = (batch_size, 512, 1, 1)

        y = self.layers(z.squeeze()) #squeeze하면 1들을 없애서 간단하게 되고 이걸 layers에 통과시키면 끝.
        # |y| = (batch_size, output_size)
        
 #block들을 반복할때마다 이미지개수는 줄어들고 커널들(특징들)은 늘어난다. 커널 하나가 어떤 feature을 의미하기때문에. 최종적으로 512개의 feature들이 나오게됨. 거기다가 우리는 softmax를 씌워서 우리가 원하는대로 classificaton 수행. 
        
      
        return y
