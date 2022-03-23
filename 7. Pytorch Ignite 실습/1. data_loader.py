import torch

from torch.utils.data import Dataset, DataLoader


class MnistDataset(Dataset):

    def __init__(self, data, labels, flatten=True): # 28 X28 차원을 하나의 벡터로 만들어주기위해 flatten
        self.data = data #이미지 
        self.labels = labels #원핫 longtensor
        self.flatten = flatten

        super().__init__()

    def __len__(self):
        return self.data.size(0) #몇개의 샘플이 있냐

    def __getitem__(self, idx): #미니배치
        x = self.data[idx] #첫번째 차원에서 slicing 또는 selection 수행, 
        y = self.labels[idx] #y는 long인덱스로 되어있어서 (1,) 이렇게 숫자 하나 들어있다.

        if self.flatten: #self.flatten이 True이면 x차원을 1차원으로 펴준다. 원래 (28*28)차원이엇는데 (784,)로 하나의 차원으로 바뀜
            x = x.view(-1) #flatten(), .view(-1)는 둘다 tensor를 1차원으로 펴주는 역할을 한다.

        return x, y

# DataLoader가 dataset을 가지고있다가 x,y를 호출하면은 하나의 애들을 고를때마다 (784,)와 (1,) 크기의 텐서들이 나오고 개네들을 합치면 미니배치 완성
# 만약 배치사이즈가 256이라고 하면  (784, )가 256개 있으니깐 (784,)X 256 이라서 (256, 784)가 될것이다. 


#  batch size는 하나의 소그룹에 속하는 데이터 수를 의미
    
    


def load_mnist(is_train=True, flatten=True): #is_train 은 train set이냐 valid set이냐 를 의미
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        '../data', train=is_train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    x = dataset.data.float() / 255.
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1)

    return x, y


def get_loaders(config): #Mnist으로부터 받은 data들을 x,y로 정제해주기
    
    x, y = load_mnist(is_train=True, flatten=False)
    
    # MNIST같은 경우에는 train과 test set이 fix가 되어있음. 6만장의 train set과 1만장의 test set으로 구성되어있는데, test set은 fix되어있고 train set은 6만장에서 train과 valid로 나눠야한다. 비율은 주로 8대 2나 6대4임.

    train_cnt = int(x.size(0) * config.train_ratio) #x.size(0)은 전체 사이즈, 6만장
    valid_cnt = x.size(0) - train_cnt

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm(x.size(0)) 
    #x와y를 각각 셔플링하면 x와 y의 쌍이 깨지게되니깐 우선 인덱스를 먼저 만들어놓고 그거에 따라서 x와 y를 같이 셔플링하기위함이다.
    train_x, valid_x = torch.index_select(
        x,
        dim=0, #|x| = (60000, 28, 28) 이고 dim = 0, 즉 6만장에서 인덱스대로 셔플링해라 라는 소리
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)
    
    train_y, valid_y = torch.index_select(
        y,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)

# |train_x| = (48000, 28, 28)
# |valid_x| = (12000, 28, 28)

( #우리는 방금 만든 Dataset을 Dataloader넣을거임, train, valid, test loader 각각 만들어줘야함.
    

    train_loader = DataLoader(
        dataset=MnistDataset(train_x, train_y, flatten=True),
        batch_size=config.batch_size, #배치 사이즈 미리 지정
        shuffle=True, # train set은 무조건 셔플링해야함
    )
    valid_loader = DataLoader(
        dataset=MnistDataset(valid_x, valid_y, flatten=True),
        batch_size=config.batch_size,
        shuffle=True, #valid set은 셔플링 해도 되고 안해도 됨
    )

    test_x, test_y = load_mnist(is_train=False, flatten=False)
    test_loader = DataLoader(
        dataset=MnistDataset(test_x, test_y, flatten=True),
        batch_size=config.batch_size,
        shuffle=False, #test set은 주로 셔플링안함
    )

    return train_loader, valid_loader, test_loader
