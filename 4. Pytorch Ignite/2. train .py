import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import ImageClassifier
from trainer import Trainer
from data_loader import get_loaders

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)

    config = p.parse_args()

    return config


def main(config): #기본변수(model, optimizer, crit, trainer, train)설정
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    train_loader, valid_loader, test_loader = get_loaders(config)#dataloader 파일에서 구현한 train,valid,test_loader 가져오기

    print("Train:", len(train_loader.dataset)) #len 은 data_loader파일에서 구현한 len함수
    print("Valid:", len(valid_loader.dataset))
    print("Test:", len(test_loader.dataset))

    model = ImageClassifier(28**2, 10).to(device) # 입력은 784(28**2)이고 출력은 10개의 class로 분류할거야.그리고 cpu인지 gpu인지 여기로 옮길거야
    optimizer = optim.Adam(model.parameters()) #model.parameters()하면 model안에 있는 파라미터들이 iterative하게 나옴. 그럼 이걸 대상으로 아담이 optimize한다
   
    crit = nn.NLLLoss() #classification이니깐 CrossEntropy를 한다.

    trainer = Trainer(config) #trainer.py 파일에 Trainer 함수잇음
    trainer.train(model, crit, optimizer, train_loader, valid_loader)

if __name__ == '__main__':
    config = define_argparser()
    main(config)
