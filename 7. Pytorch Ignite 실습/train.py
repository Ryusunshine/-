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
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1) #--gpu_id = gpu에서 돌릴건지 말건지

    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)

    config = p.parse_args()

    return config


def main(config): #main이 위의 config을 받아옴
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    train_loader, valid_loader, test_loader = get_loaders(config) #dataloader 받아옴

    print("Train:", len(train_loader.dataset)) #몇개의 샘플들이 있는지 알고싶을때
    print("Valid:", len(valid_loader.dataset))
    print("Test:", len(test_loader.dataset))

    #ImageClassifer = model 파일에서 선언한 함수
    model = ImageClassifier(28**2, 10).to(device) #입력은 784이고 10개의 class로 구분할거야. cpu인지 gpu인지 보고 gpu이면 (device)로 옮겨줘 
    optimizer = optim.Adam(model.parameters()) #model.parameters()하면 model안에 있는 파라미터들이 iterative하게 나옴. 그럼 이걸 대상으로 아담이 optimize한다

    crit = nn.NLLLoss() #classification이니깐 CrossEntropy를 한다. 그래서 model.py 파일을 보면 softmax로 끝나잇는걸 알수잇다.

    trainer = Trainer(config)
    trainer.train(model, crit, optimizer, train_loader, valid_loader) #test loader는 여기서 쓸 필요없음

if __name__ == '__main__':
    config = define_argparser()
    main(config)
