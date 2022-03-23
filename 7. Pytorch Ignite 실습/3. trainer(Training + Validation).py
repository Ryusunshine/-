from copy import deepcopy

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as torch_utils

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from utils import get_grad_norm, get_parameter_norm

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class MyEngine(Engine): 
    
    # func = feed-forward -> loss 계산 -> backpropagation -> gradient descent -> 현재 상태 출력 과정을 담은 함수

    def __init__(self, func, model, crit, optimizer, config):
     
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.config = config

        super().__init__(func) # Ignite Engine only needs function to run.

        self.best_loss = np.inf
        self.best_model = None

        self.device = next(model.parameters()).device

    @staticmethod
    def train(engine, mini_batch): #train함수는 기본적으로 engine을 받아오고 현재 minibatch를 받아온다
        #for문안에 train_loader가 들어가면 iterative하게 x,y가 나오는데 그게 미니배치 이다. 그래서 미니배치안에 x,y가 튜플로 들어있다.
        # 이 mini-batch는 dataloader.py 파일안에 load_mnist함수에서 for문을 통해 나오는 x,y이다. 그래서 미니배치안에서 튜플이 들어잇다.
        #바로 # func = feed-forward -> loss 계산 -> backpropagation -> gradient descent -> 현재 상태 출력 과정 코드
      
        engine.model.train() # Because we assign model as class variable, we can easily access to it.
        engine.optimizer.zero_grad() #한번의 iteration 마다 불러오는거니깐 zero.grad()해준다.

        x, y = mini_batch
        x, y = x.to(engine.device), y.to(engine.device) #모델하고 같은device에 보내주는 작업

        # feed-forward
        y_hat = engine.model(x) #x의 배치사이즈는 256 이고 flatten 된 값이 들어갓을거라 784차원이 들어가서 |x| = (256, 784)
        # y_hat은 (256(배치사이즈). 10(longTensor)

        loss = engine.crit(y_hat, y) #loss 는 스칼라값으로 나옴
        loss.backward()

      #y가 regression 문제일수도 잇으니깐 y가 longtensor이면 classification 문제이고 regression이면 일반 floattensor이다
        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor): 
            #isinstace(y, torch.LongTensor) = y가 torch.LongTensor에 있거나 torch.cuda.LongTensor에 있으면 True 반환한다. 이 모델이 regressino task일수도 잇으니깐 그때 보는 방법이 y가 longtensor이면 regression이 아니라 classification인거고 y가 일반 floatTensor이면 regression임
            accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0)) 
            ##argmax하면 인덱스 하나가 뽑히고 그거랑 y를 비교하면 True/False로 되어있을거다. 그걸 인제 sum를 때리면 1값인 True만 더해져서 개수가 나오고 y.size(0)로 y의 전체사이즈로 나누면 맞은개수나옴
            #dim = -1 은 (bs, 10)에서 10을 가리킴. 
        else:
            accuracy = 0

        #학습이 잘되고잇나 보조지표 역할
        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))
        #gradient norm을 통해서 현재 gradient가 얼마나 가파른지 알수잇음 

        # Take a step of gradient descent.
        engine.optimizer.step()

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            '|param|': p_norm,
            '|g_param|': g_norm,
        }

    @staticmethod
    def validate(engine, mini_batch): #feed-forward -> loss 계산 -> 현재 상태 출력 까지 과정
        engine.model.eval()

        with torch.no_grad():
            x, y = mini_batch
            x, y = x.to(engine.device), y.to(engine.device)

            #우리는 지금 Mnist classification하고있으니깐 model 에 들어가면 출력값으로 10개의 class값중 확률값으로 저장돼!
            y_hat = engine.model(x) #현재 x의 크기는 |x| = (batch_size, 784)
                                    #y_hat의 크기는 |y_hat| = (batch_size, 10)

            loss = engine.crit(y_hat, y) #loss결과값은 scalar

            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0)) 
            else:
                accuracy = 0

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
        }

    @staticmethod
    def attach(train_engine, validation_engine, verbose=VERBOSE_BATCH_WISE):
        
        #train_Engine과 validation_engine에다가 붙일거임
        # Attaching would be repaeted for serveral metrics.
       #미니배치마다 loss와 accuracy 를 return을 하면은 engine에 잇는 runningAverge가 engine에 metricname으로 다시 붙인다
        
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach( 
                engine,
                metric_name,
            )

        training_metric_names = ['loss', 'accuracy', '|param|', '|g_param|']

        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)#attach_running_average함수를 호출해서 train_engine에 training_metric_names = ['loss', 'accuracy', '|param|', '|g_param|'] 을 붙어놓음. print하는게 아니라 내부적으로 계산을 하는거임

        # If the verbosity is set, progress bar would be shown for mini-batch iterations.
        # Without ignite, you can use tqdm to implement progress bar.
        #1. 첫번째는 매 iteration마다 출력하고싶다하면 
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)

        # If the verbosity is set, statistics would be shown after each epoch.
        #1. 두번째는 epoch이 끝낫을때만 출력
        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED) #decorator을 써서 event handler을 등록할수잇다. epoch이 끝낫을때 출력을 해라
            def print_train_logs(engine):
                print('Epoch {} - |param|={:.2e} |g_param|={:.2e} loss={:.4e} accuracy={:.4f}'.format(
                    engine.state.epoch,
                    engine.state.metrics['|param|'],
                    engine.state.metrics['|g_param|'],
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                ))

        validation_metric_names = ['loss', 'accuracy']
        
        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        # Do same things for validation engine.
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                print('Validation - loss={:.4e} accuracy={:.4f} best_loss={:.4e}'.format(
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                    engine.best_loss,
                ))

    @staticmethod
    def check_best(engine): #best model 저장
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss: # If current epoch returns lower validation loss,
            engine.best_loss = loss  # Update lowest validation loss.
            engine.best_model = deepcopy(engine.model.state_dict()) # Update best model weights.

    @staticmethod
    def save_model(engine, train_engine, config, **kwargs):
        torch.save(
            {
                'model': engine.best_model,
                'config': config,
                **kwargs
            }, config.model_fn
        )


class Trainer(): #training & validation 과정 같이 들어잇는 함수

    def __init__(self, config):
        self.config = config

    def train( #train에서는 train_engine, validation_engine을 실행시키는게 다임.
        self,
        model, crit, optimizer,
        train_loader, valid_loader
    ):
        train_engine = MyEngine( #MyEngine인 변수들(model, crit, optimizer)을 주면서 동시에 MyEngine.train 실제 staticmethod를 MyEngine한테 function으로 넘겨줌 
            MyEngine.train, #staticmethod 인 MyEngine안에 있는 train 함수를 실행시킴. train안에서 iteration이 돌때마다 저 함수 호출
            model, crit, optimizer, self.config
        )
        validation_engine = MyEngine(#validation 도 마찬가지로 MyEngine.validate를 function으로 넘겨줌
            MyEngine.validate,
            model, crit, optimizer, self.config
        )

        MyEngine.attach( #위에서 만들었던 통계수치를 붙이는걸 여기서 한다
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )

        def run_validation(engine, validation_engine, valid_loader): #training의 한 epoch이 끝나면 validation의 한 epoch이 시작되야해서 trianing을 run을 시켜주면 validation이 호출이 언제될지 알려줘야함.그래서 train engine한테 야 너 epoch 끝날때마다 validation engine을 한 epoch씩 실행시켜라고 명령하게해준다. 그래서 이 run_validation함수는 train_engine의 한 epoch이 끝날때마다 호출해서 필요한 argument(validation_engine, valid_loader)들을 입력하여 validation engine에 넣어주고 다시 run 시켜줌. valid_loader중에서 한 epoch만 돌리라고 order. train_engine이 20번 epoch돌때 validation_engine은 한 epoch씩 돌리라는 뜻
            validation_engine.run(valid_loader, max_epochs=1)

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            run_validation, # function
            validation_engine, valid_loader, # arguments
        )
        validation_engine.add_event_handler( #validation engine의 한 epoch이 끝났을때 check_best 를 호출하고 
            Events.EPOCH_COMPLETED, # event
            MyEngine.check_best, # function
        )
        validation_engine.add_event_handler( #validation의 epoch이 끝낫을때 save_model를 호출해라
            Events.EPOCH_COMPLETED, # event
            MyEngine.save_model, # function
            train_engine, self.config, # arguments
        )

        train_engine.run( #이제 비로소 train_engine실행. 앞에서는 붙이는 작업을 열심히 한거고 이제 실제 실행
            train_loader,
            max_epochs=self.config.n_epochs,
        )

        model.load_state_dict(validation_engine.best_model)

        return model
