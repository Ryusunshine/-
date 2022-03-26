import torch
import torch.nn as nn


class SequenceClassifier(nn.Module):

    def __init__( #28 row가 28개 들어옴
        self,
        input_size, #한 row씩 들어오니깐 28개가 28row번 들어온다. 그래서 입력사이즈는 28.
        hidden_size,
        output_size,#출력사이즈도 28
        n_layers=4, #layer도 너무 깊게쌓으면 학습이 잘 안되므로 4개 추천
        dropout_p=.2,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True, #batch_first 를 써줘야 오류가 안남. 중요! 원래는 (time-step, bs, size = 28)순으로 오는데  우리는 (bs, time-step, size)쓸거임
            dropout=dropout_p,
            bidirectional=True,#우리는 non-autoregressive 쓸거니깐 bidirectional 쓸수잇음. 기본값이 False라서 True로 써줘야함
        )
        self.layers = nn.Sequential( #마지막으로 나온 하나의 layer에 softmax해준다.
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2), #hidden size가 정방향과 역방향에서 들어오므로 *2를 해준다. 
            nn.Linear(hidden_size * 2, output_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        # |x| = (batch_size, h, w) #grayscale이니간 (배치사이즈. 높이, 너비)로 들어온다. 이미지이긴하지만 h가 time-step인거고 w가 매 time step당 입력으로 들어오는 벡터의 크기

        z, _ = self.rnn(x) # 여기 강의에서는 z값만 중요하므로 뒤에값은 버리기위해 underscore(_)를 썼다.
        # |z| = (batch_size, h(높이)(28), hidden_size * 2). h는 마지막 층의 layer들을 모두 다 받아오는거이기때문에 
        #출력값으로 마지막 time-step의 hidden size가 나옴. (28(time-step), bs, size = 28)
        z = z[:, -1]
        # |z| = (batch_size, hidden_size * 2) #첫번째 차원 (batch_size)는 다 가져오고,batch는 자르면안됨. ([:]) 두번째 차원은 마지막 한개만 가져오기위해 [-1] 범위를 정해준다.
        y = self.layers(z)
        # |y| = (batch_size, output_size)

        return y
