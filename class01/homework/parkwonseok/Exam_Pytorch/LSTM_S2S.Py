import torch
from torch import nn, Tensor

SOS_token = 0
EOS_token = 1

from sentance_preprocess import read_language
import random
lang_input, lang_output, pairs, \
= read_language('ENG','KOR',reverse= False, verbose= False)

for idx in range(10):
    print(random.choice(pairs))

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers ): # ~는 코드채워나가면서 채우자
        super(nn.LSTM, self).init__() # 상속 받기기
        self.input_size = input_size# 자기참조 주식예측 input_size는 뭐지? : 고가 저가 시작가 셀프 뒤에 하나씩 넣는다
        self.hidden_size = hidden_size # LSTM에서 나와야하는 것 일단 cellstate 신경안씀
        self.num_layers = num_layers
        self.batch_first = True
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size, 
            num_layers = num_layers,
            batch_first = True # 받을 정보에대해 ex)cnn 이미지가 어떻게? ((갯수 배치),h,w,3)이미지정보 rgb
            ) 
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size) # 크기값을 넣어줘야함
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size)
                        #                 배치만큼 들어옴  히든사이즈30개 인코더 통과한다음에 hidden size 내뱉으면

        LSTM_out, (hidden, cell) = self.lstm(x, (h0, c0)) # 인풋 h0, c0

        return LSTM_out
    



