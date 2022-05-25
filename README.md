# 하이퍼파라미터 설명
## argument로 epochs, augmentation 등을 수정 할 수 있습니다.
## epochs
### 기본값은 20이고 조정가능합니다.
## augmentation 
### -aug 0 또는 -aug 1로 조정가능합니다 기본값 0은 resize만 하고 1은 RandomHorizontalfilp(p=0.2)가 추가됩니다.
## optimizer
### Adam을 사용했고 weight_decay를 1e-4로 주었습니다. 이것역시 -wdecay 로 변경할 수 있습니다.
## scheduler
### CosineAnnealingLR을 사용했고 T_max=epochs, eta_min=lr/10 으로 해서 마지막 에폭까지 lr이 lr/10 까지 감소합니다.
## LSTM
### layer 수는 3개 hidden layer 수는 256개로 설정했습니다.
## Loss
### classfication 이라서 CrossEntropyLoss를 사용했습니다.
## amp
### 학습을 빨리 진행시키기 위해서 amp를 사용했습니다.

# train 하는 방법
## 예시
## python main.py -mode train -mtype rlstm -gpu 1 -epochs 20 -aug 1 -saveName model ....
## -mode
train인지 test인지 정하는 것입니다.  
default=test
## -mtype
어떤 모델로 할지 고르는 것입니다.  
default=dConv  
직접 짠 resnet: -mtype lstm, resnet: -mtype rlstm, 3dConv: -mtype (안 적거나 앞에서 안한거)
## -gpu
몇 번 gpu를 쓸지 정하는 것입니다.  
default=0
## -epochs
default=20
## -aug
위에도 있었는데 0이면 resize만 1이면 Horizontalflip 추가  
default=0
## -saveName
./models/{saveName} 에 저장됩니다.  
default=Unix Time
## -lr
초기 lr 지정  
default=1e-3
## -wdecay
optimizer 파라미터로 넘어감  
default=1e-4
# test 하는 방법
## 예시
## ptyhon main.py -mode test -gpu .... -mtype rlstm
## 모델이 총 3가지 있습니다
*custom resnet(직접짠거) + LSTM, -mtype lstm
*torchvision resnet + LSTM, -mtype rlstm
*3d convlution, 위 2개 제외
