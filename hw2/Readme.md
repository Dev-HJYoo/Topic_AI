# 1. 각 .py 설명

* main.py: 과제와 관련된 모든 task에 대해 차례대로 실행을 하는 파일 (FP32 -> DQ -> QAT)
* resnet.py: resnet 18 네트워크와 Quantization network를 정의한 파일
* utils.py: 기타 실행에 필요한 함수를 선언한 파일

# 2. 실행 방법

처음 실행시 명령어.

`
python main.py --epoch 200
`

epoch 200으로 학습을 시작합니다. 또한, FP32 학습및 테스트 부터 DQ를 거쳐 QAT까지 학습및 테스트를 끝냅니다.



`
python main.py --resume
`

checkpoint/ckpt.pth 파일을 불러와 테스트를 진행할 수 있습니다. FP32 테스트 부터 DQ와 QAT까지 학습 및 테스트를 진행합니다.

# 3. main.py의 각 라인 설명

line 95: FP32을 학습할 시 사용할 epochs (args로 할당받을 수 있음.) \
line 97~131: CIFAR-10 dataset , transforms, loader 및 resnet18 선언 \
line 132~139: --resume 사용시 checkpoint를 할당 \
line 141~150: FP32로 학습 및 테스트 진행 \
\
line 156~163: FP32 테스트 \
line 165~189: FP32 -> DQ 변환 및 테스트  \
\
line 195~205: QAT 관련 설정 값 \
line 207~219: QAT 학습 및 테스트 