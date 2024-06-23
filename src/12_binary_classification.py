import torch
from torch import nn
from torch.utils.data import DataLoader
import utils.torch_util as tu
import model_layers.binary_classification.main_model as binary_classification
import os
from torch import optim

"""
[이진 분류 모델 - 토치 모델]
- 이번에는 분류 문제를 토치 모델로 만들어보겠습니다.
    분류 문제를 해결하는 방법은 여러가지가 있지만,
    앞서 알아본 신경망 모델 역시 회귀뿐 아니라 분류 모델로도 사용이 가능합니다.
    여기서는 로지스틱 회귀 라는 분류 모델을 사용할 것입니다.
    
- 로지스틱 회귀는, 회귀 모델의 연장선인데, 다른점으로는, 입력받은 값을 0에서 1 사이의 값으로 변환하는 것 뿐입니다.
    이렇게 된다면 해당 모델의 결과값은, '확률' 이 되는 것입니다.
    반환값이 1차원이라면, 해당 차원이 의미하는 클래스에 속하느냐 아니냐는 의미의 이진 분류 모델이 되며,
    반환값이 2차원 이상이라면, 각 차원별로 어디에 속할 확률이 높은지에 대한 다중 클래스 분류 모델이 되는 것입니다.
    로지스틱 회귀 모델의 구조는, 모델의 끝부분에, 결과값을 0에서 1 사이로 만들어주는 Sigmoid 함수를 결합하기만 하면 됩니다.
    
- 분류 모델에서 오차 함수는 회귀모델에서와 같이 MSE 를 사용하는 것이 아닙니다.
    MSE 는 값의 한도가 없을 때를 가정하여, 잔차를 제곱하여 학습이 잘 되도록 하는 오차함수이므로,
    결과값이 1 과 0 사이에 머무는 분류 모델의 오차 함수로는 부적격합니다.
    분류 모델에서 흔히 사용하는 오차 함수는 교차 엔트로피 오차(Cross Entropy Error) 함수를 사용합니다.
    수식으로는,
    -((y log f(x)) + ((1-y)log(1-f(x))))
    로 나타냅니다.
    log 앞의 y 가 0 이 된다면 해당 계산은 사라지는 것이 됩니다.
    즉, y 가 1 이라면 우측계산이 사라지고, y 가 0 이라면 좌측 계산이 사라집니다.
    이때, y 와 f(x) 가 동일한 값, 즉 1 log 1 이라면 0 이 되며,
    1 log 0.01 이라면 -2 가 되고, 1 log 0.00001 이라면 -5 가 됩니다. 최종적으로 앞의 음수 부호가 사라지므로, 
    정답이 가까울 수록 값은 0 에 가까워지고, 오답으로 멀어질수록 값이 커지는 오차 함수임을 알 수 있습니다.

- 아래 코드는 이진 분류 모델을 본격적으로 토치 모델로 만들어 사용하는 예시를 보여줍니다.
    앞서 작성한 NN 모델 템플릿대로 작성 하였습니다.
"""


def main():
    # 사용 가능 디바이스
    device = tu.get_gpu_support_device(gpu_support=True)

    # 데이터셋 객체 생성 (ex : tensor([[-10., 100., 82.], ...], device = cpu), tensor([[327.7900], ...], device = cpu))
    # CSV 파일로 데이터셋 형성 (1 행에 라벨이 존재하고, 그 라벨로 x, y 데이터를 분류 합니다.)
    dataset = tu.CsvModelDataset(
        csv_file_full_url="../resources/datasets/12_binary_classification/binary.csv",
        x_column_labels=['x1', 'x2', 'x3'],
        y_column_labels=['y1']
    )

    # 학습용, 검증용, 테스트용 데이터를 비율에 따라 분리
    train_dataset, validation_dataset = tu.split_dataset(
        dataset=dataset,
        train_data_rate=0.8,
        validation_data_rate=0.2
    )

    # 데이터 로더 래핑
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=True, drop_last=True)

    # 모델 생성
    model = binary_classification.MainModel()

    # 모델 학습
    # 손실 함수는 nn.BCELoss 를 사용합니다.(Binary Cross Entropy)
    # 구현한다면,
    # losses = -(y_train * torch.log(model_out) +
    #            (1 - y_train) * torch.log(1 - model_out))
    # 이렇게 합니다.
    # model_out 은 확률값으로 0 과 1 사이의 실수입니다.
    tu.train_model(
        device=device,
        model=model,
        criterion=nn.BCELoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.001),
        train_dataloader=train_dataloader,
        num_epochs=10000,
        validation_dataloader=validation_dataloader,
        check_point_file_save_directory_path="../_by_product_files/check_point_files/binary_classification",
        # check_point_load_file_full_path="../_by_product_files/check_point_files/~/checkpoint(2024_02_29_17_51_09_330).pt",
        log_freq=1000
    )

    # 모델 저장
    model_file_save_directory_path = "../_by_product_files/torch_model_files/binary_classification"
    if not os.path.exists(model_file_save_directory_path):
        os.makedirs(model_file_save_directory_path)
    save_file_full_path = tu.save_model_file(
        model=model,
        model_file_save_directory_path=model_file_save_directory_path
    )

    # # 저장된 모델 불러오기
    model = torch.load(save_file_full_path, map_location=device)
    print("Model Load Complete!")
    print(model)

    # 모델 순전파
    with torch.no_grad():
        model.eval()
        for x, y in validation_dataloader:
            x = x.to(device)

            outputs = model(x)

            print(outputs)
            print(outputs >= torch.FloatTensor([0.5]).to(device))
            print("--------------------")


if __name__ == '__main__':
    main()
