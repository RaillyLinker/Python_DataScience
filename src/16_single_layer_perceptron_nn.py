import torch
from torch import nn
from torch.utils.data import DataLoader
import utils.torch_util as tu
import model_layers.single_layer_perceptron.main_model as single_layer_perceptron
import os
from torch import optim

"""
[단일 퍼셉트론]
- 아래 예시는 단층 퍼셉트론을 토치 모델로 만들어 사용하는 예시입니다.
    앞선 퍼셉트론 예시에서는 활성화 함수를 Step Function(역치를 넘어가면 1, 넘지 않았다면 0 을 반환)을 사용했지만,
    여기선 선형 함수에 활성화 함수를 Sigmoid 로 붙였습니다.
    고로, 아래에 구현한 단층 퍼셉트론은 Logistic Regression 과 다를 것이 없습니다.
- 활성화 함수에는 Sigmoid 외에도 여러가지 방법이 존재합니다.
    일단 Sigmoid 함수는,
    def sigmoid(x): # 시그모이드 함수 정의
        return 1/(1+np.exp(-x))
    이렇게 구현이 가능 합니다.
    이렇게 하면 값이 커질수록 한없이 1 에 가까워지고, 값이 작아지면 한없이 0 에 가까워지는 S자형 결과값을 얻을 수 있습니다.
"""


def main():
    # 사용 가능 디바이스
    device = tu.get_gpu_support_device(gpu_support=True)

    # 데이터셋 객체 생성 (ex : tensor([[-10., 100., 82.], ...], device = cpu), tensor([[327.7900], ...], device = cpu))
    # CSV 파일로 데이터셋 형성 (1 행에 라벨이 존재하고, 그 라벨로 x, y 데이터를 분류 합니다.)
    # 여기서 X 값은 (True, True) 와 같이 Boolean 값이고, Y 값도 (True) 같은 Boolean 값입니다.
    # 즉, 이진분류를 해결하는 문제이자, 논리 회로를 구성하는 문제입니다.
    # 단순히 학습시키는 아래와 같은 예시에는 상관이 없지만, 일단 단층 신경망은 Xor 회로를 구현할 수 없습니다.
    # 다층 신경망부터 Xor 회로를 만들 수 있게 됩니다.
    dataset = tu.CsvModelDataset(
        csv_file_full_url="../resources/datasets/16_single_layer_perceptron_nn/perceptron.csv",
        x_column_labels=['x1', 'x2'],
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
    validation_dataloader = DataLoader(validation_dataset, batch_size=10, shuffle=True, drop_last=True)

    # 모델 생성
    model = single_layer_perceptron.MainModel()

    # 모델 학습
    tu.train_model(
        device=device,
        model=model,
        criterion=nn.BCELoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.001),
        train_dataloader=train_dataloader,
        num_epochs=10000,
        validation_dataloader=validation_dataloader,
        check_point_file_save_directory_path="../_check_point_files/single_layer_perceptron",
        # check_point_load_file_full_path="../_check_point_files/~/checkpoint(2024_02_29_17_51_09_330).pt",
        log_freq=1000
    )

    # 모델 저장
    model_file_save_directory_path = "../_torch_model_files/single_layer_perceptron"
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
        inputs = torch.FloatTensor([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]).to(device)
        outputs = model(inputs)

        print("---------")
        print(outputs)
        print(outputs <= 0.5)


if __name__ == '__main__':
    main()
