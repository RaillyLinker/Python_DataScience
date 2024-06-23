import torch
from torch import nn
from torch.utils.data import DataLoader
import utils.torch_util as tu
import model_layers.linear_regression.main_model as linear_regression
import os
from torch import optim

"""
[선형 회귀 모델 - 토치 모델]
- 아래 코드는 선형 회귀 모델을 본격적으로 토치 모델로 만들어 사용 하는 예시를 보여 줍니다.
    커스텀 뉴럴 네트워크 클래스를 작성하고,
    학습 데이터는 외부 데이터 파일에서 받아와 사용합니다.
    학습시에는 학습, 검증 및 테스트를 수행하며, 마지막으로 모델을 저장하거나 불러오는 등의 예시를 포함합니다.
    모델 생성, 학습, 검증의 정형화된 부분을 템플릿화 하였으므로 이를 참고하여 앞으로 모델을 작성하면 됩니다.
"""


def main():
    # 사용 가능 디바이스
    device = tu.get_gpu_support_device(gpu_support=True)

    # 데이터셋 객체 생성 (ex : tensor([[-10., 100., 82.], ...], device = cpu), tensor([[327.7900], ...], device = cpu))
    # CSV 파일로 데이터셋 형성 (1 행에 라벨이 존재하고, 그 라벨로 x, y 데이터를 분류 합니다.)
    dataset = tu.CsvModelDataset(
        csv_file_full_url="../resources/datasets/11_linear_regression_nn(basic_nn_form)/linear.csv",
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
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=10, shuffle=True, drop_last=True)

    # 모델 생성
    model = linear_regression.MainModel()

    # 모델 학습
    tu.train_model(
        device=device,
        model=model,
        criterion=nn.MSELoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.001),
        train_dataloader=train_dataloader,
        num_epochs=10000,
        validation_dataloader=validation_dataloader,
        check_point_file_save_directory_path="../_by_product_files/check_point_files/linear_regression",
        # check_point_load_file_full_path="../_by_product_files/check_point_files/~/checkpoint(2024_02_29_17_51_09_330).pt",
        log_freq=1000
    )

    # 모델 저장
    model_file_save_directory_path = "../_by_product_files/torch_model_files/linear_regression"
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

    # 불러온 모델 순전파
    with torch.no_grad():
        model.eval()
        inputs = torch.FloatTensor(
            [
                [1 ** 2, 1],
                [5 ** 2, 5],
                [11 ** 2, 11]
            ]
        ).to(device)
        outputs = model(inputs)
        print(outputs)


if __name__ == '__main__':
    main()
