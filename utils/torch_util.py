import torch
import pandas as pd
from torch.utils.data import Dataset, random_split
from datetime import datetime
import os


# (GPU 디바이스 값 반환)
def get_gpu_support_device(
        # GPU 지원 설정 (True 로 설정 해도 현재 디바이스가 GPU 를 지원 하지 않으면 CPU 를 사용 합니다.)
        gpu_support
):
    if gpu_support:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = "cpu"

    print(f"Device Selected : {device}")
    return device


# (모델 파일 저장)
def save_model_file(
        # 파일로 저장할 모델 객체
        model,
        # 생성된 모델 파일을 저장할 폴더 위치 (ex : "../_by_product_files/torch_model_files")
        model_file_save_directory_path
):
    save_file_full_path = f"{model_file_save_directory_path}/model({datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]}).pt"
    torch.save(
        model,
        save_file_full_path
    )

    print(f"Model File Saved : {save_file_full_path}")
    return save_file_full_path


# (모델 사용 데이터 셋)
class CsvModelDataset(Dataset):
    def __init__(
            self,
            # 데이터를 읽어올 CSV 파일 경로 - 1 행에 라벨이 존재하고, 그 라벨로 x, y 데이터를 분류 합니다. (ex : "../datasets/linear.csv")
            csv_file_full_url,
            # 독립 변수로 사용할 컬럼의 라벨명 리스트 (ex : ['x1', 'x2'])
            x_column_labels,
            # 종속 변수로 사용할 컬럼의 라벨명 리스트 (ex : ['y1'])
            y_column_labels
    ):
        # CSV 에서 데이터 가져오기 (ex : [{x : [x1, x2], y : [y1]}, {x : [x1, x2], y : [y1]}, ...])
        # CSV 파일 읽기
        data_frame = pd.read_csv(csv_file_full_url)

        # # 데이터 프레임을 변환 (ex : [{x : [x1, x2], y : [y1]}, {x : [x1, x2], y : [y1]}, ...])
        self.data = [
            {
                "x": [item[column] for column in x_column_labels],
                "y": [item[y_column] for y_column in y_column_labels]
            }
            for item in data_frame.to_dict(orient='records')
        ]
        self.length = len(self.data)

    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]["x"]), torch.FloatTensor(self.data[index]["y"])

    def __len__(self):
        return self.length


# (Dataset 분리)
def split_dataset(
        # 분리할 데이터 셋
        dataset,
        # 학습 데이터 비율 (ex : 0.8)
        train_data_rate,
        # 검증 데이터 비율 (ex : 0.2)
        validation_data_rate
):
    # rate 파라미터들의 합이 1인지 확인
    total_rate = train_data_rate + validation_data_rate
    assert total_rate == 1.0, f"Data split rates do not add up to 1.0 (current total: {total_rate})"

    # 전체 데이터 사이즈
    dataset_size = len(dataset)
    print(f"Total Data Size : {dataset_size}")

    # 목적에 따라 데이터 분리
    train_size = int(dataset_size * train_data_rate)  # 학습 데이터
    validation_size = int(dataset_size * validation_data_rate)  # 검증 데이터

    # 학습, 검증, 테스트 데이터를 무작위로 나누기
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    print(f"Training Data Size : {len(train_dataset)}")
    print(f"Validation Data Size : {len(validation_dataset)}")

    return train_dataset, validation_dataset


# (모델 학습)
def train_model(
        # 사용할 디바이스
        device,
        # 학습할 모델
        model,
        # 손실 함수 (ex : nn.MSELoss())
        criterion,
        # 옵티마이저 (ex : optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay))
        optimizer,
        # 학습 데이터 로더
        train_dataloader,
        num_epochs=1000,  # 학습 에포크 수
        validation_dataloader=None,  # 검증 데이터 로더(None 이라면 검증은 하지 않음)
        # 체크포인트 파일 저장 폴더 경로 - None 이라면 저장하지 않음 (ex : "../check_points")
        check_point_file_save_directory_path=None,
        # 불러올 체크포인트 파일 경로 - None 이라면 불러 오지 않음 (ex : "../check_points/checkpoint(2024_02_29_17_51_09_330).pt")
        check_point_load_file_full_path=None,
        # 그래디언트 클리핑 기울기 최대 값 (ex : 1)
        # 그래디언트 최대값을 설정하여, 그래디언트 폭주를 막아 오버피팅을 억제합니다.
        # RNN 등 기울기가 폭주될 가능성이 있는 모델에 적용 하세요.
        grad_clip_max_norm=None,
        # 로그를 몇 에폭 만에 한번 실행 할지 여부 (0 이하는 검증도, 로깅도 하지 않음)
        log_freq=1000
):
    # 학습 시작 시간 기록
    start_time = datetime.now()

    print("Model Training Start!")
    # 모델 디바이스 설정
    model.to(device)

    # 손실 함수 디바이스 설정
    criterion.to(device)

    # 모델에 학습 모드 설정 (Dropout, Batchnorm 등의 기능 활성화)
    model.train()

    # 체크포인트 불러오기
    if check_point_load_file_full_path is not None:
        checkpoint = torch.load(check_point_load_file_full_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Check Point Loaded")
        print(f"ckp_description : {checkpoint['description']}")

    # 학습 루프
    for training_epoch in range(num_epochs):
        training_loss = 0.0
        training_correct = 0  # 올바르게 예측된 개수 초기화

        # 배치 학습
        for tx, ty in train_dataloader:
            tx = tx.to(device)
            ty = ty.to(device)

            # 모델 순전파
            model_out = model(tx)

            # 모델 결과물 loss 계산
            loss = criterion(model_out, ty)

            # 옵티마이저 기울기 초기화
            optimizer.zero_grad()
            # loss 에 따른 신경망 역전파 후 기울기 계산
            loss.backward()

            # 그래디언트 클리핑
            if grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)

            # 신경망 학습
            optimizer.step()

            training_loss += loss
            training_correct += (torch.argmax(model_out, dim=1) == ty).sum().item()

        training_loss = training_loss / len(train_dataloader)

        # 1000 에 한번 실행
        if log_freq > 0 and (training_epoch + 1) % log_freq == 0:
            # 학습 시작부터 얼마나 시간이 지났는지 계산
            elapsed_time = datetime.now() - start_time
            validation_loss_string = "None"
            validation_loss = 0
            if validation_dataloader is not None:
                # 검증 계산
                model.eval()  # Dropout, Batchnorm 등의 기능 비활성화
                with torch.no_grad():  # Gradient 추적 계산을 하지 않음
                    for vx, vy in validation_dataloader:
                        vx = vx.to(device)
                        vy = vy.to(device)
                        model_out = model(vx)

                        loss = criterion(model_out, vy)
                        validation_loss += loss
                model.train()
                validation_loss = validation_loss / len(validation_dataloader)
                validation_loss_string = f"{validation_loss:.3f}"

            tl_vl_str = "None"
            if validation_loss_string != "None":
                tl_vl_str = f"{abs(training_loss - float(validation_loss)):.3f}"

            print(
                f"\nTrainingEpoch : {training_epoch + 1:4d},\n"
                f"TrainingLoss : {training_loss:.3f},\n"
                f"ValidationLoss : {validation_loss_string},\n"
                f"TL-VL ABS : {tl_vl_str}\n"
                f"Elapsed Time: {elapsed_time}"
            )

            if check_point_file_save_directory_path is not None:
                if not os.path.exists(check_point_file_save_directory_path):
                    os.makedirs(check_point_file_save_directory_path)

                # 학습 체크포인트 저장
                current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
                check_point_file_name = f"checkpoint({current_time}).pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "training_loss": training_loss,
                        "validation_loss": validation_loss_string,
                        "model": "CustomModel",
                        "description": f"CustomModel 체크포인트-{current_time}"
                    },
                    f"{check_point_file_save_directory_path}/{check_point_file_name}",
                )
                print(f"CheckPoint File Saved : {check_point_file_name}")

    print("\nModel Training Complete!")
