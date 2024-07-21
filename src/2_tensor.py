import torch
import numpy as np

"""
[텐서(Tensor) 다루기]
- 텐서란, 데이터의 묶음입니다.
    1 혹은 2 와 같이 값이 하나로 존재하는 것을 1차원 텐서,
    
    (1, 2) 혹은 (4, 6) 과 같이 값이 두개로 존재하는 것은 2차원 텐서...
    
    이런식으로 늘어납니다.

    머신러닝, 딥러닝 분야에서는 대량의 데이터를 다루고 분석하게 되는데, 이때 텐서간의 연산이 수행되므로 딥러닝 라이브러리인
    torch 에서는 이러한 텐서를 다루는 방법을 기본으로 제공해주고, 이를 잘 다룰줄 알아야 합니다.

- 아래 코드는 Tensor 를 다루는 예시입니다.
"""


def main():
    # 파이썬에는 torch 가 나오기 이전에도 텐서를 쉽게 다룰 수 있도록 넘파이 라이브러리가 제공되어 사용되고 있었습니다.
    # 넘파이로 텐서 생성
    print("넘파이로 텐서 생성")
    t = np.array([0., 1., 2., 3., 4., 5., 6.])
    print(t)  # 출력 : [0. 1. 2. 3. 4. 5. 6.]
    print("")

    print("넘파이 텐서 속성 확인")
    print(t.ndim)  # 출력 : 1
    print(t.shape)  # 출력 : (7,)
    print("")

    print("넘파이 텐서 조회")
    # 인덱스를 통한 원소 접근
    print(t[0], t[1], t[-1])  # 출력 : 0.0 1.0 6.0
    # [시작 번호 : 끝 번호]로 범위 지정을 통해 가져온다.
    print(t[2:5], t[4:-1])  # 출력 : [2. 3. 4.] [4. 5.]
    # 시작 번호를 생략한 경우와 끝 번호를 생략한 경우
    print(t[:2], t[3:])  # 출력 : [0. 1.] [3. 4. 5. 6.]
    print("")

    """
    - 딥러닝 모델을 만들 때, 텐서를 다룰 때에는 torch 의 tensor 객체를 사용해야 합니다.
        앞서 넘파이로 다루는 텐서와는 달리, 
        딥러닝에 필요한 여러 기능을 제공해주고, GPU 연산까지 가능하게 해주는 PyTorch 의 기본 단위입니다.
    """
    # torch 텐서 생성
    print("torch 텐서 생성")
    # 출력 : tensor([1, 2, 3])
    print(torch.tensor([1, 2, 3]))
    # 출력 : tensor([[1., 2., 3.],
    #         [4., 5., 6.]])
    print(torch.Tensor([[1, 2, 3], [4, 5, 6]]))
    # 출력 : tensor([1, 2, 3])
    print(torch.LongTensor([1, 2, 3]))
    # 출력 : tensor([1., 2., 3.])
    print(torch.FloatTensor([1, 2, 3]))
    print("")

    # torch 텐서 속성 확인
    print("torch 텐서 속성 확인")
    tensor = torch.rand(1, 2)
    # 출력 : tensor([[0.6330, 0.6025]])
    print(tensor)
    # 출력 : 2
    print(tensor.dim())  # 차원
    # 출력 : torch.Size([1, 2])
    print(tensor.shape)  # 형태
    # 출력 : torch.Size([1, 2])
    print(tensor.size())  # 형태 (shape 와 동일)
    # 출력 : torch.float32
    print(tensor.dtype)  # 데이터 타입
    # 출력 : cpu
    print(tensor.device)  # 할당된 디바이스(gpu 라면 gpu 메모리, cpu 라면 일반 메모리)
    print("")

    print("torch 텐서 조회")
    t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
    # 출력 : tensor(0.) tensor(1.) tensor(6.)
    print(t[0], t[1], t[-1])  # 인덱스로 접근
    # 출력 : tensor([2., 3., 4.]) tensor([4., 5.])
    print(t[2:5], t[4:-1])  # 슬라이싱
    # 출력 : tensor([0., 1.]) tensor([3., 4., 5., 6.])
    print(t[:2], t[3:])  # 슬라이싱
    print("")

    print("torch 브로드 캐스팅 확인")
    m1 = torch.FloatTensor([[3, 3]])
    m2 = torch.FloatTensor([[2, 2]])
    # 출력 : tensor([[5., 5.]])
    print(m1 + m2)

    m1 = torch.FloatTensor([[1, 2]])
    m2 = torch.FloatTensor([3])  # [3] -> [3, 3]
    # 출력 : tensor([[4., 5.]])
    print(m1 + m2)

    # 2 x 1 Vector + 1 x 2 Vector
    m1 = torch.FloatTensor([[1, 2]])
    m2 = torch.FloatTensor([[3], [4]])  # [[3], [4]] -> [[3, 3], [4, 4]]
    # 출력 : tensor([[4., 5.],
    #         [5., 6.]])
    print(m1 + m2)
    print("")

    print("torch 텐서 계산")
    # 행렬곱 matmul
    m1 = torch.FloatTensor([[1, 2], [3, 4]])
    m2 = torch.FloatTensor([[1], [2]])
    print(m1.shape)  # 출력 : torch.Size([2, 2])
    print(m2.shape)  # 출력 : torch.Size([2, 1])
    # 출력 : tensor([[ 5.],
    #         [11.]])
    print(m1.matmul(m2))  # 2 x 1

    # 원소별 곱샘 mul
    m1 = torch.FloatTensor([[1, 2], [3, 4]])
    m2 = torch.FloatTensor([[1], [2]])
    print(m1.shape)  # 출력 : torch.Size([2, 2])
    print(m2.shape)  # 출력 : torch.Size([2, 1])
    # 출력 : tensor([[1., 2.],
    #         [6., 8.]])
    print(m1 * m2)  # 2 x 2
    # 출력 : tensor([[1., 2.],
    #         [6., 8.]])
    print(m1.mul(m2))

    # 평균
    t = torch.FloatTensor([1, 2])
    # 출력 : tensor(1.5000)
    print(t.mean())
    t = torch.FloatTensor([[1, 2], [3, 4]])
    # 출력 : tensor(2.5000)
    print(t.mean())
    # 출력 : tensor([2., 3.])
    print(t.mean(dim=0))
    # 출력 : tensor([1.5000, 3.5000])
    print(t.mean(dim=1))
    # 출력 : tensor([1.5000, 3.5000])
    print(t.mean(dim=-1))

    # 덧셈
    t = torch.FloatTensor([[1, 2], [3, 4]])
    # 출력 : tensor([[1., 2.],
    #         [3., 4.]])
    print(t)
    # 출력 : tensor(10.)
    print(t.sum())  # 단순히 원소 전체의 덧셈을 수행
    # 출력 : tensor([4., 6.])
    print(t.sum(dim=0))  # 행을 제거
    # 출력 : tensor([3., 7.])
    print(t.sum(dim=1))  # 열을 제거
    # 출력 : tensor([3., 7.])
    print(t.sum(dim=-1))  # 열을 제거

    # 최대
    t = torch.FloatTensor([[1, 2], [3, 4]])
    # 출력 : tensor([[1., 2.],
    #         [3., 4.]])
    print(t)
    # 출력 : tensor(4.)
    print(t.max())
    # 출력 : torch.return_types.max(
    # values=tensor([3., 4.]),
    # indices=tensor([1, 1]))
    print(t.max(dim=0))
    # 출력 : torch.return_types.max(
    # values=tensor([2., 4.]),
    # indices=tensor([1, 1]))
    print(t.max(dim=1))
    # 출력 : torch.return_types.max(
    # values=tensor([2., 4.]),
    # indices=tensor([1, 1]))
    print(t.max(dim=-1))
    print("")

    # torch 텐서 차원 변환
    print("torch 텐서 차원 변환")
    tensor = torch.rand(1, 2)
    # 출력 : tensor([[0.1613, 0.1578]])
    print(tensor)
    tensor = tensor.reshape(2, 1)
    # 출력 : tensor([[0.1613],
    #         [0.1578]])
    print(tensor)
    # 출력 : torch.Size([2, 1])
    print(tensor.shape)

    # 3 차원 데이터 준비
    t = np.array([[[0, 1, 2],
                   [3, 4, 5]],
                  [[6, 7, 8],
                   [9, 10, 11]]])
    ft = torch.FloatTensor(t)
    # 출력 : torch.Size([2, 2, 3])
    print(ft.shape)
    # 출력 : tensor([[ 0.,  1.,  2.],
    #         [ 3.,  4.,  5.],
    #         [ 6.,  7.,  8.],
    #         [ 9., 10., 11.]])
    print(ft.view([-1, 3]))  # ft라는 텐서를 (?, 3)의 크기로 변경
    # (2, 2, 3) -> (2 × 2, 3) -> (4, 3)
    # 출력 : torch.Size([4, 3])
    print(ft.view([-1, 3]).shape)
    # 출력 : tensor([[[ 0.,  1.,  2.]],
    #         [[ 3.,  4.,  5.]],
    #         [[ 6.,  7.,  8.]],
    #         [[ 9., 10., 11.]]])
    print(ft.view([-1, 1, 3]))
    # 출력 : torch.Size([4, 1, 3])
    print(ft.view([-1, 1, 3]).shape)

    # 차원 축소(Squeeze)
    ft = torch.FloatTensor([[0], [1], [2]])
    # 출력 : tensor([[0.],
    #         [1.],
    #         [2.]])
    print(ft)
    # 출력 : torch.Size([3, 1])
    print(ft.shape)
    # 출력 : tensor([0., 1., 2.])
    print(ft.squeeze())
    # 출력 : torch.Size([3])
    print(ft.squeeze().shape)

    # 차원 확장(UnSqueeze)
    ft = torch.Tensor([0, 1, 2])
    # 출력 : torch.Size([3])
    print(ft.shape)
    # 아래 코드는
    # print(ft.view(1, -1))
    # print(ft.view(1, -1).shape)
    # 이렇게도 구현이 가능합니다.
    # 출력 : tensor([[0., 1., 2.]])
    print(ft.unsqueeze(0))  # 인덱스가 0부터 시작하므로 0은 첫번째 차원을 의미한다.
    # 출력 : torch.Size([1, 3])
    print(ft.unsqueeze(0).shape)
    # 출력 : tensor([[0.],
    #         [1.],
    #         [2.]])
    print(ft.unsqueeze(1))  # 인덱스가 0부터 시작하므로 1은 두번째 차원을 의미한다.(현재 상태에선 마지막 인덱스를 뜻하는 -1 을 넣어도 동일)
    # 출력 : torch.Size([3, 1])
    print(ft.unsqueeze(1).shape)
    print("")

    # torch 텐서 자료형 설정
    print("torch 텐서 자료형 설정")
    # 출력 : tensor([[0.6249, 0.7556, 0.5049],
    #         [0.1579, 0.0148, 0.5885],
    #         [0.9484, 0.2258, 0.0255]])
    tensor = torch.rand((3, 3), dtype=torch.float)
    print(tensor)

    lt = torch.LongTensor([1, 2, 3, 4])
    # 출력 : tensor([1, 2, 3, 4])
    print(lt)
    # 출력 : tensor([1., 2., 3., 4.])
    print(lt.float())  # 타입 변환

    bt = torch.ByteTensor([True, False, False, True])
    # 출력 : tensor([1, 0, 0, 1], dtype=torch.uint8)
    print(bt)
    # 출력 : tensor([1, 0, 0, 1])
    print(bt.long())
    # 출력 : tensor([1., 0., 0., 1.])
    print(bt.float())

    # 연결 (concatenate)
    x = torch.FloatTensor([[1, 2], [3, 4]])
    y = torch.FloatTensor([[5, 6], [7, 8]])
    # 출력 : tensor([[1., 2.],
    #         [3., 4.],
    #         [5., 6.],
    #         [7., 8.]])
    print(torch.cat([x, y], dim=0))
    # 출력 : tensor([[1., 2., 5., 6.],
    #         [3., 4., 7., 8.]])
    print(torch.cat([x, y], dim=1))

    # 스택킹(Stacking)
    x = torch.FloatTensor([1, 4])
    y = torch.FloatTensor([2, 5])
    z = torch.FloatTensor([3, 6])
    # 아래 코드는
    # print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))
    # 이것과 동일합니다.
    # 출력 : tensor([[1., 4.],
    #         [2., 5.],
    #         [3., 6.]])
    print(torch.stack([x, y, z]))
    print("")

    # 같은 형태 텐서 생성
    print("같은 형태 텐서 생성")
    x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
    # 출력 : tensor([[0., 1., 2.],
    #         [2., 1., 0.]])
    print(x)
    # 출력 : tensor([[1., 1., 1.],
    #         [1., 1., 1.]])
    print(torch.ones_like(x))  # 입력 텐서와 크기를 동일하게 하면서 값을 1로 채우기
    # 출력 : tensor([[0., 0., 0.],
    #         [0., 0., 0.]])
    print(torch.zeros_like(x))  # 입력 텐서와 크기를 동일하게 하면서 값을 0으로 채우기
    print("")

    # 덮어쓰기 연산
    # mul 과 같은 계산을 하면 결과값이 새로 생성되어 반환되는 것인데,
    # 결과값을 입력 데이터에 바로 저장하려면 연산 함수명 뒤에 _ 를 붙이면 됩니다.
    print("덮어쓰기 연산")
    x = torch.FloatTensor([[1, 2], [3, 4]])
    # 출력 : tensor([[2., 4.],
    #         [6., 8.]])
    print(x.mul(2.))  # 곱하기 2를 수행한 결과를 출력
    # 출력 : tensor([[1., 2.],
    #         [3., 4.]])
    print(x)  # 기존의 값 출력
    # 덮어쓰기 연산 _
    # 출력 : tensor([[2., 4.],
    #         [6., 8.]])
    print(x.mul_(2.))  # 곱하기 2를 수행한 결과를 변수 x에 값을 저장하면서 결과를 출력
    # 출력 : tensor([[2., 4.],
    #         [6., 8.]])
    print(x)  # 기존의 값 출력
    print("")

    # 텐서 GPU 장치 설정 cuda
    print("torch 텐서 자료형 설정 cuda")
    if torch.cuda.is_available():
        device = "cuda"
        tensor = torch.tensor(data=[1, 2, 3], dtype=torch.float32, device="cuda")
    else:
        device = "cpu"
        tensor = torch.FloatTensor([1, 2, 3])
    print(device)
    print(tensor)
    tensor = torch.rand((1, 1), device=device)
    print(tensor)
    print("")

    # 텐서 GPU 장치 설정 mps
    print("torch 텐서 자료형 설정 mps")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"
    print(device)
    tensor = torch.FloatTensor([1, 2, 3], device=device)
    print(tensor)
    print("")

    # 텐서 장치 변환 cuda
    if torch.cuda.is_available():
        print("torch 텐서 장치 변환 cuda")
        cpu = torch.FloatTensor([1, 2, 3])
        print(cpu)
        gpu = cpu.cuda()
        print(gpu)
        gpu2cpu = gpu.cpu()
        print(gpu2cpu)
        cpu2gpu = cpu.to("cuda")
        print(cpu2gpu)
        print("")

    # 텐서 장치 변환 mps
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("torch 텐서 장치 변환 mps")
        cpu = torch.FloatTensor([1, 2, 3])
        print(cpu)
        gpu = cpu.to("mps")
        print(gpu)
        gpu2cpu = gpu.cpu()
        print(gpu2cpu)
        print("")

    # 넘파이 배열을 텐서로
    print("넘파이 배열을 텐서로")
    ndarray = np.array([1, 2, 3], dtype=np.uint8)
    print(torch.tensor(ndarray))
    print(torch.Tensor(ndarray))
    print(torch.from_numpy(ndarray))
    print("")

    # 텐서를 넘파이 배열로
    print("텐서를 넘파이 배열로")
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"

    tensor = torch.FloatTensor([1, 2, 3]).to(device)
    ndarray = tensor.detach().cpu().numpy()
    print(ndarray)
    print(type(ndarray))
    print("")

    """
    추가 : 데이터를 다룰 때, 데이터의 양이 너무 많으면 한꺼번에 메모리에 들고 가서 처리를 하기에는 메모리 용량이 작을 수도 있습니다.
    이때, 데이터를 한꺼번에 들고가서 처리하는 것을 Batch 작업이라고 하며, 그 데이터 크기를 Batch Size 라고 합니다.
    """


if __name__ == '__main__':
    main()
