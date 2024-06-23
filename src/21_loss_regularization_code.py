"""
[손실 규제]
- 먼저, 단어부터 정리합시다.
    Regularization를 정규화로 번역하기도 하지만,
    이는 보다 보편적으로 정규화로 사용되는 Normalization 과 혼동될 수 있으므로 규제 또는 정형화라는 번역을 사용하는데,
    저는 규제라고 하겠습니다.

- 복잡한 모델이 간단한 모델보다 과적합될 가능성이 높습니다.
    간단한 모델은 적은 수의 매개변수를 가진 모델을 말합니다.
    복잡한 모델을 좀 더 간단하게 하는 방법으로 가중치 규제(Regularizaiton)가 있습니다.

    손실 규제에는 L1, L2 규제가 있습니다.

    L1 규제는 L1 norm 이라는 값을 비용 함수에 추가하고,
    L2 규제는 L2 norm 을 비용 함수에 추가합니다.

    전체 코드는 아래에 적기로 하고,

    L1 norm 은 아래와 같습니다.

    l1_norm = sum(p.abs().sum() for p in model.parameters())

    보다시피 모델의 모든 파라미터의 절대값을 더한 값이 L1 norm 입니다.

    이것을 손실값에 더해준다는 것은,
    손실 값을 최소화 하는 목적을 가진 오차 역전파 프로세스에 있어서는,

    비용 함수를 최소화하기 위해서는 가중치 w들의 값이 작아져야 하는 방향으로 학습을 유도하게 되는 것입니다.

    계속해서 L1 규제로 예를 들어봅시다.
    L1 규제를 사용하면 비용 함수가 최소가 되게 하는 가중치와 편향을 찾는 동시에 가중치들의 절대값의 합도 최소가 되어야 합니다.
    이렇게 되면, 가중치 w의 값들은 0 또는 0에 가까이 작아져야 하므로 어떤 특성들은 모델을 만들 때 거의 사용되지 않게 됩니다.

    바로 그것입니다.
    필요치 않는 특성을 비활성화 시킴으로써 모델의 구조를 단순화 시키는 것과 같은 효과를 노리는 것입니다.

    예를 들어
    H(x) = w1x1 + w2x2 + w3x3 + w4x4 라는 수식이 있다고 해봅시다.
    여기에 L1 규제를 사용하였더니, w3 의 값이 0이 되었다고 해봅시다.
    이는 x3 특성은 사실 모델의 결과에 별 영향을 주지 못하는 특성임을 의미합니다.

    이렇게 된다면 특성 4개인 모델에 사용되는 특성이 하나를 줄이게 되어 특성 3개로 모델을 단순화 시키는 것이겠죠.

    L2 규제는 L1 규제와는 달리 가중치들의 제곱을 최소화합니다.
    그 결과, w의 값이 완전히 0이 되기보다는 0에 가까워지기는 경향을 띕니다. (특성이 완전히 죽지 않게 만듬)
    L1 규제는 어떤 특성들이 모델에 영향을 주고 있는지를 정확히 판단하고자 할 때 유용합니다.

    만약, 이런 판단이 필요없다면 경험적으로는 L2 규제가 더 잘 동작하므로 L2 규제를 더 권장합니다.
    인공 신경망에서 L2 규제는 가중치 감쇠(weight decay)라고도 부릅니다.

    파이토치에서는 옵티마이저의 weight_decay 매개변수를 설정하므로서 L2 규제를 적용합니다.
    weight_decay 매개변수의 기본값은 0입니다. weight_decay 매개변수에 다른 값을 설정할 수도 있습니다.
"""


# L1 규제 (L1 Regularization) 구현
def l1_regularization(train_dataloader, device, model, criterion):
    for x, y in train_dataloader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)

        # L1 Norm 적용 비율 (0.001 같은 작은 값을 사용하는 것이 일반적입니다.)
        _lambda = 0.5
        # L1 Norm : 모델 파라미터에 abs 적용 후 더하기
        l1_norm = sum(p.abs().sum() for p in model.parameters())

        # L1 규제를 적용하려면, l1_norm * _lambda 값을 손실값에 더해주면 됩니다.
        loss = criterion(output, y) + _lambda * l1_norm


# L2 규제 (L2 Regularization) 구현
def l2_regularization(train_dataloader, device, model, criterion):
    for x, y in train_dataloader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)

        # L2 Norm 적용 비율 (0.001 같은 작은 값을 사용하는 것이 일반적입니다.)
        _lambda = 0.5
        # L2 Norm : 모델 파라미터를 제곱 후 더하기
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())

        # L2 규제를 적용하려면, l2_norm * _lambda 값을 손실값에 더해주면 됩니다.
        loss = criterion(output, y) + _lambda * l2_norm
