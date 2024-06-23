"""
[ELMo(Embeddings from Language Model)]
- 논문 링크 : https://aclweb.org/anthology/N18-1202
    ELMo(Embeddings from Language Model)는 2018년에 제안된 새로운 워드 임베딩 방법론입니다.
    ELMo라는 이름은 세서미 스트리트라는 미국 인형극의 케릭터 이름이기도 한데,
    뒤에서 배우게 되는 BERT나 최근 마이크로소프트가 사용한 Big Bird라는 NLP 모델 또한 ELMo에 이어 세서미 스트리트의 케릭터의 이름을 사용했습니다.
    ELMo는 Embeddings from Language Model의 약자입니다. 해석하면 '언어 모델로 하는 임베딩'입니다.
    ELMo의 가장 큰 특징은 사전 훈련된 언어 모델(Pre-trained language model)을 사용한다는 점입니다.
    이는 ELMo의 이름에 LM이 들어간 이유입니다.

- Bank라는 단어를 생각해봅시다. Bank Account(은행 계좌)와 River Bank(강둑)에서의 Bank는 전혀 다른 의미를 가지는데,
    Word2Vec이나 GloVe 등으로 표현된 임베딩 벡터들은 이를 제대로 반영하지 못한다는 단점이 있습니다.
    예를 들어서 Word2Vec이나 GloVe 등의 임베딩 방법론으로 Bank란 단어를 [0.2 0.8 -1.2]라는 임베딩 벡터로 임베딩하였다고 하면,
    이 단어는 Bank Account(은행 계좌)와 River Bank(강둑)에서의 Bank는 전혀 다른 의미임에도 불구하고 두 가지 상황 모두에서 [0.2 0.8 -1.2]의 벡터가 사용됩니다.
    같은 표기의 단어라도 문맥에 따라서 다르게 워드 임베딩을 할 수 있으면 자연어 처리의 성능을 올릴 수 있을 것입니다.
    워드 임베딩 시 문맥을 고려해서 임베딩을 하겠다는 아이디어가 문맥을 반영한 워드 임베딩(Contextualized Word Embedding) 입니다.

- biLM(Bidirectional Language Model)의 사전 훈련
    다음 단어를 예측하는 작업인 언어 모델링을 상기해봅시다.
    RNN 언어 모델은 문장으로부터 단어 단위로 입력을 받는데, RNN 내부의 은닉 상태 h는 시점(time step)이 지날수록 점점 업데이트되갑니다.
    이는 결과적으로 RNN의 h의 값이 문장의 문맥 정보를 점차적으로 반영한다고 말할 수 있습니다.
    그런데 ELMo는 순방향 RNN 뿐만 아니라, 반대 방향으로 문장을 스캔하는 역방향 RNN 또한 활용합니다.
    ELMo는 양쪽 방향의 언어 모델을 둘 다 학습하여 활용한다고하여 이 언어 모델을 biLM(Bidirectional Language Model) 이라고 합니다.

    ELMo에서 말하는 biLM은 기본적으로 다층 구조(Multi-layer)를 전제로 합니다.
    은닉층이 최소 2개 이상이라는 의미입니다.

    이때 biLM의 각 시점의 입력이 되는 단어 벡터는 임베딩 층(embedding layer)을 사용해서 얻은 것이 아니라
    합성곱 신경망을 이용한 문자 임베딩(character embedding)을 통해 얻은 단어 벡터입니다.
    문자 임베딩은 마치 서브단어(subword)의 정보를 참고하는 것처럼 문맥과 상관없이 dog란 단어와 doggy란 단어의 연관성을 찾아낼 수 있습니다.
    또한 이 방법은 OOV에도 견고한다는 장점이 있습니다.

    주의할 점은 앞서 설명한 양방향 RNN과 ELMo에서의 biLM은 다릅니다.
    양방향 RNN은 순방향 RNN의 은닉 상태와 역방향의 RNN의 은닉 상태를 연결(concatenate)하여 다음층의 입력으로 사용합니다.
    반면, biLM의 순방향 언어모델과 역방향 언어모델이라는 두 개의 언어 모델을 별개의 모델로 보고 학습합니다.

- https://wikidocs.net/217189
"""
