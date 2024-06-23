import collections
import re

"""
[하위 단어 토크나이징]
인간이 이해하는 방식으로 자연어 데이터 분석을 하기 위해서는 공백 단위 텍스트 분리보다는 형태소 단위의 텍스트 분리가 효과적일 것입니다.
하지만 언어라는 것은 형태소일지라도 시간이 지남에 따라 생성, 소멸, 수정의 변화가 있을 수 있습니다.
또한 휴먼 에러로 인하여 오탈자로 인한 오류가 있을 수도 있죠.
즉, 단순한 형태소 분석은 신조어, 외래어, 전문용어, 고유어 등을 처리할 때 약점을 보입니다.
이렇게 정적인 형태소 분리에서 만들어진 단어 사전(Vocabulary) 에 없는 단어가 입력 되었을 때를 OOV(Out Of Vocabulary), 혹은 UNK(Unknown Token) 라고 부릅니다.
자연어 처리에서는 이러한 OOV 를 고려해야만 하는 것입니다.
이를 해결하기 위한 방법 중 하나는 분리된 단어를 더 작은 단위로 쪼개는 방법이 있습니다.
예를들어 영어의 birthplace 는 birth + place 로 나눌 수 있고, 그렇게 나누어진 단어들의 조합으로 실제 의미를 알 수 있습니다.
한국어로 하자면 수영장은 수영 + 장, 안경태의 경우는 안경 + 태...
이와같이 단어의 합으로 이루어진 단어가 많이 있으며, 사전에는 없는 신조어가 입력되더라도 이러한 작업으로 어느정도 그 의미를 유추할 수 있는 것이죠.
이것을 하위 단어 토큰화라고 합니다.
하위 단어 토큰화에는 두가지 방식이 존재합니다.
"""

"""
[바이트페어 인코딩]
바이트페어 인코딩 (Byte Pair Encoding : BPE) 란, 다이어그램 코딩(Diagram Coding) 이라고도 합니다.
1994년에 제안된 데이터 압축 알고리즘으로 시작된 방식이지만, 지금은 이처럼 자연어 처리(NLP) 분야, 
특히 텍스트 또는 언어 모델링(GPT-3, GPT-4, etc...)에서 토큰화(tokenization) 기법으로 널리 사용되고 있습니다.

이 방식의 기본 아이디어는 말뭉치(corpus) 내에서 가장 빈번하게 등장하는 바이트 쌍(혹은 문자 쌍)을 반복적으로 합치면서 새로운 단어 또는 토큰을 생성하는 것입니다.
이렇게 함으로써, 모델이 처리해야 할 토큰의 수를 줄이고, 언어의 다양한 변형(variation)을 더 잘 핸들링할 수 있게 됩니다.

예시로 설명하겠습니다.

아래와 같은 문자열이 주어졌을 때 BPE을 수행한다고 해봅시다.

aaabdaaabac

BPE은 기본적으로 연속적으로 가장 많이 등장한 글자의 쌍을 찾아서 하나의 글자로 병합하는 방식을 수행합니다. 
태생이 압축 알고리즘인 만큼, 여기서는 글자 대신 바이트(byte)라는 표현을 사용하겠습니다. 
예를 들어 위의 문자열 중 가장 자주 등장하고 있는 바이트의 쌍(byte pair)은 'aa'입니다. 
이 'aa'라는 바이트의 쌍을 하나의 바이트인 'Z'로 치환해보겠습니다.

ZabdZabac
Z=aa

위와 같이 두개씩 가장 많이 나타나는(그리고 가장 처음 발견된) aa 를 Z 로 치환하고, 이 Z 에 aa 가 할당되어 있다는 것을 기록해둡니다.
그리고 다음으로 위 문자열 중에서 가장 많이 등장하고 있는 바이트의 쌍은 'ab'입니다. 이 'ab'를 'Y'로 치환해봅시다.

ZYdZYac
Y=ab
Z=aa

자, 다시 해봅시다. 치환된 글자 역시 압축 대상에서 빠질 수가 없습니다.
다음으로 가장 많이 등장하고 있는 바이트의 쌍은 'ZY'입니다. 이를 'X'로 치환해봅시다.

XdXac
X=ZY
Y=ab
Z=aa

이제 더 이상 병합할 바이트의 쌍은 없으므로 BPE는 위의 결과를 최종 결과로 하여 종료됩니다.

위와 같이 문자를 치환하는 과정에서 단어 사전을 만들 수 있으므로 자연어 처리에 사용 할 수 있는 것입니다.

BPE을 요약하면, 글자(charcter) 단위에서 점차적으로 단어 집합(vocabulary)을 만들어 내는 Bottom up 방식의 접근을 사용합니다.
고로 실제로 의미를 지닌 단어를 분리한다고 단언할 수는 없지만, 많은 데이터에서 이처럼 단어를 뽑아내다보면 확률상 발생 빈도가 높은 글자들의 조합이
생겨나게 될 것이므로 신뢰성이 있는 방식입니다.

이러한 방식을 통해 모델은 미처 학습하지 못한 단어들도 서브워드 또는 문자의 조합으로 표현하고 이해할 수 있게 됩니다.
예를 들어, 'un-'와 같은 접두사나 '-ing'과 같은 접미사를 포함한 단어들은 학습 과정에서 직접 마주치지 않았더라도,
이들의 구성 요소를 통해 의미를 유추할 수 있습니다.
BPE는 따라서 언어 모델이 새로운 단어나 희귀 단어에 대해 더 나은 일반화(generalization) 능력을 가지도록 돕습니다.

(BPE 의 구현)
먼저, 말뭉치에서 단어를 추려옵니다.

# dictionary
l o w : 5,  l o w e r : 2,  n e w e s t : 6,  w i d e s t : 3

위와 같이 말뭉치 안에서 찾아낸 단어와, 해당 단어가 몇번 사용되었는지에 대한 카운트를 준비해둡니다.
low 가 5 번, lower 가 2번... 이렇게 준비하고, 글자 단위로 나누어(바이트 단위로) 준비하세요.

이때는 글자 단위로 나뉜 것이므로 중첩되지 않는 글자들을 모두 사전으로 등록하면 됩니다.

l, o, w, e, r, n, s, t, i, d

이렇게 초기 사전이 형성 되네요.

자, 이제부터는 위에서 간단히 설명한대로 탐색 및 치환을 반복하면 됩니다.

먼저 위 데이터에서 탐색을 해봅시다.

2개씩 탐색을 하니, e s 가 빈도가 가장 많네요.
newest 가 말뭉치에서 6번, widest 가 3 번 탐색되었으니 합쳐서 9번 발생한 것이 됩니다.

그러면 이를 치환합시다.

이때, 압축을 위한 것이 아니라 하위 단어를 선정하기 위한 것이므로,
e, s 가 아니라 es 로 합쳐서

l o w : 5,
l o w e r : 2,
n e w es t : 6,
w i d es t : 3

이렇게 됩니다.
이제 기존 사전에서 es 가 합쳐져서,

l, o, w, e, r, n, s, t, i, d, es

사전은 위와 같이 될 것입니다.

자, 한번 탐색 및 치환을 했습니다.
이제 다음 탐색을 해봅시다.

다음에는 치환된 es 와 t 가 쌍으로 하여 빈도가 9번으로 가장 많네요.

l o w : 5,
l o w e r : 2,
n e w est : 6,
w i d est : 3

l, o, w, e, r, n, s, t, i, d, es, est

위와 같이 치환됩니다.

다음으로 빈도가 많은 쌍은 또 뭘까요?

l 과 o 입니다.

lo w : 5,
lo w e r : 2,
n e w est : 6,
w i d est : 3

l, o, w, e, r, n, s, t, i, d, es, est, lo

이러합니다.

이렇게 반복하여 10번을 실행하였다고 한다면,

low : 5,
low e r : 2,
newest : 6,
widest : 3

l, o, w, e, r, n, s, t, i, d, es, est, lo, low, ne, new, newest, wi, wid, widest

이렇게 되네요.

인코딩의 경우는

위와 같은 단어사전을 기반으로, 새로 들어온 단어를 글자단위로 쪼개서 위 사전에 대입하여 서브 워드 인코딩을 하면 됩니다.

자, 보다시피 글자에서 시작하여 단어가 합성되어가며 단어 사전을 만들 수 있음을 알 수 있습니다.

최소 글자단위로 줄어들기 때문에, OOV 를 줄일 수 있으며, 단어 그 자체로서의 의미 역시(의미를 정확히 지정하지 않아도) 지켜질 수 있습니다.

실제 구현은 아래와 같습니다.
"""


# BPE 구현
class BytePairEncoder:
    def __init__(self):
        super().__init__()
        # 어느 시점에 어느 pair 가 merge 된 지에 대한 history dict
        # ex : {('e', 's'): 1, ('es', 't'): 2, ('est', '</w>'): 3, ('l', 'o'): 4, ...}
        self.bpe_merge_history = {}

        # 합쳐진 단어가 무슨 pair 로 합쳤는지에 대한 history dict
        # ex : {'es': ('e', 's'), 'est': ('es', 't'), 'est</w>': ('est', '</w>'), ...}
        self.bpe_merge_word_dict = {}

    def bpe_train(
            self,
            # 단어 / 빈도 사전 (ex : {'l o w </w>': 5, ...})
            word_count_dict
    ):
        # num_merges 에 설정된 횟수 만큼 탐색 -> 치환 반복
        merge_count = 1
        while True:
            pairs = collections.defaultdict(int)
            # 단어 / 빈도 사전을 순회
            for word, freq in word_count_dict.items():
                # 단어를 글자 단위로 쪼개기
                symbols = word.split()
                # 쪼개진 글자를 순회(pair 이므로 끝부분에서 -1 까지)
                for symbol_idx in range(len(symbols) - 1):
                    # 기준이 되는 현재 인덱스 단어와, 현재 인덱스 단어의 쌍을 입력 후 freq 더하기
                    # {('l', 'o'): 7, ...}
                    pairs[symbols[symbol_idx], symbols[symbol_idx + 1]] += freq

            # 현 상태에서 word_count_dict 에 존재하는 pair / freq dict 가 준비된 상황
            if pairs.__len__() == 0:
                # 탐색이 끝남
                break

            # pairs 에서 가장 freq 가 많은 pair 를 선정
            most_freq_pair = max(pairs, key=pairs.get)

            # word_count_dict 에서 most_freq_pair 에 해당하는 글자를 제거하고, pair 를 합친 글자를 치환하기
            v_out = {}
            p = re.compile(r'(?<!\S)' + re.escape(' '.join(most_freq_pair)) + r'(?!\S)')
            for word in word_count_dict:
                w_out = p.sub(''.join(most_freq_pair), word)
                v_out[w_out] = word_count_dict[word]
            word_count_dict = v_out

            self.bpe_merge_history[most_freq_pair] = merge_count
            self.bpe_merge_word_dict[most_freq_pair[0] + most_freq_pair[1]] = most_freq_pair

            merge_count += 1

    def encode(self, origin_word):
        # origin_word 를 글자 단위로 쪼개고, 마지막에 </w> 를 붙입니다.
        # ex : ('l', 'o', 'k', 'i', '</w>')
        word = tuple(origin_word) + ('</w>',)

        # 글자 단위 pair 생성
        # ex : {('i', '</w>'), ('l', 'o'), ('o', 'k'), ('k', 'i')}
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char

        if not pairs:
            # pairing 을 할 수 없다면 그대로 반환
            return origin_word

        iteration = 0
        while True:
            # train 에서 merge 한 history 인 bpe_merge_history 를 가지고 입력된 글자를 치환하기를 반복
            iteration += 1

            # train 에서 merge 한 history 인 bpe_merge_history 를 가지고, merge 에 사용한 동일한 pair 를 탐색
            bigram = min(pairs, key=lambda pair: self.bpe_merge_history.get(pair, float('inf')))
            if bigram not in self.bpe_merge_history:
                break
            first, second = bigram

            # 새 단어의 글자 리스트
            new_word = []
            i = 0
            # 글자 수 -1 만큼 순회하면서 글자를 합성
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = set()
                prev_char = word[0]
                for char in word[1:]:
                    pairs.add((prev_char, char))
                    prev_char = char

        # 특별 토큰인 </w>는 출력하지 않는다.
        if word[-1] == '</w>':
            word = word[:-1]
        elif word[-1].endswith('</w>'):
            word = word[:-1] + (word[-1].replace('</w>', ''),)

        return word


# 말뭉치에서 단어 / 빈도 사전 추출하기
word_count_dict = {
    'l o w </w>': 5,
    'l o w e r </w>': 2,
    'n e w e s t </w>': 6,
    'w i d e s t </w>': 3
}
bpe_obj = BytePairEncoder()

# bpe 객체에 단어사전 학습
bpe_obj.bpe_train(word_count_dict=word_count_dict)

# OOV 를 인코딩
print("BPE 인코딩 결과")
# ex : ('lo', 'k', 'i')
print(bpe_obj.encode("loki"))

# ex : ('low', 'i', 'n', 'g')
print(bpe_obj.encode("lowing"))
print("")

# 위에서 보이다시피 bpe 를 사용하면 OOV 도 에러가 나지 않으며, 학습에 사용되는 문장이 많아질수록 정확도는 높아집니다.

"""
[워드피스]
WordPiece Tokenizer 는 BPE의 변형 알고리즘입니다. 
해당 알고리즘은 BPE가 빈도수에 기반하여 가장 많이 등장한 쌍을 병합하는 것과는 달리, 
병합되었을 때 말뭉치의 우도(Likelihood)를 가장 높이는 쌍을 병합합니다.
이 알고리즘은 Google이 유명 딥 러닝 모델 BERT를 훈련하기 위해서 사용되기도 하였습니다.

(WPT 설명)

"""



