import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac

"""
[텍스트 데이터 증강]
텍스트 데이터를 증강시키는 여러 기법을 모아두었습니다.

데이터 변형으로 인한 증강 방식을 사용할 때,
변형의 정도가 너무 작다면 동일 데이터를 사용한 것과 같이 오버피팅이 발생하며,
변형의 정도가 너무 크다면 데이터 품질이 저하될 수 있습니다.
"""

texts = [
    "Those who can imagine anything, can create the impossible.",
    "We can only see a short distance ahead, but we can see plenty there that needs to be done.",
    "If a machine is expected to be infallible, it cannot also be intelligent.",
]

# 1. 삽입 : 문장에 영향을 끼치지 않은 의미 없는 문자, 단어, 수식어 등을 추가 하는 방식으로 데이터를 증강
augmented_texts = naw.ContextualWordEmbsAug(model_path="bert-base-uncased", action="insert").augment(texts)

print("1. Insert")
for text, augmented in zip(texts, augmented_texts):
    print(f"src : {text}")
    print(f"dst : {augmented}")
    print("------------------")
print("")

# 2. 삭제 : 문장에 영향을 끼치지 않은 내에서 문자, 단어, 수식어 등을 삭제 하는 방식으로 데이터를 증강
augmented_texts = nac.RandomCharAug(action="delete").augment(texts)

print("2. Delete")
for text, augmented in zip(texts, augmented_texts):
    print(f"src : {text}")
    print(f"dst : {augmented}")
    print("------------------")
print("")

# 3. 교체 : 단어와 문자의 위치를 교환 합니다.
# ex : 문제점을 찾지 말고 해결책을 찾으로 -> 해결책을 찾으라 문제점을 찾지 말고
# 이 방식은 의미가 변경되지 않은 양질의 데이터 증강을 기대할 수 있지만, 오류에 따라 의미 자체가 달라질 가능성이 있습니다.
# 현재로서는 꼭 데이터 증강 후 확인하여 실제로 적용을 해야합니다.
augmented_texts = naw.RandomWordAug(action="swap").augment(texts)

print("3. Swap")
for text, augmented in zip(texts, augmented_texts):
    print(f"src : {text}")
    print(f"dst : {augmented}")
    print("------------------")
print("")

# 4. 대체 : 단어나 문자를 임의의 단어나 문자로 바꾸거나 동의어로 변경합니다.
# ex : 해 -> 태양, 계란 -> 달걀
# 이 방식은 교체보다 데이터 정합성이 어긋날 가능성이 적습니다.
# 다만, 여전히 감독이 필요한 증강 방식입니다.
augmented_texts = naw.SynonymAug(aug_src='wordnet').augment(texts)

print("4. Replace")
for text, augmented in zip(texts, augmented_texts):
    print(f"src : {text}")
    print(f"dst : {augmented}")
    print("------------------")
print("")

# 4.1. 대체 심화 : 단어나 문자를 임의의 단어나 문자로 바꾸거나 동의어로 변경합니다. (3번 보다 성능이 좋습니다.)
reserved_tokens = [
    ["can", "can't", "cannot", "could"],
]

augmented_texts = naw.ReservedAug(reserved_tokens=reserved_tokens).augment(texts)

print("4.1. Replace2")
for text, augmented in zip(texts, augmented_texts):
    print(f"src : {text}")
    print(f"dst : {augmented}")
    print("------------------")
print("")

# 5. 역번역 : 입력 텍스트를 특정 언어로 번역 후 다시 본래 언어로 번역
# 번역을 거치는 동안 뜻은 유지되면서 표현 방식이 변경되어 데이터가 증강되는 기법입니다.
# 입력 텍스트와 번역 모델의 성능에 좌우되는 방식입니다.
# augmented_texts = naw.BackTranslationAug(
#     from_model_name='facebook/wmt19-en-de',
#     to_model_name='facebook/wmt19-de-en'
# ).augment(texts)
#
# print("5. Back-Translation")
# for text, augmented in zip(texts, augmented_texts):
#     print(f"src : {text}")
#     print(f"dst : {augmented}")
#     print("------------------")
# print("")
