import fast_jieba
from fast_jieba.analyse import extract_tags, textrank

fast_jieba.add_word("就读", freq=10000, tag='v')
ws = fast_jieba.posseg("小明就读于北京清华大学物理系")
print(ws)
print("***********")
words = fast_jieba.tokenize("小明就读北京清华大学物理系")
print(words)

tags = extract_tags("小明就读北京清华大学物理系")
print(tags)

tags = textrank("小明就读北京清华大学物理系")
print(tags)


print("**************")
texts = ["小明就读北京清华大学物理系" for _ in range(4)]

words = fast_jieba.batch_tokenize(texts)
print(words)

words = fast_jieba.batch_cut(texts)
print(words)

words = fast_jieba.batch_posseg(texts)
print(words)
