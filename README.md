# FastJieba

## fast_jieba for python, create by pyo3, use rust crate `jieba_rs`

```python
import fast_jieba
from fast_jieba.analyse import extract_tags, textrank

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

```
