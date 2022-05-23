from .fast_jieba import Jieba

_DEFAULT_JIEBA = Jieba()


def add_word(word, freq=None, tag=None):
    _DEFAULT_JIEBA.add_word(word, freq, tag)


def suggest_freq(segment):
    _DEFAULT_JIEBA.suggest_freq(segment)


def load_dict(path):
    _DEFAULT_JIEBA.load_dict(path)


def cut(text, hmm=True, cut_all=False):
    if cut_all:
        return _DEFAULT_JIEBA.cut_all(text)
    return _DEFAULT_JIEBA.cut(text, hmm)


def batch_cut(texts, hmm=True, cut_all=False):
    if cut_all:
        return _DEFAULT_JIEBA.batch_cut_all()
    return _DEFAULT_JIEBA.batch_cut(texts, hmm)


def cut_for_search(text, hmm=True):
    return _DEFAULT_JIEBA.cut_for_search(text, hmm)


def batch_cut_for_search(texts, hmm=True):
    return _DEFAULT_JIEBA.batch_cut_for_search(texts, hmm)


def posseg(text, hmm=True):
    return _DEFAULT_JIEBA.tagging(text, hmm)


def batch_posseg(texts, hmm=True):
    return _DEFAULT_JIEBA.batch_tagging(texts, hmm)


def tokenize(text, mode="default", hmm=True):
    assert mode in ["default", "search"], f"invalid mode {mode}"
    return _DEFAULT_JIEBA.tokenize(text, mode, hmm)


def batch_tokenize(texts, mode="default", hmm=True):
    assert mode in ["default", "search"], f"invalid mode {mode}"
    return _DEFAULT_JIEBA.batch_tokenize(texts, mode, hmm)

