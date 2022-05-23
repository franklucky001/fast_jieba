from typing import List, Tuple, Optional, Union

def add_word(word: str, freq: Optional[int] = None, tag: Optional[str] = None):
    ...

def suggest_freq(segment: str):
    ...


def load_dict(path: str):
    ...

def cut(text: str, hmm:bool = True, cut_all:bool = False) -> List[str]:
    ...

def batch_cut(texts: List[str], hmm:bool = True, cut_all:bool = False) -> List[List[str]]
    ...

def cut_for_search(text:str, hmm:bool = True) -> List[str]:
    ...

def batch_cut_for_search(texts: List[str], hmm:bool = True) -> List[List[str]]:
    ...

def posseg(text:str, hmm:bool = False) -> List[Tuple[str, str]]:
    ...

def batch_posseg(texts: List[str], hmm: bool = False) -> List[List[Tuple[str, str]]]:
    ...

def tokenize(text: str, mode:str = "default", hmm:bool = True) -> List[Tuple[str, int, int]]:
    """
    :param text:
    :param mode: default | search
    :param hmm:
    :return: list of words term with start index and end index
    """
    ...

def batch_tokenize(texts: List[str], mode:str = "default", hmm:bool = True) -> List[List[Tuple[str, int, int]]]:
    ...


def extract_tags(text: str, topK: int = 20, withWeight:bool = False, allowPos: Tuple[str, ...] = ()) \
        -> Union[List[str], List[Tuple[str, int]]]:
    ...

def textrank(text: str, topK: int = 20, withWeight:bool = False, allowPos: Tuple[str, ...] = ())\
        -> Union[List[str], List[Tuple[str, int]]]:
    ...
