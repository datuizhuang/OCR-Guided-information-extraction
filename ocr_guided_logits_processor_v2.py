import re
from queue import Queue

import torch
from transformers import AutoTokenizer


SPLIT_CHAR_LIST: list[str] = [':', '-']
RE_PATTERN_LIST: list[str] = ['\D+|\d']

def split_text_func(texts: list[str]) -> list[str]:
    """ split text
    for example:
        origin ocr: "abcd:efg"
        result: "abcdefg", "abcd:", "efg"

    Parameters
    ----------
    texts : list[str]

    Returns
    -------
    list[str]
        _description_
    """
    for split_char in SPLIT_CHAR_LIST:
        q = Queue()
        [q.put(text) for text in texts]
        new_texts: list[str] = []
        while not q.empty():
            text: str = q.get()
            if not text:
                continue
            new_texts.append(text)
            if split_char in text:
                idx = text.index(split_char)
                new_texts.append(text[:idx + 1])
                q.put(text[idx + 1: ])
        texts = list(set(new_texts))
    
    new_texts: list[str] = []
    for pattern in RE_PATTERN_LIST:
        for text in texts:
            new_texts.extend(re.findall(pattern=pattern, string=text))
    new_texts.extend(texts)
    texts = list(set(texts))
    return texts


class TokenTree:
    def __init__(self, token: int = -1):
        self.childrens: dict[int, "TokenTree"] = dict()
        self.token = token
        self.maybe_last = False

    def __contains__(self, token: int) -> bool:
        return token in self.childrens

    def __getitem__(self, token: int) -> "TokenTree":
        return self.childrens[token]

    def add_node(self, token: int, node: "TokenTree") -> "TokenTree":
        self.childrens[token] = node
        return node

    @property
    def is_leaf(self):
        return len(self.childrens) == 0

    @classmethod
    def build_from_ocr(cls, texts: list[str], tokenizer: AutoTokenizer) -> "TokenTree":
        root = TokenTree()
        cur_root = root
        for text in texts:
            tokens: list[int] = tokenizer.encode(text, add_special_tokens=False)
            for token in tokens:
                if token in cur_root:
                    cur_root = cur_root[token]
                else:
                    cur_root = cur_root.add_node(token=token, node=TokenTree(token=token))
            cur_root.maybe_last = True
            cur_root = root
        return root


class OCRLogitProcessor:
    def __init__(
        self,
        ocr_tree: TokenTree,
        tokenizer: AutoTokenizer,
        special_tree: TokenTree = None,
        special_tokens: list[int] = None,
        filter_value: int = -float("Inf"),
    ):
        # origin pointer
        self.ocr_tree = ocr_tree
        self.cur_ocr_trees = [ocr_tree]
        
        # current pointer
        self.special_tree = special_tree
        self.cur_special_trees = [special_tree]
        
        self.special_tokens = special_tokens or list()
        
        self.filter_value = filter_value
        
        self.is_valid = True
        self.tokenizer = tokenizer
    
    def clear(self):
        self.is_valid = True
        self.cur_ocr_trees = [self.ocr_tree]
        self.cur_special_trees = [self.special_tree]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """process logit by ocr informations

        Parameters
        ----------
        input_ids : torch.LongTensor
            model input
        scores : torch.FloatTensor
            input_ids logits

        Returns
        -------
        torch.FloatTensor
            input_ids logits
        """
        if not self.ocr_tree or not self.is_valid:
            return scores
        
        vocab_size: int = input_ids.size()[-1]
        dim: int = scores.dim()

        can_next_tokens: set[int] = set()
        for node in self.cur_ocr_trees:
            can_next_tokens |= set(node.childrens.keys())
        for node in self.cur_special_trees:
            can_next_tokens |= set(node.childrens.keys())
        can_next_tokens |= self.special_tokens
        can_next_tokens: list[int] = list(can_next_tokens)
        
        mask = torch.BoolTensor = torch.ones((vocab_size), device=input_ids.device, dtype=torch.bool)
        mask[can_next_tokens] = 0
        for _ in range(dim - 1):
            mask = mask[None, ...]
        scores_processes: torch.FloatTensor = scores.masked_fill(mask=mask, value=self.filter_value)
        return scores_processes


class OCRUpdateCriteria:
    def __init__(self, logit_processor: OCRLogitProcessor):
        self.processor = logit_processor
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        bs: int = input_ids.size()[0]
        assert bs == 1, 'now only support bs == 1'
        next_token = input_ids[..., -1]
        next_token: int = int(next_token[0])
        
        in_ocr, in_special = False, False
        
        cur_ocr_trees: set[TokenTree] = set()
        for node in self.processor.cur_ocr_trees:
            if next_token in node:
                in_ocr = True
                tmp_node = node[next_token]
                cur_ocr_trees.add(self.processor.ocr_tree if tmp_node.is_leaf else tmp_node)
                if tmp_node.maybe_last:
                    cur_ocr_trees.add(self.processor.ocr_tree)
        self.processor.cur_ocr_trees = cur_ocr_trees
        
        cur_special_trees: set[TokenTree] = set()
        for node in self.processor.cur_special_trees:
            if next_token in node:
                in_special = True
                tmp_node = node[next_token]
                cur_special_trees.add(self.processor.special_tree if tmp_node.is_leaf else tmp_node)
                if tmp_node.maybe_last:
                    cur_special_trees.add(self.processor.special_tree)
        self.processor.cur_special_trees = cur_special_trees
        
        if not in_ocr and not in_special and next_token not in self.processor.special_tokens:
            self.processor.is_valid = False

        return torch.full((bs, ), False, device=input_ids.device, dtype=torch.bool)


def get_ocr_logit_processor(
    texts: list[str], 
    tokenizer: AutoTokenizer, 
    specials: list[str] = [], 
    special_chars: list[str] = [],
    split_text: bool = True,
) -> OCRLogitProcessor:
    if split_text:
        texts: list[str] = split_text_func(texts=texts)

    ocr_tree: TokenTree = TokenTree.build_from_ocr(texts=texts, tokenizer=tokenizer)
    special_tree: TokenTree = TokenTree.build_from_ocr(texts=specials, tokenizer=tokenizer)
    special_tokens: list[int] = []
    for char in special_chars:
        special_tokens.extend(tokenizer.encode(char))
    special_tokens = list(set(special_tokens))
    ocr_logit_processor = OCRLogitProcessor(
        ocr_tree=ocr_tree, tokenizer=tokenizer, special_tree=special_tree, special_tokens=special_tokens
    )
    return ocr_logit_processor


def get_processor_and_criteria(
    texts: list[str],
    tokenizer: AutoTokenizer,
    specials: list[str],
    special_chars: list[str] = [],
    split_text: bool = True
):
    """

    Parameters
    ----------
    texts : list[str]
        ocr texts from your own model or api
    tokenizer : AutoTokenizer
        _description_
    specials : list[str]
        some special text not in ocr texts, such as `eos` token
    split_text : bool, optional
        whether split ocr texts by some special char, by default True

    Returns
    -------
    _type_
        _description_
    """
    logit_processor = get_ocr_logit_processor(
        texts=texts, tokenizer=tokenizer, specials=specials, split_text=split_text, special_chars=special_chars
    )
    criteria = OCRUpdateCriteria(logit_processor=logit_processor)
    return logit_processor, criteria
