from queue import Queue

import torch
# replace to BaseTokenzier
# from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers import AutoTokenizer


SPLIT_CHAR_LIST: list[str] = [':', '-']


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
    return texts


class TokenTree:
    def __init__(self, token: int = -1):
        self.childrens: dict[int, "TokenTree"] = dict()
        self.token = token

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
            cur_root = root
        return root


class OCRLogitProcessor:
    def __init__(
        self,
        ocr_tree: TokenTree,
        tokenizer: AutoTokenizer,
        special_tree: TokenTree = None,
        filter_value: int = -float("Inf"),
    ):
        # origin pointer
        self.ocr_tree = ocr_tree
        self.cur_ocr_tree = ocr_tree
        
        # current pointer
        self.special_tree = special_tree
        self.cur_special_tree = special_tree
        
        self.filter_value = filter_value
        
        self.is_valid = True
        self.tokenizer = tokenizer
    
    def clear(self):
        self.is_valid = True
        self.cur_ocr_tree = self.ocr_tree
        self.cur_special_tree = self.special_tree
    
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
        
        can_next_tokens: set[int] = set(self.cur_ocr_tree.childrens.keys())
        can_next_tokens |= set(self.cur_special_tree.childrens.keys())
        can_next_tokens = list(can_next_tokens)
        
        mask = torch.BoolTensor = torch.zeros((vocab_size), device=input_ids.device, dtype=torch.bool)
        mask[can_next_tokens] = 1
        mask = ~mask
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

        if next_token in self.processor.cur_ocr_tree:
            self.processor.cur_ocr_tree = self.processor.cur_ocr_tree[next_token]
            if self.processor.cur_ocr_tree.is_leaf:
                self.processor.cur_ocr_tree = self.processor.ocr_tree
        elif next_token in self.processor.cur_special_tree:
            self.processor.cur_special_tree = self.processor.cur_special_tree[next_token]
            if self.processor.cur_special_tree.is_leaf:
                self.processor.cur_special_tree = self.processor.special_tree
        else:
            self.processor.is_valid = False

        return torch.full((bs, ), False, device=input_ids.device, dtype=torch.bool)


def get_ocr_logit_processor(
    texts: list[str], tokenizer: AutoTokenizer, specials: list[str] = [], split_text: bool = True
) -> OCRLogitProcessor:
    if split_text:
        texts: list[str] = split_text_func(texts=texts)

    ocr_tree: TokenTree = TokenTree.build_from_ocr(texts=texts, tokenizer=tokenizer)
    special_tree: TokenTree = TokenTree.build_from_ocr(texts=specials, tokenizer=tokenizer)
    ocr_logit_processor = OCRLogitProcessor(
        ocr_tree=ocr_tree, tokenizer=tokenizer, special_tree=special_tree
    )
    return ocr_logit_processor


def get_processor_and_criteria(
    texts: list[str],
    tokenizer: AutoTokenizer,
    specials: list[str],
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
        texts=texts, tokenizer=tokenizer, specials=specials, split_text=split_text
    )
    criteria = OCRUpdateCriteria(logit_processor=logit_processor)
    return logit_processor, criteria
