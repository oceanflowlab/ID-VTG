from copy import deepcopy

import torchtext
from torchtext.data import get_tokenizer
from transformers import BertTokenizer  # 添加 Hugging Face Tokenizer

tokenizers = dict()
def register_tokenizer(name):
    def decorator(module):
        tokenizers[name] = module
        return module
    return decorator


@register_tokenizer('glove')
class GloVeTokenizer:

    def __init__(self, name='6B'):

        self.vocab = torchtext.vocab.GloVe(name=name)
        self.tokenizer = get_tokenizer("basic_english")

    def __call__(self, text, max_len=None):
        """
        Args:
            text (str): text query.
            max_len (int): maximum sequence length.

        Returns:
            feats (float tensor, (c, t)): feature sequence.
        """
        # tokenize by word
        ## NOTE: unknown words are assigned zero vector
        words = self.tokenizer(text)
        feats = self.vocab.get_vecs_by_tokens(words, lower_case_backup=True)
        if max_len is not None:
            feats = feats[:max_len]
        feats = feats.transpose(0, 1)   # (c, t)

        return feats


@register_tokenizer('blip2')  # 添加 BLIP2 Tokenizer
class Blip2Tokenizer:
    
    def __init__(self, truncation_side="right", max_length=77):
        """
        初始化 BLIP2 Tokenizer
        
        参数:
        truncation_side: "left" 或 "right"，表示截断的方向
        max_length: 最大序列长度
        """
        # 创建 BERT Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", 
            truncation_side=truncation_side
        )
        # 添加特殊标记
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]", "eos_token": "[SEP]"})
        
        # 保存配置
        self.max_length = max_length
        self.truncation_side = truncation_side
        
        # 添加类型提示属性
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.pad_token = self.tokenizer.pad_token
        self.vocab_size = self.tokenizer.vocab_size
    
    def __call__(self, text, max_len=None):
        """
        Args:
            text (str or list[str]): 单个文本或文本列表
            max_len (int, optional): 序列最大长度，若为None使用初始化值
            
        Returns:
            dict: 包含以下键:
                input_ids (tensor): 形状为 [batch_size, sequence_length] 的 token ID 张量
                attention_mask (tensor): 形状为 [batch_size, sequence_length] 的注意力掩码
                special_tokens_mask (tensor): 特殊 token 的掩码
        """
        actual_max_len = max_len if max_len is not None else self.max_length
        
        # 处理单个文本的情况
        if isinstance(text, str):
            text = [text]
        
        # 使用 BLIP2 的 tokenizer 处理文本
        encoded = self.tokenizer(
            text, 
            padding='max_length',
            truncation=True,
            max_length=actual_max_len,
            return_tensors='pt',
            return_special_tokens_mask=True
        )
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }    
    @property
    def vocab(self):
        """获取词表字典 (id -> token)"""
        return self.tokenizer.get_vocab()


def make_tokenizer(name):
    return tokenizers[name]()
