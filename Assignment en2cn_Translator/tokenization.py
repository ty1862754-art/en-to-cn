import torch
import numpy as np
from collections import Counter
from nltk import word_tokenize # download nltk data in tutorial.ipynb at the first access

def seq_padding(X, pad_id=0):
    """
    add padding to a batch data 
    """
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [pad_id] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def casual_mask(size):
    # Creating a square matrix of dimensions 'size x size' filled with ones
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0


class PrepareData:
    def __init__(self, train_file, dev_file, batch_size, unk_id, pad_id):
        self.unk_id = unk_id
        self.pad_id = pad_id
    
        # 01. Read the data and tokenize
        self.train_en, self.train_cn = self.load_data(train_file) 
        self.dev_en, self.dev_cn = self.load_data(dev_file)

        # 02. build dictionary: English and Chinese
        self.en_word_dict, self.en_total_words, self.en_index_dict = self.build_dict(self.train_en)
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = self.build_dict(self.train_cn)

        # 03. word to id by dictionary
        self.train_en, self.train_cn = self.wordToID(self.train_en, self.train_cn, self.en_word_dict, self.cn_word_dict)
        self.dev_en, self.dev_cn = self.wordToID(self.dev_en, self.dev_cn, self.en_word_dict, self.cn_word_dict)
     
        # 04. batch + padding + mask
        self.train_data = self.splitBatch(self.train_en, self.train_cn, batch_size)
        self.dev_data = self.splitBatch(self.dev_en, self.dev_cn, batch_size=1, shuffle=False)


    def load_data(self, path):
        """
        Read English and Chinese Data 
        tokenize the sentence and add start/end marks(Begin of Sentence; End of Sentence)
        en = [['BOS', 'i', 'love', 'you', 'EOS'], 
              ['BOS', 'me', 'too', 'EOS'], ...]
        cn = [['BOS', '我', '爱', '你', 'EOS'], 
              ['BOS', '我', '也', '是', 'EOS'], ...]
        """
        en = []
        cn = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                en.append(["BOS"] + word_tokenize(line[0].lower()) + ["EOS"])
                cn.append(["BOS"] + word_tokenize(" ".join([w for w in line[1]])) + ["EOS"])
        return en, cn

    def build_dict(self, sentences, max_words=50000):
        """
        sentences: list of word list 
        build dictionary as {key(word): value(id)}
        """
        word_count = Counter()
        for sentence in sentences:
            for s in sentence:
                word_count[s] += 1

        ls = word_count.most_common(max_words)
        total_words = len(ls) + 2
        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dict['UNK'] = self.unk_id
        word_dict['PAD'] = self.pad_id
        # inverted index: {key(id): value(word)}
        index_dict = {v: k for k, v in word_dict.items()}
        return word_dict, total_words, index_dict

    def wordToID(self, en, cn, en_dict, cn_dict):
        """ 
        TODO: convert word to id with the dictionary generated from training english and chinese data

        convert input/output word lists to id lists. 
        Use input word list length to sort, reduce padding.
        
        1. English example: ['BOS', 'i', 'love', 'you', 'EOS'] -> [2, 23, 45, 67, 3]
        2. Chinese example: ['BOS', '我', '爱', '你', 'EOS'] ->  [2, 19, 890, 5, 3]
        3. return out_en_ids, out_cn_ids
        """
        out_en_ids = []
        for sentence in en:
            out_en_ids.append([en_dict.get(word, self.unk_id) for word in sentence])
            
        out_cn_ids = []
        for sentence in cn:
            out_cn_ids.append([cn_dict.get(word, self.unk_id) for word in sentence])
            
        # Use input word list length to sort, reduce padding.
        sorted_pairs = sorted(zip(out_en_ids, out_cn_ids), key=lambda x: len(x[0]))
        out_en_ids, out_cn_ids = zip(*sorted_pairs)
        
        return list(out_en_ids), list(out_cn_ids)

    def splitBatch(self, en, cn, batch_size, shuffle=True):
        """
        get data into batches
        """
        idx_list = np.arange(0, len(en), batch_size)
        if shuffle:
            np.random.shuffle(idx_list)

        batch_indexs = []
        for idx in idx_list:
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))

        batches = []
        for batch_index in batch_indexs:
            batch_en = [en[index] for index in batch_index]
            batch_cn = [cn[index] for index in batch_index]
            # paddings: batch, batch_size, batch_MaxLength
            batch_en = seq_padding(batch_en, pad_id=self.pad_id) # batch_id [B L]
            batch_cn = seq_padding(batch_cn, pad_id=self.pad_id) 
            batches.append(MaskBatch(batch_en, batch_cn, pad_id=self.pad_id))
        return batches


# Attention Mask 这里的mask为0对应掩码的位置，与通常的mask定义相反
class MaskBatch:
    '''Object for holding a batch of data with mask during training.'''
    def __init__(self, src, tgt=None, pad_id=0):
        # convert words id to long format.
        src = torch.from_numpy(src).long()
        tgt = torch.from_numpy(tgt).long()
        self.src = src
        # get the padding postion binary mask 
        self.src_mask = (src != pad_id).unsqueeze(1).unsqueeze(1) # mask [B 1 1 src_L]
        if tgt is not None:
            # decoder input
            self.tgt = tgt[:, :-1]
            # decoder target
            self.tgt_y = tgt[:, 1:]
            # add attention mask to decoder input
            self.tgt_mask = self.make_decoder_mask(self.tgt, pad_id)
            # check decoder output padding number
            self.ntokens = (self.tgt_y != pad_id).data.sum()

    def make_decoder_mask(self, tgt, pad_id):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(1) # mask [B 1 1 tgt_L] 
        tgt_mask = tgt_mask & casual_mask(tgt.size(-1)).unsqueeze(1).type_as(tgt_mask.data) 
        return tgt_mask 
    