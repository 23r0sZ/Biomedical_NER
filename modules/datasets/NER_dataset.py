from transformers import BertTokenizer
from torch.utils.data import Dataset,DataLoader
import torch
import numpy as np
from modules.datasets.utils import corpus_reader
import pickle

class NER_Dataset(Dataset):
    def __init__(self, tag2idx, sentences, labels, tokenizer_path = '', do_lower_case=True):
        self.tag2idx = tag2idx
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=do_lower_case)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = []
        for x in self.labels[idx]:
            if x in self.tag2idx.keys():
                label.append(self.tag2idx[x])
            else:
                label.append(self.tag2idx['O'])
        bert_tokens = []
        orig_to_tok_map = []
        bert_tokens.append('[CLS]')
        #append dummy label 'X' for subtokens
        modified_labels = [self.tag2idx['X']]
        for i, token in enumerate(sentence):
            if len(bert_tokens) >= 512:
                break
            orig_to_tok_map.append(len(bert_tokens))
            modified_labels.append(label[i])
            new_token = self.tokenizer.tokenize(token)
            bert_tokens.extend(new_token)
            modified_labels.extend([self.tag2idx['X']] * (len(new_token) -1))

        bert_tokens.append('[SEP]')
        modified_labels.append(self.tag2idx['X'])
        token_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        if len(token_ids) > 511:
            token_ids = token_ids[:512]
            modified_labels = modified_labels[:512]
        return token_ids, len(token_ids), orig_to_tok_map, modified_labels, self.sentences[idx]


def pad(batch):
    '''Pads to the longest sample'''
    get_element = lambda x: [sample[x] for sample in batch]
    seq_len = get_element(1)
    maxlen = np.array(seq_len).max()
    do_pad = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    tok_ids = do_pad(0, maxlen)
    attn_mask = [[(i>0) for i in ids] for ids in tok_ids] 
    LT = torch.LongTensor
    label = do_pad(3, maxlen)

    # sort the index, attn mask and labels on token length
    token_ids = get_element(0)
    token_ids_len = torch.LongTensor(list(map(len, token_ids)))
    _, sorted_idx = token_ids_len.sort(0, descending=True)

    tok_ids = LT(tok_ids)[sorted_idx]
    attn_mask = LT(attn_mask)[sorted_idx]
    labels = LT(label)[sorted_idx]
    org_tok_map = get_element(2)
    sents = get_element(-1)

    return tok_ids, attn_mask, org_tok_map, labels, sents, list(sorted_idx.cpu().numpy())


def dataset_train(config, bert_tokenizer="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", do_lower_case=True):
    training_data, validation_data = config.data_dir+config.training_data, config.data_dir+config.val_data 
    train_sentences, train_labels, label_set = corpus_reader(training_data, delim='\t')
    label_set.append('X')
    tag2idx = {t:i for i, t in enumerate(label_set)}
    #print('Training datas: ', len(train_sentences))
    train_dataset = NER_Dataset(tag2idx, train_sentences, train_labels, tokenizer_path = bert_tokenizer, do_lower_case=do_lower_case)
    # save the tag2indx dictionary. Will be used while prediction
    with open(config.apr_dir + 'tag2idx.pkl', 'wb') as f:
        pickle.dump(tag2idx, f, pickle.HIGHEST_PROTOCOL)
    dev_sentences, dev_labels, _ = corpus_reader(validation_data, delim='\t')
    dev_dataset = NER_Dataset(tag2idx, dev_sentences, dev_labels, tokenizer_path = bert_tokenizer, do_lower_case=do_lower_case)

    #print(len(train_dataset))
    train_iter = DataLoader(dataset=train_dataset,
                                batch_size=config.batch_size,
                                shuffle=True,
                                num_workers=4,
                                collate_fn=pad)
    eval_iter = DataLoader(dataset=dev_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=pad)
    return train_iter, eval_iter, tag2idx

def dataset_test(config, tag2idx, bert_tokenizer="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", do_lower_case=True):
    test_data = config.data_dir+config.test_data
    test_sentences, test_labels, _ = corpus_reader(test_data, delim='\t')
    test_dataset = NER_Dataset(tag2idx, test_sentences, test_labels, tokenizer_path = bert_tokenizer, do_lower_case=do_lower_case)
    test_iter = DataLoader(dataset=test_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=pad)
    return test_iter