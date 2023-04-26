from collections import OrderedDict

def corpus_reader(path, delim='\t', word_idx=0, label_idx=-1):
    tokens, labels = [], []
    tmp_tok, tmp_lab = [], []
    label_set = []
    with open(path, 'r') as reader:
        for line in reader:
            line = line.strip()
            cols = line.split(delim)
            if len(cols) < 2:
                if len(tmp_tok) > 0:
                    tokens.append(tmp_tok); labels.append(tmp_lab)
                tmp_tok = []
                tmp_lab = []
            else:
                tmp_tok.append(cols[word_idx])
                tmp_lab.append(cols[label_idx])
                label_set.append(cols[label_idx])
    return tokens, labels, list(OrderedDict.fromkeys(label_set))

if __name__=='__main__':
    tokens,labels,a=corpus_reader('/media/ba/N_Vol/ubuntu/NER-code/my_bert_crf/dataset_NER/BC5CDR-chem-IOB/devel.tsv')
    print(tokens)
    print(labels)