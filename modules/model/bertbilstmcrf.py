from transformers import BertPreTrainedModel,BertModel
from torch import nn
import torch
from torchcrf import CRF
import torch.nn.functional as F
log_soft = F.log_softmax

class Bert_BiLSTM_CRF(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert_BiLSTM_CRF, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bilstm = nn.LSTM(config.hidden_size, config.hidden_size // 2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
        self.crf = CRF(self.num_labels, batch_first=True)
    
    def forward(self, input_ids, attn_masks, labels=None):
        outputs = self.bert(input_ids, attn_masks)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        bilstm_output, _ = self.bilstm(sequence_output)
        emission = self.classifier(bilstm_output)
        attn_masks = attn_masks.type(torch.uint8)
        if labels is not None:
            loss = -self.crf(log_soft(emission, 2), labels, mask=attn_masks, reduction='mean')
            return loss
        else:
            prediction = self.crf.decode(emission, mask=attn_masks)
            return prediction
