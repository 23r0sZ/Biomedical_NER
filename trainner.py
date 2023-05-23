from modules.utils import seed_torch
from modules.datasets.NER_dataset import NER_Dataset,dataset_train,dataset_test
import torch
from transformers import BertTokenizer
import numpy as np
import datetime
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import timeit
import subprocess
from optparse import OptionParser
from config.config import Config as config
import pickle
import re
from modules.model.bert import BertForTokenClassification
seed_torch()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(train_iter, eval_iter, tag2idx, config, bert_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"):
    #print('#Tags: ', len(tag2idx))
    unique_labels = list(tag2idx.keys())
    print(unique_labels)
    #model = Bert_CRF.from_pretrained(bert_model, num_labels = len(tag2idx))
    model = BertForTokenClassification.from_pretrained(bert_model, num_labels=len(tag2idx))
    print(model)
    model.train()
    if torch.cuda.is_available():
      model.cuda()
    num_epoch = config.epoch
    gradient_acc_steps = 1
    t_total = len(train_iter) // gradient_acc_steps * num_epoch
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr, eps=config.eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)
    global_step = 0
    model.zero_grad()
    model.train()
    training_loss = []
    validation_loss = []
    fb1_scores = []
    train_iterator = trange(num_epoch, desc="Epoch", disable=0)
    start_time = timeit.default_timer()

    for epoch in (train_iterator):
        epoch_iterator = tqdm(train_iter, desc="Iteration", disable=-1)
        tr_loss = 0.0
        tmp_loss = 0.0
        model.train()
        for step, batch in enumerate(epoch_iterator):
            s = timeit.default_timer()
            token_ids, attn_mask, _, labels, _, _= batch
            #print(labels)
            inputs = {'input_ids' : token_ids.to(device),
                     'attention_mask' : attn_mask.to(device),
                     'labels' : labels.to(device)
                     }  
            loss= model(**inputs)[0] 
            loss.backward()
            tmp_loss += loss.item()
            tr_loss += loss.item()
            if (step + 1) % 1 == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
            if step == 0:
                print('\n%s Step: %d of %d Loss: %f' %(str(datetime.datetime.now()), (step+1), len(epoch_iterator), loss.item()))
            if (step+1) % 100 == 0:
                print('%s Step: %d of %d Loss: %f' %(str(datetime.datetime.now()), (step+1), len(epoch_iterator), tmp_loss/1000))
                tmp_loss = 0.0
      
        print("Training Loss: %f for epoch %d" %(tr_loss/len(train_iter), epoch))
        training_loss.append(tr_loss/len(train_iter))
        #'''
        #Y_pred = []
        #Y_true = []
        val_loss = 0.0
        model.eval()
        writer = open(config.apr_dir + 'prediction_'+str(epoch)+'.csv', 'w')
        for i, batch in enumerate(eval_iter):
            token_ids, attn_mask, org_tok_map, labels, original_token, sorted_idx = batch
            #attn_mask.dt
            inputs = {'input_ids': token_ids.to(device),
                      'attention_mask' : attn_mask.to(device)
                     }  
            
            # dev_inputs = {'input_ids' : token_ids.to(device),
            #              'attn_masks' : attn_mask.to(device),
            #              'labels' : labels.to(device)
            #              } 
            with torch.torch.no_grad():
                outputs = model(**inputs)
            val_loss += outputs[0].item()
            logits = outputs[1].detach().cpu().numpy()
            tag_seqs = [list(p) for p in np.argmax(logits, axis=2)]
            #print(labels.numpy())
            #print(type(tag_seqs), tag_seqs[0][0])
            y_true = list(labels.cpu().numpy())
            for i in range(len(sorted_idx)):
                o2m = org_tok_map[i]
                pos = sorted_idx.index(i)
                for j, orig_tok_idx in enumerate(o2m):
                    writer.write(original_token[i][j] + '\t')
                    writer.write(unique_labels[y_true[pos][orig_tok_idx]] + '\t')
                    pred_tag = unique_labels[tag_seqs[pos][orig_tok_idx]]
                    if pred_tag == 'X':
                        pred_tag = 'O'
                    writer.write(pred_tag + '\n')
                writer.write('\n')
                
        validation_loss.append(val_loss/len(eval_iter))
        writer.flush()
        print('Epoch: ', epoch)
        command = "python conlleval.py < " + config.apr_dir + "prediction_"+str(epoch)+".csv"
        process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
        result = process.communicate()[0].decode("utf-8")
        print(result)
        fb1_pattern = r"FB1:\s+(\d+\.\d+)%"
        fb1_match = re.search(fb1_pattern, result)
        if fb1_match:
            fb1_score = fb1_match.group(1)
            fb1_scores.append(fb1_score)
        if fb1_score and (fb1_score >= max(fb1_scores) or epoch==num_epoch):
            print('saving best model...')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': tr_loss/len(train_iter),
            }, config.apr_dir + 'best_model' + '.pt')
        if epoch==num_epoch:
            print('saving last model...')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': tr_loss/len(train_iter),
            }, config.apr_dir + 'last_model' + '.pt')
    total_time = timeit.default_timer() - start_time
    print('Total training time: ',   total_time)
    return training_loss, validation_loss

def test(config, test_iter, model, unique_labels, test_output):
    model.eval()
    writer = open(config.apr_dir + test_output, 'w')
    for i, batch in enumerate(test_iter):
        token_ids, attn_mask, org_tok_map, labels, original_token, sorted_idx = batch
        #attn_mask.dt
        inputs = {'input_ids': token_ids.to(device),
                  'attention_mask' : attn_mask.to(device)
                 }  
        with torch.torch.no_grad():
            outputs = model(**inputs)
        y_true = list(labels.cpu().numpy())
        logits = outputs[1].detach().cpu().numpy()
        tag_seqs = [list(p) for p in np.argmax(logits, axis=2)]
        for i in range(len(sorted_idx)):
            o2m = org_tok_map[i]
            pos = sorted_idx.index(i)
            for j, orig_tok_idx in enumerate(o2m):
                writer.write(original_token[i][j] + '\t')
                writer.write(unique_labels[y_true[pos][orig_tok_idx]] + '\t')
                pred_tag = unique_labels[tag_seqs[pos][orig_tok_idx]]
                if pred_tag == 'X':
                    pred_tag = 'O'
                writer.write(pred_tag + '\n')
            writer.write('\n')
    writer.flush()
    command = "python conlleval.py < " + config.apr_dir + test_output
    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
    result = process.communicate()[0].decode("utf-8")
    print(result)

def load_model(config, do_lower_case=True):
    f = open(config.apr_dir +'tag2idx.pkl', 'rb')
    tag2idx = pickle.load(f)
    unique_labels = list(tag2idx.keys())
    model = BertForTokenClassification.from_pretrained(config.bert_model, num_labels=len(tag2idx))
    checkpoint = torch.load(config.apr_dir + config.model_name, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    global bert_tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=do_lower_case)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model, bert_tokenizer, unique_labels, tag2idx


def usage(parameter):
    parameter.print_help()
    print("Example usage (training):\n", \
        "\t python trainner.py --mode train ")

    print("Example usage (testing):\n", \
        "\t python trainner.py --mode test ")

if __name__ == "__main__":
    user_input = OptionParser()
    user_input.add_option("--mode", dest="model_mode", metavar="string", default='traning',
                      help="mode of the model (required)")
    (options, args) = user_input.parse_args()

    if options.model_mode == "train":
        train_iter, eval_iter, tag2idx = dataset_train(config=config, bert_tokenizer=config.bert_model, do_lower_case=True)
        t_loss, v_loss = train(train_iter, eval_iter, tag2idx, config=config, bert_model=config.bert_model)
    elif options.model_mode == "test":
        model, bert_tokenizer, unique_labels, tag2idx = load_model(config=config, do_lower_case=True)
        test_iter = dataset_test(config, tag2idx, bert_tokenizer=config.bert_model, do_lower_case=True)
        print('test len: ', len(test_iter))
        test(config, test_iter, model, unique_labels, config.test_out)
    else:
        usage(user_input)
