import os
import json
import numpy as np

import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertModel, BertConfig
from pypinyin import pinyin, Style

from metrics import calculate_metric, get_p_r_f, classification_report


def set_seed(seed=123):
    """
    设置随机数种子，保证实验可重现
    :param seed:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()


def conver_labels_to_biolabels(labels):
    label2id = {"O": 0}
    id2label = {0: "O"}
    i = 1
    for label in labels:
        tmp_label = "B-" + label
        label2id[tmp_label] = i
        id2label[i] = tmp_label
        i += 1
        tmp_label = "I-" + label
        label2id[tmp_label] = i
        id2label[i] = tmp_label
        i += 1
    return label2id, id2label

def bio_decode(text, label, id2label, ent_labels):
    entities = {label: [] for label in ent_labels}
    label = [id2label[lab] if lab != -100 else "O" for lab in label]
    for i,(tex, lab) in enumerate(zip(text, label)):
      if "B-" in lab:
        word = []
        word.append(tex)
        elabel = lab.split("-")[-1]
        start = i
        i += 1
        while i < len(text) and "I-" + elabel == label[i]:
            word.append(text[i])
            i += 1
        entities[elabel].append((("").join(word), start))
    return entities
      
def get_pinyin_ids(text, pingyin_dir):
  config_path = pingyin_dir
  with open(os.path.join(config_path, 'pinyin_map.json'), encoding='utf8') as fin:
    pinyin_dict = json.load(fin)
  # load char id map tensor
  with open(os.path.join(config_path, 'id2pinyin.json'), encoding='utf8') as fin:
    id2pinyin = json.load(fin)
  # load pinyin map tensor
  with open(os.path.join(config_path, 'pinyin2tensor.json'), encoding='utf8') as fin:
    pinyin2tensor = json.load(fin)

  pinyin_list = pinyin(text, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
  pinyin_locs = {}
  # get pinyin of each location
  for index, item in enumerate(pinyin_list):
    pinyin_string = item[0]
    # not a Chinese character, pass
    if pinyin_string == "not chinese":
      continue
    if pinyin_string in pinyin2tensor:
      pinyin_locs[index] = pinyin2tensor[pinyin_string]
    else:
      ids = [0] * 8
      for i, p in enumerate(pinyin_string):
        if p not in pinyin_dict["char2idx"]:
          ids = [0] * 8
          break
        ids[i] = pinyin_dict["char2idx"][p]
      pinyin_locs[index] = ids

  # find chinese character location, and generate pinyin ids
  pinyin_ids = []
  for idx in range(len(text)):
    if idx in pinyin_locs:
      pinyin_ids.append(pinyin_locs[idx])
    else:
      pinyin_ids.append([0] * 8)

  return pinyin_ids


class NerDataset(Dataset):
    def __init__(self, examples, args):
        self.examples = examples
        self.tokenizer = args.tokenizer
        self.max_seq_len = args.max_seq_len
        self.label2id = args.label2id
        self.pinyin_dir = args.pinyin_dir

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        example = self.examples[item]
        example = json.loads(example)
        text = example["text"]
        labels = example["labels"]
        token_type_ids = [0] * self.max_seq_len
        label_ids = [self.label2id[label] for label in labels]
        pinyin_ids = get_pinyin_ids(text, self.pinyin_dir)
        # print(label_ids)
        if len(label_ids) > self.max_seq_len - 2:
            attention_mask = [1] * self.max_seq_len
            input_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + text[:self.max_seq_len - 2] + ['[SEP]'])
            pinyin_ids = [[0] * 8] + pinyin_ids[:self.max_seq_len - 2] + [[0] * 8]
            label_ids = [-100] + label_ids[:self.max_seq_len - 2] + [-100]
        else:
            attention_mask = [1] * (len(label_ids) + 2) + [0] * (self.max_seq_len - len(label_ids) - 2)
            input_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + text + ['[SEP]']) + [0] * (
                    self.max_seq_len - len(text) - 2)
            pinyin_ids = [[0]*8] +pinyin_ids + [[0]*8] + [[0] * 8 for _ in range(self.max_seq_len - len(text) - 2)]
            label_ids = [-100] + label_ids + [-100] + [-100] * (self.max_seq_len - 2 - len(label_ids))

        assert len(input_ids) == self.max_seq_len
        assert len(attention_mask) == self.max_seq_len
        assert len(token_type_ids) == self.max_seq_len
        assert len(pinyin_ids) == self.max_seq_len
        assert len(label_ids) == self.max_seq_len

        input_ids = torch.tensor(np.array(input_ids)).long()
        attention_mask = torch.tensor(np.array(attention_mask)).long()
        token_type_ids = torch.tensor(np.array(token_type_ids)).long()
        pinyin_ids = torch.tensor(np.array(pinyin_ids)).long()
        label_ids = torch.tensor(np.array(label_ids)).long()

        output = {
            "input_ids": input_ids,
            "pinyin_ids": pinyin_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": label_ids,
        }
        return output


class PinyinEmbedding(nn.Module):
  def __init__(self, embedding_size: int, pinyin_out_dim: int, config_path):
      """
          Pinyin Embedding Module
      Args:
          embedding_size: the size of each embedding vector
          pinyin_out_dim: kernel number of conv
      """
      super(PinyinEmbedding, self).__init__()
      with open(os.path.join(config_path, 'pinyin_map.json')) as fin:
          pinyin_dict = json.load(fin)
      self.pinyin_out_dim = pinyin_out_dim
      self.embedding = nn.Embedding(len(pinyin_dict['idx2char']), embedding_size)
      self.conv = nn.Conv1d(in_channels=embedding_size, out_channels=self.pinyin_out_dim, kernel_size=2,
                            stride=1, padding=0)

  def forward(self, pinyin_ids):
      """
      Args:
          pinyin_ids: (bs*sentence_length*pinyin_locs)
      Returns:
          pinyin_embed: (bs,sentence_length,pinyin_out_dim)
      """
      # input pinyin ids for 1-D conv
      embed = self.embedding(pinyin_ids)  # [bs,sentence_length,pinyin_locs,embed_size]
      bs, sentence_length, pinyin_locs, embed_size = embed.shape
      view_embed = embed.view(-1, pinyin_locs, embed_size)  # [(bs*sentence_length),pinyin_locs,embed_size]
      input_embed = view_embed.permute(0, 2, 1)  # [(bs*sentence_length), embed_size, pinyin_locs]
      # conv + max_pooling
      pinyin_conv = self.conv(input_embed)  # [(bs*sentence_length),pinyin_out_dim,H]
      pinyin_embed = F.max_pool1d(pinyin_conv, pinyin_conv.shape[-1])  # [(bs*sentence_length),pinyin_out_dim,1]
      return pinyin_embed.view(bs, sentence_length, self.pinyin_out_dim)  # [bs,sentence_length,pinyin_out_dim]

class GlyphEmbedding(nn.Module):
  """Glyph2Image Embedding"""

  def __init__(self, font_npy_files):
      super(GlyphEmbedding, self).__init__()
      font_arrays = [
          np.load(np_file).astype(np.float32) for np_file in font_npy_files
      ]
      self.vocab_size = font_arrays[0].shape[0]
      self.font_num = len(font_arrays)
      self.font_size = font_arrays[0].shape[-1]
      # N, C, H, W
      font_array = np.stack(font_arrays, axis=1)
      self.embedding = nn.Embedding(
          num_embeddings=self.vocab_size,
          embedding_dim=self.font_size ** 2 * self.font_num,
          _weight=torch.from_numpy(font_array.reshape([self.vocab_size, -1]))
      )

  def forward(self, input_ids):
      """
          get glyph images for batch inputs
      Args:
          input_ids: [batch, sentence_length]
      Returns:
          images: [batch, sentence_length, self.font_num*self.font_size*self.font_size]
      """
      # return self.embedding(input_ids).view([-1, self.font_num, self.font_size, self.font_size])
      return self.embedding(input_ids)




class NerModel(nn.Module):
    def __init__(self, args):
        super(NerModel, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.config = BertConfig.from_pretrained(args.bert_dir)
        self.dropout = nn.Dropout(0.1)
        self.pinyinEmbedding = PinyinEmbedding(embedding_size=128, pinyin_out_dim=self.config.hidden_size, config_path=args.pinyin_dir)
        self.glyphEmbedding = GlyphEmbedding(font_npy_files=args.glyph_dir)
        self.glyph_map = nn.Linear(1728, self.config.hidden_size)
        self.map_fc = nn.Linear(self.config.hidden_size * 3, self.config.hidden_size)
        self.classifier = nn.Linear(self.config.hidden_size, args.num_labels)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self,
          input_ids,
          pinyin_ids,
          token_type_ids,
          attention_mask,
          labels):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]
        pinyin_embeddings = self.pinyinEmbedding(pinyin_ids)
        glyph_embeddings = self.glyphEmbedding(input_ids)
        glyph_embeddings = self.glyph_map(glyph_embeddings)
        concat_embeddings = torch.cat((sequence_output, pinyin_embeddings, glyph_embeddings), 2)
        inputs_embeds = self.map_fc(concat_embeddings)
        inputs_embeds = self.dropout(inputs_embeds)
        logits = self.classifier(inputs_embeds)

        loss = None
        if labels is not None:
            loss = self.criterion(logits.view(-1, self.args.num_labels), labels.view(-1))

        return loss, logits


class NerPipeline:
    def __init__(self, model, args):
        self.model = model
        self.args = args

    def save_model(self):
        torch.save(self.model.state_dict(), self.args.save_dir)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.args.save_dir, map_location="cpu"))

    def build_optimizer_and_scheduler(self, t_total):
        module = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        # 差分学习率
        no_decay = ["bias", "LayerNorm.weight"]
        model_param = list(module.named_parameters())

        bert_param_optimizer = []
        other_param_optimizer = []

        for name, para in model_param:
            space = name.split('.')
            # print(name)
            if "bert" in space[0]:
                bert_param_optimizer.append((name, para))
            else:
                other_param_optimizer.append((name, para))

        optimizer_grouped_parameters = [
            # bert other module
            {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay, 'lr': self.args.lr},
            {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': self.args.lr},

            # 其他模块，差分学习率
            {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay, 'lr': self.args.other_lr},
            {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': self.args.other_lr},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(self.args.warmup_proportion * t_total), num_training_steps=t_total
        )

        return optimizer, scheduler

    def eval_forward(self, data_loader):
        span_logits = None
        span_labels = None
        self.model.eval()
        for eval_step, batch_data in enumerate(data_loader):
            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(self.args.device)
            labels = batch_data["labels"]
            output = self.model(batch_data["input_ids"],
                      batch_data["pinyin_ids"],
                      batch_data["token_type_ids"],
                      batch_data["attention_mask"],
                      labels)
            labels = labels.detach().cpu().numpy()
            loss, span_logit = output
            span_logit = span_logit.detach().cpu().numpy()
            span_logit = np.argmax(span_logit, -1)
            if span_logits is None:
                span_logits = span_logit
                span_labels = labels
            else:
                span_logits = np.append(span_logits, span_logit, axis=0)
                span_labels = np.append(span_labels, labels, axis=0)

        return span_logits, span_labels

    def get_metric(self, span_logits, span_labels, callback):
        batch_size = len(callback)
        total_count = [0 for _ in range(len(self.args.labels))]
        role_metric = np.zeros([len(self.args.labels), 3])
        for span_logit, label, tokens in zip(span_logits, span_labels, callback):
            span_logit = span_logit[1:len(tokens)+1]
            label = label[1:len(tokens)+1]
            pred_entities = bio_decode(tokens, span_logit, self.args.id2label, self.args.labels)
            gt_entities = bio_decode(tokens, label, self.args.id2label, self.args.labels)
            # print("========================")
            # print(tokens)
            # print(label)
            # print(span_logit)
            # print(pred_entities)
            # print(gt_entities)
            # print("========================")
            for idx, _type in enumerate(self.args.labels):
                if _type not in pred_entities:
                    pred_entities[_type] = []
                if _type not in gt_entities:
                    gt_entities[_type] = []
                total_count[idx] += len(gt_entities[_type])
                role_metric[idx] += calculate_metric(pred_entities[_type], gt_entities[_type])

        return role_metric, total_count

    def train(self, train_loader, dev_loader=None):

        t_total = len(train_loader) * self.args.train_epoch
        optimizer, scheduler = self.build_optimizer_and_scheduler(t_total)

        global_step = 0
        self.model.zero_grad()
        self.model.to(self.args.device)
        eval_step = self.args.eval_step
        best_f1 = 0.
        for epoch in range(1, self.args.train_epoch + 1):
            for step, batch_data in enumerate(train_loader):
                self.model.train()
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(self.args.device)
                labels = batch_data["labels"]
                output = self.model(batch_data["input_ids"],
                            batch_data["pinyin_ids"],
                                    batch_data["token_type_ids"],
                                    batch_data["attention_mask"],
                                    labels)
                loss, logits = output
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                loss.backward()
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()
                global_step += 1
                print('【train】Epoch: %d/%d Step: %d/%d loss: %.5f' % (
                    epoch, self.args.train_epoch, global_step, t_total, loss.item()))
                if dev_loader is not None and global_step % eval_step == 0:
                  span_logits, span_labels = self.eval_forward(dev_loader)
                  role_metric, total_count = self.get_metric(span_logits, span_labels, dev_callback)
                  mirco_metrics = np.sum(role_metric, axis=0)
                  mirco_metrics = get_p_r_f(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
                  p, r, f = mirco_metrics[0], mirco_metrics[1], mirco_metrics[2]
                  print('【eval】 precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2]))
                  if f > best_f1:
                    best_f1 = f
                    print("【best f1】 {:.4f}".format(mirco_metrics[2]))
                    self.save_model()
        if dev_loader is None:
          self.save_model()

    def test(self, test_loader):
        self.load_model()
        self.model.to(self.args.device)
        with torch.no_grad():
          span_logits, span_labels = self.eval_forward(test_loader)
          role_metric, total_count = self.get_metric(span_logits, span_labels, test_callback)
          print(self.args.labels)
          print(classification_report(role_metric, self.args.labels, {i:label for i,label in enumerate(self.args.labels)}, total_count))

    def predict(self, text):
        self.load_model()
        self.model.eval()
        self.model.to(self.args.device)
        with torch.no_grad():
            tmp_tokens = [i for i in text]
            encode_dict = self.args.tokenizer.encode_plus(text=tmp_tokens,
                                    max_length=self.args.max_seq_len,
                                    padding="max_length",
                                    truncating="only_first",
                                    return_token_type_ids=True,
                                    return_attention_mask=True)
            tokens = ['[CLS]'] + tmp_tokens + ['[SEP]']
            token_ids = torch.from_numpy(np.array(encode_dict['input_ids'])).unsqueeze(0).to(self.args.device)
            pinyin_ids = get_pinyin_ids(tmp_tokens, self.args.pinyin_dir)
            if len(pinyin_ids) > self.args.max_seq_len - 2:
              pinyin_ids = [[0] * 8] + pinyin_ids[:self.args.max_seq_len - 2] + [[0] * 8]
            else:
              pinyin_ids = [[0]*8] + pinyin_ids + [[0]*8] + [[0] * 8 for _ in range(self.args.max_seq_len - len(tmp_tokens) - 2)]
            attention_mask = torch.from_numpy(np.array(encode_dict['attention_mask'])).unsqueeze(0).to(
                self.args.device)
            pinyin_ids = torch.from_numpy(np.array(pinyin_ids)).unsqueeze(0).to(
                self.args.device)
            token_type_ids = torch.from_numpy(np.array(encode_dict['token_type_ids'])).unsqueeze(0).to(self.args.device)
            output = self.model(token_ids, pinyin_ids, token_type_ids, attention_mask, labels=None)
            loss, logit = output
            logit = logit.detach().cpu().numpy()
            logit = np.argmax(logit, -1)
            logit = logit[0][1:len(tmp_tokens)+1]
            print(bio_decode(tmp_tokens, logit, self.args.id2label, self.args.labels))


if __name__ == "__main__":
    class Args:
        data_name = "resume"
        save_dir = "checkpoints/{}/model_aug.pt".format(data_name)
        bert_dir = "/content/drive/MyDrive/chinese-bert-wwm-ext"
        pinyin_dir = "./data/pingyin/"
        glyph_dir = ["./data/glyph/" + font for font in ["STFangsong.ttf.npy", "STXingkai.ttf.npy", "方正古隶繁体.ttf.npy"]]
        with open("data/resume/mid_data/labels.txt", "r") as fp:
            labels = fp.read().strip().split("\n")
        label2id, id2label = conver_labels_to_biolabels(labels)
        print(label2id)
        print(id2label)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = BertTokenizer.from_pretrained(bert_dir, do_lower_case=False)

        train_epoch = 10
        max_seq_len = 128
        train_batch_size = 64
        eval_batch_size = 32
        eval_step = 100
        lr = 3e-5
        other_lr = 2e-3
        adam_epsilon = 1e-8
        warmup_proportion = 0.1
        max_grad_norm = 5
        weight_decay = 0.01
        num_labels = len(labels) * 2 + 1


    args = Args()

    with open("data/resume/mid_data/train.txt", "r") as fp:
        train_examples = fp.read().strip().split("\n")

    train_dataset = NerDataset(train_examples, args)

    with open("data/resume/mid_data/valid.txt", "r") as fp:
        dev_examples = fp.read().strip().split("\n")
    
    dev_callback = [args.tokenizer.tokenize(" ".join(json.loads(example)["text"]))[:args.max_seq_len-2] for example in dev_examples]
    dev_dataset = NerDataset(dev_examples, args)
 
    with open("data/resume/mid_data/test.txt", "r") as fp:
        test_examples = fp.read().strip().split("\n")
    print(test_examples[0])
    test_callback = [args.tokenizer.tokenize(" ".join(json.loads(example)["text"]))[:args.max_seq_len-2] for example in test_examples]
    test_dataset = NerDataset(test_examples, args)


    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=1)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.eval_batch_size, num_workers=1)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.eval_batch_size, num_workers=1)


    model = NerModel(args)

    nerPipeline = NerPipeline(model, args)
    nerPipeline.train(train_loader, dev_loader=dev_loader)
    nerPipeline.test(test_loader)

    example = {"id": 5, "text": ["陈", "学", "军", "先", "生", "：", "1", "9", "6", "7", "年", "5", "月", "出", "生", "，", "大", "学", "毕", "业", "，", "高", "级", "经", "济", "师", "。"], "labels": ["B-NAME", "I-NAME", "I-NAME", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-EDU", "I-EDU", "O", "O", "O", "B-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "O"]}
    text = example["text"]
    print("text=", "".join(text))
    nerPipeline.predict(text)

"""
【eval】 precision=0.9591 recall=0.9501 f1_score=0.9546
【best f1】 0.9546
['ORG', 'TITLE', 'CONT', 'NAME', 'PRO', 'LOC', 'RACE', 'EDU']
          precision    recall  f1-score   support

     ORG       0.93      0.93      0.93       551
   TITLE       0.95      0.94      0.95       762
    CONT       1.00      1.00      1.00        28
    NAME       0.99      1.00      1.00       112
     PRO       0.97      0.86      0.91        33
     LOC       0.67      0.80      0.73         6
    RACE       0.93      0.93      0.93        14
     EDU       0.97      0.94      0.96       112

micro-f1       0.95      0.94      0.94      1618
"""
