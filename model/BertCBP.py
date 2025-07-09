# This file is the Prompt-CBP model structure

import torch
import torch.nn

from transformers import BertModel

class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
        self.trans = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, config.num_hidden_layers * 2 * config.hidden_size)
        )

    def forward(self, prefix: torch.Tensor):
        prefix_tokens = self.embedding(prefix)
        past_key_values = self.trans(prefix_tokens)
        return past_key_values


class BertCBP(torch.nn.Module):
    def __init__(self, config):
        super(BertCBP, self).__init__()
        self.bert = BertModel.from_pretrained("E:\pythonProject\BERT-CBP\DNABERT-2-117M", trust_remote_code=True)
        self.mode = config.mode
        self.config = config
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        self.conv1d = torch.nn.Conv1d(config.hidden_size, config.hidden_size * 2, kernel_size=3, padding=1)
        self.cnn_activation = torch.nn.ReLU()
        # self.pooling = nn.MaxPool1d(2, stride=2)
        self.pooling = torch.nn.AvgPool1d(2, stride=2)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.bn = torch.nn.BatchNorm1d(config.hidden_size * 2)
        self.lstm = torch.nn.LSTM(input_size=config.hidden_size * 2, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)

        self.classifier = torch.nn.Linear(128 * 2, 2)  #
        # self.init_weights()

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        # print(past_key_values.shape)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        # print(past_key_values[0].shape)

        return past_key_values

    def forward(
            self,
            input_ids=None
    ):
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        if self.mode == 1:
            outputs = self.bert(input_ids=input_ids)
        elif self.mode == 2:
            outputs = self.bert(input_ids=input_ids, past_key_values=past_key_values)
        elif self.mode == 3:
            outputs = self.bert(input_ids=input_ids)
        elif self.mode == 4:
            outputs = self.bert(input_ids=input_ids, past_key_values=past_key_values)

        bert_output = outputs[0]
        # print(bert_output.shape)
        sequence_output = self.cnn_activation(self.conv1d(bert_output.permute(0, 2, 1)))
        # print(sequence_output.shape)
        sequence_output = self.pooling(sequence_output)
        sequence_output = self.dropout(sequence_output)

        sequence_output = self.bn(sequence_output)
        sequence_output = sequence_output.permute(0, 2, 1)
        # print(sequence_output.shape)

        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output)
        lstm_output = lstm_output[:, -1, :]

        logits = self.classifier(lstm_output)

        return logits
