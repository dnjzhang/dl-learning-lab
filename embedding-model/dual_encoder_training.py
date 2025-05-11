#!/usr/bin/env python
# coding: utf-8
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd

from transformers import AutoTokenizer


class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, output_embed_dim):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(vocab_size, embed_dim)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(embed_dim, nhead=8, batch_first=True),
            num_layers=3,
            norm=torch.nn.LayerNorm([embed_dim]),
            enable_nested_tensor=False
        )
        self.projection = torch.nn.Linear(embed_dim, output_embed_dim)

    def forward(self, tokenizer_output):
        x = self.embedding_layer(tokenizer_output['input_ids'])
        x = self.encoder(x, src_key_padding_mask=tokenizer_output['attention_mask'].logical_not())
        cls_embed = x[:, 0, :]
        return self.projection(cls_embed)


def train_loop(dataset):
    embed_size = 512
    output_embed_size = 128
    max_seq_len = 64
    batch_size = 32

    # define the question/answer encoders
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    question_encoder = Encoder(tokenizer.vocab_size, embed_size,
                               output_embed_size)
    answer_encoder = Encoder(tokenizer.vocab_size, embed_size,
                             output_embed_size)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True)
    optimizer = torch.optim.Adam(
        list(question_encoder.parameters()) + list(answer_encoder.parameters()
    ), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    running_loss = []
    for _, data_batch in enumerate(dataloader):

        # Tokenize the question/answer pairs (each is a batch of 32 questions and 32 answers)
        question, answer = data_batch
        question_tok = tokenizer(question, padding=True, truncation=True, return_tensors='pt', max_length=max_seq_len)
        answer_tok = tokenizer(answer, padding=True, truncation=True, return_tensors='pt', max_length=max_seq_len)

        # Compute the embeddings: the output is of dim = 32 x 128
        question_embed = question_encoder(question_tok)
        answer_embed = answer_encoder(answer_tok)

        # Compute similarity scores: a 32x32 matrix
        # row[N] reflects similarity between question[N] and answers[0...31]
        similarity_scores = question_embed @ answer_embed.T

        # we want to maximize the values in the diagonal
        target = torch.arange(question_embed.shape[0], dtype=torch.long)
        loss = loss_fn(similarity_scores, target)
        running_loss += [loss.item()]

        # this is where the magic happens
        optimizer.zero_grad()    # reset optimizer so gradients are all-zero
        loss.backward()
        optimizer.step()

    return question_encoder, answer_encoder


def train(dataset, num_epochs=10):
    embed_size = 512
    output_embed_size = 128
    max_seq_len = 64
    batch_size = 32

    n_iters = len(dataset) // batch_size + 1

    # define the question/answer encoders
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    question_encoder = Encoder(tokenizer.vocab_size, embed_size, output_embed_size)
    answer_encoder = Encoder(tokenizer.vocab_size, embed_size, output_embed_size)

    # define the dataloader, optimizer and loss function
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(list(question_encoder.parameters()) + list(answer_encoder.parameters()), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = []
        for idx, data_batch in enumerate(dataloader):

            # Tokenize the question/answer pairs (each is a batc of 32 questions and 32 answers)
            question, answer = data_batch
            question_tok = tokenizer(question, padding=True, truncation=True, return_tensors='pt',
                                     max_length=max_seq_len)
            answer_tok = tokenizer(answer, padding=True, truncation=True, return_tensors='pt', max_length=max_seq_len)
            if idx == 0 and epoch == 0:
                print(question_tok['input_ids'].shape, answer_tok['input_ids'].shape)

            # Compute the embeddings: the output is of dim = 32 x 128
            question_embed = question_encoder(question_tok)
            answer_embed = answer_encoder(answer_tok)
            if idx == 0 and epoch == 0:
                print(question_embed.shape, answer_embed.shape)

            # Compute similarity scores: a 32x32 matrix
            # row[N] reflects similarity between question[N] and answers[0...31]
            similarity_scores = question_embed @ answer_embed.T
            if idx == 0 and epoch == 0:
                print(similarity_scores.shape)

            # we want to maximize the values in the diagonal
            target = torch.arange(question_embed.shape[0], dtype=torch.long)
            loss = loss_fn(similarity_scores, target)
            running_loss += [loss.item()]
            if idx == n_iters - 1:
                print(f"Epoch {epoch}, loss = ", np.mean(running_loss))

            # this is where the magic happens
            optimizer.zero_grad()  # reset optimizer so gradients are all-zero
            loss.backward()
            optimizer.step()

    return question_encoder, answer_encoder


# ## Let's train

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, datapath):
        self.data = pd.read_csv(datapath, sep="\t", nrows=300)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]['questions'], self.data.iloc[idx]['answers']


dataset = MyDataset('./shared_data/nq_sample.tsv')
dataset.data.head(5)

qe, ae = train(dataset, num_epochs=5)

# Use Encoders to find relevant answers:
print("Demonstrate how Embedding database query works ...")

question = 'What is the tallest mountain in the world?'
print("Question: " + question)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
question_tok = tokenizer(question, padding=True, truncation=True, return_tensors='pt', max_length=64)
question_emb = qe(question_tok)[0]
print(question_tok)
print(question_emb[:5])

answers = [
    "What is the tallest mountain in the world?",
    "The tallest mountain in the world is Mount Everest.",
    "Who is donald duck?"
]
answer_tok = []
answer_emb = []
for answer in answers:
    tok = tokenizer(answer, padding=True, truncation=True, return_tensors='pt', max_length=64)
    answer_tok.append(tok['input_ids'])
    emb = ae(tok)[0]
    answer_emb.append(emb)
print("Answers tokens:")
print(answer_tok)
print("Answers embeddings:")
print(answer_emb[0][:5])
print(answer_emb[1][:5])
print(answer_emb[2][:5])

print("Relevance ranking using DOT products:")
print(question_emb @ torch.stack(answer_emb).T)