import os
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from data_utils import SentenceREDataset, get_idx2tag
from model import SentenceRE
# 训练

def train():

    epochs = 20
    max_len = 256

    train_batch_size = 32
    device = torch.device("cuda:1")

    learning_rate = 3e-5

    # train_dataset
    train_dataset = SentenceREDataset("datasets/train_known.jsonl", "datasets/relation.txt",256)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    # model
    idx2tag = get_idx2tag("datasets/relation.txt")
    tagset_size = len(idx2tag)
    model = SentenceRE(tagset_size).cuda()
    model.train()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for epoch in range(0, epochs):
        for i_batch, sample_batched in enumerate(tqdm(train_loader, desc='Training')):
            token_ids = sample_batched['token_ids'].to(device)
            token_type_ids = sample_batched['token_type_ids'].to(device)
            attention_mask = sample_batched['attention_mask'].to(device)
            e1_mask = sample_batched['e1_mask'].to(device)
            e2_mask = sample_batched['e2_mask'].to(device)
            tag_ids = sample_batched['tag_id'].to(device)
            logits = model(token_ids, token_type_ids, attention_mask, e1_mask, e2_mask)
            loss = criterion(logits, tag_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(epoch,loss)
        
train()
