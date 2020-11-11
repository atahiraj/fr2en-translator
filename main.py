import argparse
import math
import time
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torchtext.vocab import GloVe, FastText, Vectors
from torchtext.data.metrics import bleu_score

from model import Seq2Seq, Decoder, Encoder
import config


def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SRC = Field(tokenize="spacy",
                tokenizer_language="fr",
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    TRG = Field(tokenize="spacy",
                tokenizer_language="en",
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    train_data, valid_data, test_data = Multi30k.splits(exts=('.fr', '.en'),
                                                        fields=(SRC, TRG))

    # Building the vocabs with the embeddings
    if config.EMB.lower() == "glove":
        en_embedding_vectors = GloVe(name='6B', dim=config.EMB_SIZE)
        fr_embedding_vectors = en_embedding_vectors
        
    elif config.EMB.lower() == "fasttext":
        en_embedding_vectors = FastText()
        fr_embedding_vectors = en_embedding_vectors

    else: # word2vec
        en_embedding_vectors = Vectors(name = './.vector_cache/enwiki_20180420_300d.txt')
        fr_embedding_vectors = Vectors(name = './.vector_cache/frwiki_20180420_300d.txt')

    SRC.build_vocab(train_data, min_freq=2, vectors=fr_embedding_vectors)
    TRG.build_vocab(train_data, min_freq=2, vectors=en_embedding_vectors)

    PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=config.BATCH_SIZE,
        device=device)

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    
    model = Seq2Seq(
        emb_dim=en_embedding_vectors.dim,
        src_vectors=SRC.vocab.vectors,
        trg_vectors=TRG.vocab.vectors,
        hid_dim=config.HID_DIM,
        n_layers=config.N_LAYERS,
        input_dim=INPUT_DIM,
        enc_dropout=config.ENC_DROPOUT,
        output_dim=OUTPUT_DIM,
        dec_dropout=config.DEC_DROPOUT,
        device=device
    ).to(device)

    
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    if config.TEST:
        model.load_state_dict(torch.load(config.LOAD))
        test_loss = evaluate(model, test_iterator, criterion)
        bleu_score = calculate_bleu(test_data, SRC, TRG, model, device)
        print(
            f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} | BLEU: {100 * bleu_score:.3f}')

    else:
        model.apply(init_weights)

        def dfs_freeze(model):
            for param in model.parameters():
                param.requires_grad = False
            for name, child in model.named_children():
                dfs_freeze(child)

        #dfs_freeze(model.encoder.embedding)
        #dfs_freeze(model.decoder.embedding)

        optimizer = optim.Adam(
            filter(
                lambda p: p.requires_grad,
                model.parameters()))

        def count_parameters(model: nn.Module):
            return sum(p.numel()
                       for p in model.parameters() if p.requires_grad)

        print(
            f'The model has {count_parameters(model):,} trainable parameters')

        CLIP = 1
        best_valid_loss = float('inf')

        for epoch in range(config.N_EPOCHS):

            start_time = time.time()

            train_loss = train(
                model,
                train_iterator,
                optimizer,
                criterion,
                CLIP)
            valid_loss = evaluate(model, valid_iterator, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), config.SAVE)

            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(
                f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(
                f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


def train(model, iterator, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):

        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

            #print(trg)
            #print(output)
            #exit(0)

    return epoch_loss / len(iterator)


def calculate_bleu(data, src_field, trg_field, model, device, max_len = 50):
    
    trgs = []
    pred_trgs = []
    
    for datum in data:
        
        src = vars(datum)['src']
        trg = vars(datum)['trg']
        
        pred_trg = translate_sentence(src, src_field, trg_field, model, device, max_len)
        
        #cut off <eos> token
        pred_trg = pred_trg[:-1]
        
        pred_trgs.append(pred_trg)
        trgs.append([trg])
        
    return bleu_score(pred_trgs, trgs)

def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):

    model.eval()
        
    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    src_len = torch.LongTensor([len(src_indexes)]).to(device)
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    
    for i in range(max_len):

        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
                
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden)
            
        pred_token = output.argmax(1).item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:]

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


if __name__ == "__main__":
    main()
