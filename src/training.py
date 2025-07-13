import torch
import torch.nn as nn
from tokenizer import get_tokenizer, get_raw_data, get_train_data
from torch.utils.data import Dataset, DataLoader
from CustomDataLoader import CustomDataset
from model import build_transformer

# filepath
dataset_path =  "/Users/shusanketbasyal/.cache/kagglehub/datasets/jigarpanjiyar/english-to-manipuri-dataset/versions/1"+"//english-nepali.xlsx"

# loading tokenizer
engtokenizer, neptokenizer = get_tokenizer(dataset_path)

# splitting the data
df_train, df_test  = get_train_data(dataset_path, engtokenizer, neptokenizer, split=True)

# creating dataset
df_train_dataset = CustomDataset(df_train, engtokenizer, neptokenizer, "eng", "nep", 256)
df_test_dataset = CustomDataset(df_test, engtokenizer, neptokenizer, "eng", "tgt", 256)

# data_loader
df_train_dataloader = DataLoader(df_train_dataset, batch_size=2, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transformer = build_transformer(
    engtokenizer.get_vocab_size(), 256, neptokenizer.get_vocab_size(), 256, 512, 4, 4, 8
).to(device)




# training loop
batch_size = 32
num_epochs = 1

optimizer = torch.optim.Adam(transformer.parameters(), lr = 1e-3, eps = 1e-9)
loss_fun = nn.CrossEntropyLoss(ignore_index=engtokenizer.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

for epoch in range(num_epochs):

  df_train_dataloader = DataLoader(df_train_dataset, batch_size=batch_size, shuffle=True)

  for x in df_train_dataloader:
      enc_input = x["encoder_input"].to(device)
      dec_input = x["decoder_input"].to(device)
      enc_mask = x["encoder_mask"].to(device)
      dec_mask = x["decoder_mask"].to(device)
      label = x["label"].to(device)

      encoder_output = transformer.encoder_fun(enc_input, enc_mask)

      decoder_output = transformer.decoder_fun(
          dec_input, encoder_output, enc_mask, dec_mask
      )
      logits = transformer.projection(decoder_output)
      # print(logits.view(-1, neptokenizer.get_vocab_size()).shape, label.view(-1).shape)
      loss = loss_fun(logits.view(-1, neptokenizer.get_vocab_size()), label.view(-1))
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
      print(loss.item())
