import torch
import torch.nn as nn
from tokenizer import get_tokenizer, get_train_data
from torch.utils.data import Dataset, DataLoader
from CustomDataLoader import CustomDataset
from model import build_transformer
from validation import validation_run
from utils import load_model_if

# filepath
dataset_path =  "/Users/shusanketbasyal/.cache/kagglehub/datasets/jigarpanjiyar/english-to-manipuri-dataset/versions/1"+"//english-nepali.xlsx"

# loading tokenizer
engtokenizer, neptokenizer = get_tokenizer(dataset_path)

# splitting the data
df_train, df_test  = get_train_data(dataset_path, engtokenizer, neptokenizer, split=True)

# creating dataset
df_train_dataset = CustomDataset(df_train, engtokenizer, neptokenizer, "eng", "nep", 256)
df_test_dataset = CustomDataset(df_test, engtokenizer, neptokenizer, "eng", "nep", 256)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transformer = build_transformer(
    engtokenizer.get_vocab_size(), 256, neptokenizer.get_vocab_size(), 256, 512, 4, 4, 8
).to(device)




# training loop
batch_size = 32
num_epochs = 1
optimizer = torch.optim.Adam(transformer.parameters(), lr = 1e-3, eps = 1e-9)
loss_fun = nn.CrossEntropyLoss(ignore_index=engtokenizer.token_to_id("[PAD]"), label_smoothing=0.1).to(device)
pre_load_model = load_model_if("./weights")
initial_epoch = 0

if pre_load_model is not None:
    # previous state exists
    state = torch.load(pre_load_model)
    transformer.load_state_dict(state['model_state_dict'])
    initial_epoch = state['epoch']+1
    optimizer.load_state_dict(state['optimizer_state_dict'])
    


for epoch in range(initial_epoch, num_epochs):
    df_train_dataloader = DataLoader(df_train_dataset, batch_size=batch_size, shuffle=True)
    iterators = len(df_train_dataloader)
    all_loss = torch.zeros(iterators)
    i = 0
    for x in df_train_dataloader:
        transformer.train()
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
        all_loss[i] = loss.item()
        i+=1
    
    with open(f"train_info/epoch", "a") as file:
        file.write(f"{epoch} epoch : \n")
        file.write(f"Training Loss=>{all_loss.mean()}\n")

    val_loss = validation_run(transformer, df_test_dataset, 8, 4, device, loss_fun, neptokenizer, engtokenizer, epoch)
    # after each epoch save the weights:
    filename = f"./weights/engtonepali{epoch}.pt"
    torch.save({
            'epoch': epoch,
            'model_state_dict': transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
    }, filename)
    break


# # Open a file in write mode ("w"). This will create the file if it doesn't exist.
# with open("example.txt", "w") as file:
#     file.write("Hello, world!\n")
#     file.write("This is a new line.")
