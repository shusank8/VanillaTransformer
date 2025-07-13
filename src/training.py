from tokenizer import get_tokenizer, get_data
from torch.utils.data import Dataset, DataLoader
from CustomDataLoader import CustomDataset

# filepath
dataset_path =  "/Users/shusanketbasyal/.cache/kagglehub/datasets/jigarpanjiyar/english-to-manipuri-dataset/versions/1"+"//english-nepali.xlsx"

# loading tokenizer
engtokenizer, neptokenizer = get_tokenizer(dataset_path)

# splitting the data
df_train, df_test  = get_data(dataset_path, split=True)

# creating dataset
df_train_dataset = CustomDataset(df_train, engtokenizer, neptokenizer, "eng", "nep", 256)
df_test_dataset = CustomDataset(df_test, engtokenizer, neptokenizer, "eng", "tgt", 256)

# data_loader
df_train_dataloader = DataLoader(df_train_dataset, batch_size=2, shuffle=True)
