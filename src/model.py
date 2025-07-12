from tokenizer import get_tokenizer, get_data
from torch.utils.data import Dataset, DataLoader
from CustomDataLoader import CustomDataset


dataset_path =  "/Users/shusanketbasyal/.cache/kagglehub/datasets/jigarpanjiyar/english-to-manipuri-dataset/versions/1"+"//english-nepali.xlsx"

engtokenizer, neptokenizer  = get_tokenizer(dataset_path)

df_train, df_test = get_data(dataset_path, split=True)

df_train_dataset = CustomDataset(df_train, engtokenizer, neptokenizer, "eng", "nep", 256)
df_test_dataset = CustomDataset(df_test, engtokenizer, neptokenizer, "eng", "nep", 256)


df_train_dataloader = DataLoader(df_train_dataset, batch_size=2, shuffle=True)

