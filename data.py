import pandas as pd
from sklearn.model_selection import train_test_split
from sampling import get_balance_dataset
from utils import make_index, generate_o_d
import torch
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#无过采样，标准化处理数据
def get_data(path, random):
    columns_list = ['Membrane type', 'Temperature (˚C)', 'PFOS con (ppb)', 'pH',
                    'Pressure (MPa)', 'Divalent cations (mmol/L)',
                    'Monovalent cations (mmol/L)', 'Trivalent cations (mmol/L)',
                    'PFOS rejection (%)']
    df_all = pd.read_excel(path)
    MemType = {1: 'NF270', 2: 'HYDRACORE', 3: 'PMIA', 4: 'NE70', 5: 'Poly(piperazineamide) NF', 6: 'ESNA1-K1'}
    Yc = ['PFOS rejection (%)']
    y = df_all[Yc]
    x = df_all.drop(['Data', 'PFOS rejection (%)'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.10, random_state=random)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, shuffle=True, test_size=0.111,
                                                      random_state=random)
    temp_df = pd.concat([x_train, y_train], axis=1)
    # temp_df=make_index(temp_df)
    train_data = pd.DataFrame(temp_df, columns=columns_list)
    print("train-data:\n", type(train_data))
    data_sampled = get_balance_dataset(data=train_data, column='Membrane type', random=random,
                                       columns_list=columns_list)
    # data_sampled.to_csv("过采样后的数据.csv")
    x_train = data_sampled.iloc[:, 0:8]
    y_train = data_sampled.iloc[:, 8:9]

    x_test = x_test[x_train.columns]
    x_val = x_val[x_train.columns]
    len_train = x_train.shape[0]
    len_val = x_val.shape[0]
    len_test = x_test.shape[0]
    x_new = pd.concat([x_train, x_val, x_test], axis=0)
    x_new = make_index(x_new)

    # x_new = generate_o_d(MemType, 'Membrane type', x_new)
    # # print(x_new)
    # # x_new.to_csv("111.csv")
    # x_new = pd.get_dummies(x_new)
    # # print(x_new)
    # x_new.to_csv("222.csv")
    # std = x_new.std()
    # mean = x_new.mean()
    # x_new = (x_new - mean) / std
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    x_new = scaler.fit_transform(x_new)
    x_new = pd.DataFrame(x_new)
    x_train = x_new.iloc[0:len_train, :]
    x_val = x_new.iloc[len_train:len_train + len_val, :]
    x_test = x_new.iloc[len_train + len_val:len_train + len_val + len_test, :]
    print(f"训练：{x_train.shape},验证：{x_val.shape}，测试：{x_test.shape}")
    x_train = torch.from_numpy(x_train.values).float()
    y_train = torch.from_numpy(y_train.values).float()
    x_test = torch.from_numpy(x_test.values).float()
    y_test = torch.from_numpy(y_test.values).float()
    x_val = torch.from_numpy(x_val.values).float()
    y_val = torch.from_numpy(y_val.values).float()

    return x_train, x_val, x_test, y_train, y_val, y_test


def get_dataloader(x_train, y_train):
    train_data = Data.TensorDataset(x_train, y_train)
    train_load = Data.DataLoader(
        dataset=train_data,
        batch_size=1024,
        shuffle=True,
    )
    return train_load
