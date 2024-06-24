import pandas as pd
from imblearn.over_sampling import SMOTE
data = pd.read_excel('pfas_nalv.xlsx')
data = data.drop(['Data'], axis=1)
print(data.info())
print(data.groupby('Membrane type').count())
x = data.iloc[:, 1:9]
y = data.iloc[:, 0:1]

smote = SMOTE(random_state=9204, k_neighbors=3)
x_smote, y_smote = smote.fit_resample(x, y)
df_smote = pd.concat([x_smote, y_smote], axis=1)
print(df_smote.groupby('Membrane type').count())
