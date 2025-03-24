import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix


wanted_temp = pd.read_csv('data/wanted_temperature.csv')
actual_temp = pd.read_csv('data/actual_temperature.csv')

combined_matrix = pd.DataFrame({
    'Wanted_Temp': wanted_temp['value'],
    'Actual_Temp': actual_temp['value']
})

scatter_matrix(combined_matrix, figsize=(10, 10))
plt.tight_layout()
plt.show()