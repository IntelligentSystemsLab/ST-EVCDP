import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# input data
data = pd.read_csv('results/plot_data.csv')
impulse = np.array(data['Impulse'], dtype=float).reshape(-1, 1)
Proposed = np.array(data['Proposed'], dtype=float).reshape(-1, 1)
GCN_LSTM = np.array(data['GCN-LSTM'], dtype=float).reshape(-1, 1)
GAT_LSTM = np.array(data['AST-GAT'], dtype=float).reshape(-1, 1)
GTA = np.array(data['PAG-'], dtype=float).reshape(-1, 1)

temp = np.ones_like(impulse)
GCN_LSTM_id = temp * 3
GAT_LSTM_id = temp * 2
GTA_id = temp * 1

ids = np.concatenate((GTA_id, GAT_LSTM_id, GCN_LSTM_id), axis=0)
impulses = np.concatenate((impulse, impulse, impulse), axis=0)
responses = np.concatenate((GTA, GAT_LSTM, GCN_LSTM), axis=0)

plot_data = np.concatenate((ids, impulses, responses), axis=1)
plot_data = pd.DataFrame(plot_data, columns=['ids', 'impulse', 'response'])

rc = {'font.sans-serif': ['Arial']}
sns.set(style='ticks', color_codes=True, font_scale=1.6, rc=rc)
sns.scatterplot(x='impulse', y='response', hue='ids', style='ids', palette="Set2", data=plot_data)
plt.ylim(-11, 11)
plt.twinx()
sns.scatterplot(x=np.squeeze(impulse), y=np.squeeze(Proposed))
plt.ylim(-0.6, 0.6)
plt.tight_layout()
plt.show()
