import pandas as pd
import matplotlib.pyplot as plt
import os

folder_path = './'

excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

data_dict = {}

for file in excel_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_excel(file_path)

    epochs = [i for i in range(1, 100 + 1)]
    train_loss = df['Top5_accuracy']
    if len(train_loss) > 100:
        train_loss = train_loss.iloc[:-1]

    data_dict[file] = {
        'epochs': epochs,
        'train_loss': train_loss
    }

plt.figure(figsize=(10, 6))

for file, data in data_dict.items():
    plt.plot(data['epochs'], data['train_loss'], label=f'{file[:-5]}')

plt.title('MiniImageNet Top5 Accuracy', fontsize=24)
plt.xlabel('Epoch', fontsize=24)
plt.ylabel('Accuracy(%)', fontsize=24)
plt.legend()
plt.savefig("MiniImageNet Top5 Accuracy变化" ,dpi=600 )

plt.show()