import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import requests
from io import BytesIO



def makeGraphAndSendPhoto(chatid,filePath,saveDir):
    df = pd.read_csv(filePath)
    df['index'] = np.arange(1, len(df) + 1)
    df.set_index('index', inplace=True)
    metrics_pairs = [('f1_score_test', 'f1_score_train'), ('recall_test', 'recall_train'),("precision_test", "precision_train"),("test_accuracy","train_accuracy"),("test_loss","train_loss")]
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 16), sharex=True)

    for i, (metric1, metric2) in enumerate(metrics_pairs):
        row = i // 2
        col = i % 2
        axes[row,col].plot(df.index, df[metric1], marker='o', label=metric1)
        axes[row,col].plot(df.index, df[metric2], marker='x', label=metric2)
        axes[row,col].set_title(f'{metric1.capitalize()} and {metric2.capitalize()} Over Time')
        axes[row,col].set_ylabel(metric1.capitalize())
        axes[row,col].set_xlabel("Epochs")
        axes[row,col].legend()
        axes[row,col].grid(True)

    axes[2, 1].plot(df.index, df['learning_rate'], marker='o', color='r', label='Learning Rate')
    axes[2, 1].set_title('Learning Rate vs Epoch')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Learning Rate')
    axes[2, 1].legend()
    axes[2, 1].grid(True)


    plt.xlabel('Epochs')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.savefig(saveDir, format='png')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close(fig)

    heathcliff05_bot_url = "https://api.telegram.org/<config>/sendPhoto"

    files = {
            'photo': ('plot2.png', buffer, 'image/png')
    }

    data = {
            'chat_id': chatid
    }
    response = requests.post(heathcliff05_bot_url, data=data, files=files)
    print(response.json())

if __name__== "__main__":
    filePath = ""
    caption = ""
    chatid = ""
    savedir = ""
    if len(sys.argv) > 1:
        l = len(sys.argv)
        for i in range(l):
            if sys.argv[i] == "--filepath":
                arg = sys.argv[i+1]
                filePath = arg
            if sys.argv[i] == "--chatid":
                arg = sys.argv[i+1]
                chatid = arg
            if sys.argv[i] == "--savedir":
                arg = sys.argv[i+1]
                savedir = arg
    makeGraphAndSendPhoto(chatid,filePath,savedir)
