import os
import matplotlib.pyplot as plt
import pandas as pd



def plot(df,x):
    
    acc = max(df['val_acc'])
    
    plt.plot(df['train_acc'])
    plt.plot(df['val_acc'])
    plt.title(x+' model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


    plt.plot(df['train_loss'])
    plt.plot(df['val_loss'])
    plt.title(x+' model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='center left')
    plt.show()
    print('='*20)
    print(f'Accuracy : {acc*100: .3f}')
    print('='*20)
    
res = ['aug','myc18_decay4_rnn','mywd6','resnet18LSTM','resnetaug']

def run():
    path='./models'
    for x in res:
        p = os.path.join(path,x+'/res.csv')
        df = pd.read_csv(p)
        plot(df,x)
    
if __name__=='__main__':
    run()
    