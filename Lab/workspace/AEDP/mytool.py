import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def calRMSE(model, x, y, batch_size):
    nRow = x.shape[0]
    nBatch = int(np.ceil(nRow / batch_size))
    rmse = 0
    for j in range(nBatch):
        s_id = j * batch_size
        e_id = min(nRow, (j + 1) * batch_size)
        d = x[s_id:e_id, :, :, :]
        o = model.predict(d)
        #print(f'{j}: {o.shape}, {s_id} {e_id}')
        rmse += np.sum(np.power(o.reshape(-1,) - y[s_id:e_id], 2))
    rmse = np.sqrt(rmse / nRow)
    return rmse


def drawHistory(hist):
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    if 'val_loss' in hist.history:
        loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')

    if 'accuracy' in hist.history:
        acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
    if 'val_accuracy' in hist.history:
        acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='lower left')

    plt.show()

#box plot그리기
def drawBoxPlot(o_test, y_test, n_class):
    nTest = len(y_test)
    o_test_4boxplot = np.zeros([nTest,n_class])
    o_test_4boxplot[:] = np.nan
    cnt = np.zeros(n_class,dtype=np.int)
    for i in range(nTest):
        idx = min( int(y_test[i]*n_class), n_class-1)  # ids= 나이대  #  왜  min해서 n_class-1 을 넣지? 6이 나올수 있어서? <- 이건듯
        o_test_4boxplot[cnt[idx],idx] = (o_test[i] * n_class + 2) *10
        cnt[idx] +=1
    o_test_4boxplot = o_test_4boxplot[:np.max(cnt),:]

    df_o_test_4boxplot= pd.DataFrame(o_test_4boxplot, columns=list(map(lambda x:f'{(x+2)*10}', range(n_class))) )

    sns.boxplot(data=df_o_test_4boxplot)
    plt.xlabel('Target')
    plt.ylabel('Output')
    plt.show()