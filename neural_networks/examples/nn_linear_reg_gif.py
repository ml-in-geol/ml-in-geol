import numpy as np
import matplotlib.pyplot as plt

#true model parameters
w_true = 2.0
b_true = 0.6

#generate synthetic data
npts = 200
x = np.random.normal(loc=0.0,scale=2.0,size=npts)
noise = np.random.normal(loc=0.0,scale=1.0,size=npts)
y = w_true*x + b_true + noise

#initialize weight(s) and bias(es) as random number
w = np.random.random()
b = np.random.random()

#hyper parameters
learning_rate = 1e-5 #learning rate
nepochs = 500
all_loss = []

def plot_prediction(epochs,all_loss,x,y,y_pred,w,b):

    ymax = np.max(all_loss)*1.25
    fig,axes = plt.subplots(ncols=2,figsize=[9,4])
    #print('result: y = {:2.2f}x + {:2.2f}'.format(w,b))
    epochs_here = np.arange(0,epochs+1)
    axes[0].plot(epochs_here,all_loss)
    axes[0].set_xlabel('epochs',fontsize=14)
    axes[0].set_ylabel('loss',fontsize=14)
    axes[0].set_xlim([0,500])
    axes[0].set_ylim([0,ymax])

    axes[1].scatter(x,y,alpha=0.5)
    axes[1].plot(x,y_pred,c='k')
    axes[1].set_xlabel('x',fontsize=14)
    axes[1].set_ylabel('y',fontsize=14)
    axes[1].set_xlim([-10,10])
    axes[1].set_ylim([-15,15])

    #plot text
    txt1 = 'true: y = {:2.2f}x + {:2.2f}'.format(w_true,b_true)
    txt2 = 'model: y = {:2.2f}x + {:2.2f}'.format(w,b)
    axes[1].text(0.3,0.9,txt1, ha='center', va='center', transform=axes[1].transAxes)
    axes[1].text(0.3,0.8,txt2, ha='center', va='center', transform=axes[1].transAxes)

    plt.tight_layout()
    plt.savefig('ml_gif/{:04d}.jpg'.format(epochs))
    plt.close()

#main training loop
for i in range(0,nepochs):

    y_pred = w*x + b
    loss = 0.5*np.square(y_pred - y).sum()
    grad_y_pred = (y_pred - y)
    grad_b = grad_y_pred.sum()
    grad_w = (grad_y_pred*x).sum()
    all_loss.append(loss)

    b -= learning_rate * grad_b
    w -= learning_rate * grad_w

    if i % 10 == 0:
        print(i, loss,b,w)
        #print(grad_b,grad_w)
        plot_prediction(i,all_loss,x,y,y_pred,w,b)

