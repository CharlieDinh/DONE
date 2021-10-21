import matplotlib.pyplot as plt
import h5py
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.ticker import StrMethodFormatter
import os
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 12})

def simple_read_data(alg):
    print(alg)
    hf = h5py.File("./results/"+'{}.h5'.format(alg), 'r')
    rs_glob_acc = np.array(hf.get('rs_glob_acc')[:])
    rs_train_acc = np.array(hf.get('rs_train_acc')[:])
    rs_train_loss = np.array(hf.get('rs_train_loss')[:])
    nanmax = np.nanmax(rs_train_loss)
    rs_train_loss[np.isnan(rs_train_loss)] = nanmax
    rs_train_loss[rs_train_loss > 10] = 2.0
    return rs_train_acc, rs_train_loss, rs_glob_acc

def get_training_data_value(num_users=[], loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], alpha =[], eta =[], algorithms_list=[], batch_size=[], kappa=[], dataset=""):
    Numb_Algs = len(algorithms_list)
    train_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    train_loss = np.zeros((Numb_Algs, Numb_Glob_Iters))
    glob_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    algs_lbl = algorithms_list.copy()
    for i in range(Numb_Algs):
        string_learning_rate = str(learning_rate[i])  
        string_learning_rate = string_learning_rate  + "_" + str(alpha[i])  + "_" + str(eta[i])  + "_" + str(lamb[i])
        algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users[i]) + "u" + "_" + str(batch_size[i]) + "b"  "_" +str(loc_ep1[i])
        if kappa:
            algorithms_list[i] = algorithms_list[i] + "_"+ str(kappa[i])
        train_acc[i, :], train_loss[i, :], glob_acc[i, :] = np.array(
            simple_read_data(dataset +"_"+ algorithms_list[i] + "_avg"))[:, :Numb_Glob_Iters]
        algs_lbl[i] = algs_lbl[i]
    return glob_acc, train_acc, train_loss

def get_all_training_data_value(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=0, learning_rate=0, alpha = 0, eta = 0, algorithms="", batch_size=0, dataset="" ,times = 5):
    train_acc = np.zeros((times, Numb_Glob_Iters))
    train_loss = np.zeros((times, Numb_Glob_Iters))
    glob_acc = np.zeros((times, Numb_Glob_Iters))
    algorithms_list  = [algorithms] * times
    for i in range(times):
        string_learning_rate = str(learning_rate)  
        string_learning_rate = string_learning_rate  + "_" + str(alpha)  + "_" + str(eta) + "_" + str(lamb)
        algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size) + "b"  "_" +str(loc_ep1) +  "_" +str(i)
        train_acc[i, :], train_loss[i, :], glob_acc[i, :] = np.array(simple_read_data(dataset +"_"+ algorithms_list[i]))[:, :Numb_Glob_Iters]
    
    return glob_acc, train_acc, train_loss

def get_data_label_style(input_data = [], linestyles= [], algs_lbl = [], lamb = [], loc_ep1 = 0, batch_size =0):
    data, lstyles, labels = [], [], []
    for i in range(len(algs_lbl)):
        data.append(input_data[i, ::])
        lstyles.append(linestyles[i])
        labels.append(algs_lbl[i]+str(lamb[i])+"_" +
                      str(loc_ep1[i])+"e" + "_" + str(batch_size[i]) + "b")

    return data, lstyles, labels

def average_data(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb="", learning_rate="", alpha="", eta="", algorithms="", batch_size=0, dataset = "", times = 5):
    glob_acc, train_acc, train_loss = get_all_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate,alpha,eta, algorithms, batch_size, dataset,times)
    glob_acc_data = np.average(glob_acc, axis=0)
    train_acc_data = np.average(train_acc, axis=0)
    train_loss_data = np.average(train_loss, axis=0)
    # store average value to h5 file
    max_accuracy = []
    for i in range(times):
        max_accuracy.append(glob_acc[i].max())
    
    print("std:", np.std(max_accuracy))
    print("Mean:", np.mean(max_accuracy))

    alg = dataset + "_" + algorithms
    alg = alg + "_" + str(learning_rate)+ "_" + str(alpha) + "_" + str(eta) + "_" + str(lamb) + "_" + str(num_users) + "u" + "_" + str(batch_size) + "b" + "_" + str(loc_ep1)
    alg = alg + "_" + "avg"
    if (len(glob_acc) != 0 &  len(train_acc) & len(train_loss)) :
        with h5py.File("./results/"+'{}.h5'.format(alg,loc_ep1), 'w') as hf:
            hf.create_dataset('rs_glob_acc', data=glob_acc_data)
            hf.create_dataset('rs_train_acc', data=train_acc_data)
            hf.create_dataset('rs_train_loss', data=train_loss_data)
            hf.close()

def plot_summary_one_figure(num_users=[], loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], alpha = [], eta = [], algorithms_list=[], batch_size=0, kappa =[], dataset = ""):
    Numb_Algs = len(algorithms_list)
    dataset = dataset
    #glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, hyper_learning_rate, algorithms_list, batch_size, dataset)
    
    #glob_acc =  average_smooth(glob_acc_, window='flat')
    ##train_loss = average_smooth(train_loss_, window='flat')
    #train_acc = average_smooth(train_acc_, window='flat')
    algs_lbl = algorithms_list.copy()
    glob_acc, train_acc, train_loss = get_training_data_value( num_users=num_users, loc_ep1=loc_ep1, Numb_Glob_Iters=Numb_Glob_Iters, lamb=lamb, learning_rate=learning_rate, alpha =alpha, eta =eta, algorithms_list=algorithms_list, batch_size=batch_size,kappa = kappa, dataset= dataset)


    print("max value of test accuracy",glob_acc.max())
    plt.figure(1,figsize=(5, 5))
    MIN = train_loss.min() - 0.001
    start = 0
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    for i in range(Numb_Algs):
        plt.plot(train_acc[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + "_"+str(loc_ep1[i])+"e" + "_" + str(batch_size[i]) + "b"  + "_" + str(learning_rate[i])  + "_" + str(alpha[i])  + "_" + str(eta[i]) )
    plt.legend(loc='lower right')
    plt.ylabel('Training Accuracy')
    plt.xlabel('Global rounds ' + '$K_g$')
    plt.title(dataset.upper())
    plt.ylim([0.88, train_acc.max() + 0.01])
    plt.savefig(dataset.upper() + str(loc_ep1[0]) + 'train_acc.png')
    #plt.savefig(dataset + str(loc_ep1[1]) + 'train_acc.pdf')
    plt.figure(2,figsize=(5, 5))
    for i in range(Numb_Algs):
        plt.plot(train_loss[i, start:], linestyle=linestyles[i], label=algs_lbl[i] + "_"+str(loc_ep1[i])+"e" + "_" + str(batch_size[i]) + "b" + "_" + str(learning_rate[i])  + "_" + str(alpha[i])  + "_" + str(eta[i]))
        #plt.plot(train_loss1[i, 1:], label=algs_lbl1[i])
    plt.legend(loc='upper right')
    plt.ylim([ 0.01, 0.5]) #set_ylim([0.049, 0.1])
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    plt.title(dataset.upper())
    #plt.ylim([train_loss.min(), 1])
    plt.savefig(dataset.upper() + str(loc_ep1[0]) + 'train_loss.png')
    #plt.savefig(dataset + str(loc_ep1[1]) + 'train_loss.pdf')
    plt.figure(3)
    for i in range(Numb_Algs):
        plt.plot(glob_acc[i, start:], linestyle=linestyles[i],label=algs_lbl[i]+ "_" +str(loc_ep1[i])+"e" + "_" + str(batch_size[i]) + "b" + "_" + str(learning_rate[i])  + "_" + str(alpha[i])  + "_" + str(eta[i]))
        #plt.plot(glob_acc1[i, 1:], label=algs_lbl1[i])  
    plt.legend(loc='lower right')
    #plt.ylim([0.6, glob_acc.max()])
    plt.ylim([0.88,  glob_acc.max() + 0.001])
    plt.ylabel('Testing Accuracy')
    plt.xlabel('Global rounds ')
    plt.title(dataset.upper())
    plt.savefig(dataset.upper() + str(loc_ep1[0]) + 'glob_acc.png')
    #plt.savefig(dataset + str(loc_ep1[1]) + 'glob_acc.pdf')



def get_max_value_index(num_users=[], loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], algorithms_list=[], batch_size=0, dataset=""):
    Numb_Algs = len(algorithms_list)
    glob_acc, train_acc, train_loss = get_training_data_value( num_users=num_users, loc_ep1=loc_ep1, Numb_Glob_Iters=Numb_Glob_Iters, lamb=lamb, learning_rate=learning_rate, alpha =alpha, eta =eta, algorithms_list=algorithms_list, batch_size=batch_size, dataset= dataset)

    for i in range(Numb_Algs):
        print("Algorithm: ", algorithms_list[i], "Max testing Accuracy: ", glob_acc[i].max(
        ), "Index: ", np.argmax(glob_acc[i]), "local update:", loc_ep1[i])

def average_smooth(data, window_len=20, window='hanning'):
    results = []
    if window_len<3:
        return data
    for i in range(len(data)):
        x = data[i]
        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('numpy.'+window+'(window_len)')

        y=np.convolve(w/w.sum(),s,mode='valid')
        results.append(y[window_len-1:])
    return np.array(results)


def plot_loss_differences(differences, log_scale=True):
    plt.figure()
    plt.plot(differences)
    if log_scale:
        plt.yscale("log")
    plt.xlabel("$t$")
    plt.ylabel("$f(w^t) - f(w^*)$")
    plt.savefig("loss_differences.png", bbox_inches='tight')


def plot_summary_linear(num_users=[], loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], alpha = [], eta = [], algorithms_list=[], batch_size=0, kappa = [], dataset = ""):

    Numb_Algs = len(algorithms_list)
    glob_acc, train_acc, train_loss = get_training_data_value( num_users=num_users, loc_ep1=loc_ep1, Numb_Glob_Iters=Numb_Glob_Iters, lamb=lamb, learning_rate=learning_rate, alpha =alpha, eta =eta, algorithms_list=algorithms_list, batch_size=batch_size, kappa=kappa, dataset= dataset)
    for i in range(Numb_Algs):
        print(algorithms_list[i], "loss:", glob_acc[i].max())
    
    plt.figure(1)
    linestyles = ['-','-', '--', '-.', '-.', ':']
    markers = ["o","v","s","*","x","P"]
    algs_lbl = ["DONE","DONE", "Newton","DANE", "FedDANE", "FirstOrder"]
    #plt.figure(figsize=(6,12))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    #fig, (ax1, ax2) = plt.subplots(1, 2)

    num_al = 6

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$B = $' + stringbatch ,marker = markers[i],markevery=0.2, markersize=5)

    #fig.hlines(y=0.035,xmin=0, xmax=200, linestyle='--',label = "optimal solution", color= "m" )
    ax1.legend(loc='upper right')
    ax1.set_ylim([0.045, 0.2])
    ax1.grid(True)
    ax1.set_title('$\\kappa = $' + str(kappa[0]))

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax2.plot(train_loss[i+num_al, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$B = $' + stringbatch ,marker = markers[i],markevery=0.2, markersize=5)

    ax2.set_ylim([0.045, 0.2])

    #plt.title('$\\kappa = $' + str(kappa))
    #fig.set_title('Linear Synthetic')
    ax2.grid(True)
    ax2.set_title('$\\kappa = $' + str(kappa[-1]))
    #ax1.set_ylim([0.045, 0.2])
    ax.set_xlabel('Global rounds ' + '$T$')
    ax.set_ylabel('Training Loss', labelpad = 10)
    #plt.xticks(np.arange(0.045, 2, 0.1))
    plt.savefig(dataset + str(loc_ep1[1]) + 'train_loss.pdf', bbox_inches='tight')
    plt.savefig(dataset + str(loc_ep1[1]) + 'train_loss.png', bbox_inches='tight')

def plot_summary_mnist(num_users=[], loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], alpha = [], eta = [], algorithms_list=[], batch_size=0, dataset = ""):
    Numb_Algs = len(algorithms_list)
    dataset = dataset

    algs_lbl = algorithms_list.copy()
    glob_acc, train_acc, train_loss = get_training_data_value( num_users=num_users, loc_ep1=loc_ep1, Numb_Glob_Iters=Numb_Glob_Iters, lamb=lamb, learning_rate=learning_rate, alpha =alpha, eta =eta, algorithms_list=algorithms_list, batch_size=batch_size, dataset= dataset)

    linestyles = ['-','-', '--', '-.', '-.', ':']
    markers = ["o","v","s","*","x","P"]
    algs_lbl = ["DONE","DONE", "Newton","DANE", "FedDANE", "GD"]

    print("max value of test accuracy",glob_acc.max())
    
    start = 0

    plt.figure(1,figsize=(7, 6))
    for i in range(Numb_Algs):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        plt.plot(train_acc[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$B = $' + stringbatch ,marker = markers[i],markevery=0.2, markersize=5)
        #plt.plot(train_acc[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + "_"+str(loc_ep1[i])+"e" + "_" + str(batch_size[i]) + "b"  + "_" + str(learning_rate[i])  + "_" + str(alpha[i])  + "_" + str(eta[i]) )
    plt.legend(loc='lower right')
    plt.ylabel('Training Accuracy')
    plt.xlabel('Global rounds ' + '$T$')
    plt.title(dataset.upper())
    plt.grid(True)
    plt.ylim([0.86, train_acc.max() + 0.01])
    plt.savefig(dataset.upper() +'train_acc.png')
    plt.savefig(dataset.upper() +'train_acc.pdf')
    #plt.savefig(dataset + str(loc_ep1[1]) + 'train_acc.pdf')

    plt.figure(2,figsize=(7, 6))
    for i in range(Numb_Algs):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$B = $' + stringbatch ,marker = markers[i],markevery=0.2, markersize=5)
        #plt.plot(train_loss[i, start:], linestyle=linestyles[i], label=algs_lbl[i] + "_"+str(loc_ep1[i])+"e" + "_" + str(batch_size[i]) + "b" + "_" + str(learning_rate[i])  + "_" + str(alpha[i])  + "_" + str(eta[i]))
        #plt.plot(train_loss1[i, 1:], label=algs_lbl1[i])
    plt.legend(loc='upper right')
    plt.ylim([0.2, 0.5])
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds ' + '$T$')
    plt.title(dataset.upper())
    plt.grid(True)
    #plt.ylim([train_loss.min(), 1])
    plt.savefig(dataset.upper() + 'train_loss.png')
    plt.savefig(dataset.upper() + 'train_loss.pdf')
    #plt.savefig(dataset + str(loc_ep1[1]) + 'train_loss.pdf')

    plt.figure(3,figsize=(7, 6))
    for i in range(Numb_Algs):
        #plt.plot(glob_acc[i, start:], linestyle=linestyles[i],label=algs_lbl[i]+ "_" +str(loc_ep1[i])+"e" + "_" + str(batch_size[i]) + "b" + "_" + str(learning_rate[i])  + "_" + str(alpha[i])  + "_" + str(eta[i]))
        #plt.plot(glob_acc1[i, 1:], label=algs_lbl1[i])
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$B = $' + stringbatch ,marker = markers[i],markevery=0.2, markersize=5)
       
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.ylim([0.86, 0.922])
    plt.ylabel('Testing Accuracy')
    plt.xlabel('Global rounds ' + '$T$')
    plt.title(dataset.upper())
    plt.savefig(dataset.upper() + 'glob_acc.png')
    plt.savefig(dataset.upper() +  'glob_acc.pdf')
    #plt.savefig(dataset + str(loc_ep1[1]) + 'glob_acc.pdf')

def plot_summary_mnist2(num_users=[], loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], alpha = [], eta = [], algorithms_list=[], batch_size=0, dataset = ""):
    Numb_Algs = len(algorithms_list)
    dataset = dataset

    algs_lbl = algorithms_list.copy()
    glob_acc, train_acc, train_loss = get_training_data_value( num_users=num_users, loc_ep1=loc_ep1, Numb_Glob_Iters=Numb_Glob_Iters, lamb=lamb, learning_rate=learning_rate, alpha =alpha, eta =eta, algorithms_list=algorithms_list, batch_size=batch_size, dataset= dataset)
    
    for i in range(Numb_Algs):
        print(algorithms_list[i], "acc:", glob_acc[i].max())
        print(algorithms_list[i], "loss:", train_loss[i].min())

    linestyles = ['-', '-', '-', '--', '--', '-.', ':']
    markers = ["o", "v", "s", "*", "x", "P", "+"]
    algs_lbl = ["DONE", "DONE", "DONE", "Newton", "Newton", "DANE", "GD"]

   #plt.figure(figsize=(6,12))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    #fig, (ax1, ax2) = plt.subplots(1, 2)

    num_al = 7

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0' and algs_lbl[i] != "DANE"):
            stringbatch = '$\infty$' + ", " + \
                '$\\alpha$' + " = " + str(alpha[i])
        elif(stringbatch == '0'):
             stringbatch = '$\infty$'
        
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$B = $' + stringbatch, marker = markers[i],markevery=0.2, markersize=7)

    #fig.hlines(y=0.035,xmin=0, xmax=200, linestyle='--',label = "optimal solution", color= "m" )
    ax1.legend(loc='upper right')
    ax1.set_ylim([0.2, 0.5])
    ax1.grid(True)
    ax1.set_title('Training Loss')

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0' and algs_lbl[i] != "DANE"):
            stringbatch = '$\infty$' + ", " + \
                '$\\alpha$' + " = " + str(alpha[i])
        elif(stringbatch == '0'):
             stringbatch = '$\infty$'

        ax2.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": " + '$B = $' +
                 stringbatch , marker=markers[i], markevery=0.2, markersize=7)

    ax2.set_ylim([0.88, 0.922])

    #plt.title('$\\kappa = $' + str(kappa))
    #fig.set_title('Linear Synthetic')
    ax2.grid(True)
    ax2.set_title('Testing Accuracy')
    #ax1.set_ylim([0.045, 0.2])
    ax.set_xlabel('Global rounds ' + '$T$')
    #ax.set_ylabel('Training Loss', labelpad = 10)
    #plt.xticks(np.arange(0.045, 2, 0.1))
    plt.savefig(dataset + 'acu_loss.pdf', bbox_inches='tight')
    plt.savefig(dataset +  'acu_loss.png', bbox_inches='tight')

def plot_summary_linear_R_and_alpha(num_users=[], loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], alpha = [], eta = [], algorithms_list=[], batch_size=0, kappa = [], dataset = ""):

    Numb_Algs = len(algorithms_list)
    glob_acc, train_acc, train_loss = get_training_data_value( num_users=num_users, loc_ep1=loc_ep1, Numb_Glob_Iters=Numb_Glob_Iters, lamb=lamb, learning_rate=learning_rate, alpha =alpha, eta =eta, algorithms_list=algorithms_list, batch_size=batch_size, kappa=kappa, dataset= dataset)
    for i in range(Numb_Algs):
        print(algorithms_list[i], "loss:", glob_acc[i].max())
    
    plt.figure(1)
    linestyles = ['-','-','-', '-']
    markers = ["o","v","s","*","x","P"]
    algs_lbl = ["DONE","DONE", "DONE", "DONE"]
    #plt.figure(figsize=(6,12))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    #fig, (ax1, ax2) = plt.subplots(1, 2)

    num_al = len(algs_lbl)

    for i in range(num_al):
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$R = $' + str(loc_ep1[i]) + ', $\\alpha = $' + str(alpha[i]) ,marker = markers[i],markevery=0.2, markersize=7)

    #fig.hlines(y=0.035,xmin=0, xmax=200, linestyle='--',label = "optimal solution", color= "m" )
    ax1.legend(loc='upper right')
    ax1.set_ylim([0.049, 0.1])
    ax1.grid(True)
    ax1.set_title('Fixed '+'$\\alpha$')

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        ax2.plot(train_loss[i+num_al, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$R = $' + str(loc_ep1[i+ num_al]) + ', $\\alpha = $' + str(alpha[i+ num_al]) ,marker = markers[i],markevery=0.2, markersize=7)

    ax2.set_ylim([0.049, 0.1])

    #plt.title('$\\kappa = $' + str(kappa))
    #fig.set_title('Linear Synthetic')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    ax2.set_title('Fixed R')
    #ax1.set_ylim([0.045, 0.2])
    ax.set_xlabel('Global rounds ' + '$T$')
    ax.set_ylabel('Training Loss', labelpad = 10)
    #plt.xticks(np.arange(0.045, 2, 0.1))
    plt.savefig('Linear_synthetic_R_alpha.pdf', bbox_inches='tight')
    plt.savefig('Linear_synthetic_R_alpha.png', bbox_inches='tight')
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    #fig, (ax1, ax2) = plt.subplots(1, 2)

    num_al = len(algs_lbl)

    for i in range(num_al):
        ax1.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": " + '$R = $' + str(
            loc_ep1[i]) + ', $\\alpha = $' + str(alpha[i]), marker=markers[i], markevery=0.2, markersize=7)

    #fig.hlines(y=0.035,xmin=0, xmax=200, linestyle='--',label = "optimal solution", color= "m" )
    ax1.legend(loc='upper right')
    ax1.set_ylim([0.049, 0.1])
    ax1.grid(True)
    ax1.set_title('Fixed '+'$\\alpha$')

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        ax2.plot(glob_acc[i+num_al, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": " + '$R = $' + str(
            loc_ep1[i + num_al]) + ', $\\alpha = $' + str(alpha[i + num_al]), marker=markers[i], markevery=0.2, markersize=7)

    ax2.set_ylim([0.049, 0.1])

    #plt.title('$\\kappa = $' + str(kappa))
    #fig.set_title('Linear Synthetic')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    ax2.set_title('Fixed R')
    #ax1.set_ylim([0.045, 0.2])
    ax.set_xlabel('Global rounds ' + '$T$')
    ax.set_ylabel('Testing Loss', labelpad=10)
    #plt.xticks(np.arange(0.045, 2, 0.1))
    plt.savefig('Linear_synthetic_R_alpha_test_loss.pdf', bbox_inches='tight')
    plt.savefig('Linear_synthetic_R_alpha_test_loss.png', bbox_inches='tight')

def plot_summary_linear2(num_users=[], loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], alpha = [], eta = [], algorithms_list=[], batch_size=0, kappa = [], dataset = ""):
    
    Numb_Algs = len(algorithms_list)
    glob_acc, train_acc, train_loss = get_training_data_value( num_users=num_users, loc_ep1=loc_ep1, Numb_Glob_Iters=Numb_Glob_Iters, lamb=lamb, learning_rate=learning_rate, alpha =alpha, eta =eta, algorithms_list=algorithms_list, batch_size=batch_size, kappa=kappa, dataset= dataset)
    for i in range(Numb_Algs):
        print(algorithms_list[i], "loss:", glob_acc[i].max())
    
    plt.figure(1)
    linestyles = ['-', '-', '-', '--', '-.', '-.']
    markers = ["o","v","s","*","x","P"]
    algs_lbl = ["DONE","DONE", "DONE", "Newton","DANE", "GD"]
    #plt.figure(figsize=(6,12))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    #fig, (ax1, ax2) = plt.subplots(1, 2)

    num_al = len(algs_lbl)

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": " +
                 '$B = $' + stringbatch, marker=markers[i], markevery=0.2 + i/10, markersize=7)

    #fig.hlines(y=0.035,xmin=0, xmax=200, linestyle='--',label = "optimal solution", color= "m" )
    ax1.legend(loc='upper right')
    ax1.set_ylim([0.049, 0.1])
    ax1.grid(True)
    ax1.set_title('Training loss')

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax2.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": " + '$B = $' + stringbatch, marker=markers[i], markevery=0.2 + i/10, markersize=7)

    ax2.set_ylim([0.049, 0.1])

    #plt.title('$\\kappa = $' + str(kappa))
    #fig.set_title('Linear Synthetic')
    ax2.grid(True)
    ax2.set_title('Testing loss')
    #ax1.set_ylim([0.045, 0.2])
    ax.set_xlabel('Global rounds ' + '$T$')
    #ax.set_ylabel('Training Loss', labelpad = 10)
    #plt.xticks(np.arange(0.045, 2, 0.1))
    plt.savefig('Linear_synthetic_different_loss.pdf', bbox_inches='tight')
    plt.savefig('Linear_synthetic_different_loss.png', bbox_inches='tight')
    

def plot_summary_linear_edge(num_users=[], loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], alpha = [], eta = [], algorithms_list=[], batch_size=0, kappa = [], dataset = ""):
    
    Numb_Algs = len(algorithms_list)
    glob_acc, train_acc, train_loss = get_training_data_value( num_users=num_users, loc_ep1=loc_ep1, Numb_Glob_Iters=Numb_Glob_Iters, lamb=lamb, learning_rate=learning_rate, alpha =alpha, eta =eta, algorithms_list=algorithms_list, batch_size=batch_size, kappa=kappa, dataset= dataset)
    for i in range(Numb_Algs):
        print(algorithms_list[i], "loss:", glob_acc[i].max())
    
    plt.figure(1)
    linestyles = ['-', '-', '-', '-', '-.', '-.']
    markers = ["o","v","s","*","x","P"]
    algs_lbl = ["DONE","DONE", "DONE", "DONE"]
    #plt.figure(figsize=(6,12))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    #fig, (ax1, ax2) = plt.subplots(1, 2)

    num_al = len(algs_lbl)

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": " +'$S = $' + str(num_users[i]), marker=markers[i], markevery=0.2 + i/10, markersize=7)

    #fig.hlines(y=0.035,xmin=0, xmax=200, linestyle='--',label = "optimal solution", color= "m" )
    ax1.legend(loc='upper right')
    ax1.set_ylim([0.049, 0.1])
    ax1.grid(True)
    ax1.set_title('Training loss')

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax2.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": " + '$S = $' + str(num_users[i]), marker=markers[i], markevery=0.2 + i/10, markersize=7)

    ax2.set_ylim([0.049, 0.1])

    #plt.title('$\\kappa = $' + str(kappa))
    #fig.set_title('Linear Synthetic')
    ax2.grid(True)
    ax2.set_title('Testing loss')
    #ax1.set_ylim([0.045, 0.2])
    ax.set_xlabel('Global rounds ' + '$T$')
    #ax.set_ylabel('Training Loss', labelpad = 10)
    #plt.xticks(np.arange(0.045, 2, 0.1))
    plt.savefig('Linear_synthetic_edge.pdf', bbox_inches='tight')
    plt.savefig('Linear_synthetic_edge.png', bbox_inches='tight')

def plot_summary_mnist_R_and_alpha(num_users=[], loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], alpha = [], eta = [], algorithms_list=[], batch_size=0, kappa = [], dataset = ""):

    Numb_Algs = len(algorithms_list)
    glob_acc, train_acc, train_loss = get_training_data_value( num_users=num_users, loc_ep1=loc_ep1, Numb_Glob_Iters=Numb_Glob_Iters, lamb=lamb, learning_rate=learning_rate, alpha =alpha, eta =eta, algorithms_list=algorithms_list, batch_size=batch_size, kappa=kappa, dataset= dataset)
    for i in range(Numb_Algs):
        print(algorithms_list[i], "loss:", glob_acc[i].max())
    
    plt.figure(1)
    linestyles = ['-','-','-', '-']
    markers = ["o","v","s","*","x","P"]
    algs_lbl = ["DONE","DONE", "DONE", "DONE"]
    #plt.figure(figsize=(6,12))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    #fig, (ax1, ax2) = plt.subplots(1, 2)

    num_al = len(algs_lbl)

    for i in range(num_al):
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$R = $' + str(loc_ep1[i]) + ', $\\alpha = $' + str(alpha[i]) ,marker = markers[i],markevery=0.2, markersize=7)

    #fig.hlines(y=0.035,xmin=0, xmax=200, linestyle='--',label = "optimal solution", color= "m" )
    ax1.legend(loc='upper right')
    #ax1.set_ylim([0.2, 0.5])
    ax1.set_ylim([0.21, 0.52])
    ax1.grid(True)
    ax1.set_title('Fixed '+'$\\alpha$')

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        ax2.plot(train_loss[i+num_al, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$R = $' + str(loc_ep1[i+ num_al]) + ', $\\alpha = $' + str(alpha[i+ num_al]) ,marker = markers[i],markevery=0.2, markersize=7)

    ax2.set_ylim([0.21, 0.52])

    #plt.title('$\\kappa = $' + str(kappa))
    #fig.set_title('Linear Synthetic')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    ax2.set_title('Fixed R')
    #ax1.set_ylim([0.045, 0.2])
    ax.set_xlabel('Global rounds ' + '$T$')
    ax.set_ylabel('Training Loss', labelpad = 10)
    #plt.xticks(np.arange(0.045, 2, 0.1))
    plt.savefig('MNIST_R_alpha.pdf', bbox_inches='tight')
    plt.savefig('MNIST_R_alpha.png', bbox_inches='tight')
    
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    #fig, (ax1, ax2) = plt.subplots(1, 2)

    num_al = len(algs_lbl)

    for i in range(num_al):
        ax1.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": " + '$R = $' + str(
            loc_ep1[i]) + ', $\\alpha = $' + str(alpha[i]), marker=markers[i], markevery=0.2, markersize=7)

    #fig.hlines(y=0.035,xmin=0, xmax=200, linestyle='--',label = "optimal solution", color= "m" )
    ax1.legend(loc='lower right')
    ax1.set_ylim([0.86, 0.922])
    ax1.grid(True)
    ax1.set_title('Fixed '+'$\\alpha$')

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        ax2.plot(glob_acc[i+num_al, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": " + '$R = $' + str(
            loc_ep1[i + num_al]) + ', $\\alpha = $' + str(alpha[i + num_al]), marker=markers[i], markevery=0.2, markersize=7)

    ax2.set_ylim([0.86, 0.922])

    #plt.title('$\\kappa = $' + str(kappa))
    #fig.set_title('Linear Synthetic')
    ax2.legend(loc='lower right')
    ax2.grid(True)
    ax2.set_title('Fixed R')
    #ax1.set_ylim([0.045, 0.2])
    ax.set_xlabel('Global rounds ' + '$T$')
    ax.set_ylabel('Testing Accuracy', labelpad=10)
    #plt.xticks(np.arange(0.045, 2, 0.1))
    plt.savefig('MNIST_R_alpha_accu.pdf', bbox_inches='tight')
    plt.savefig('MNIST_R_alpha_accu.png', bbox_inches='tight')

def plot_summary_human_R_and_alpha(num_users=[], loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], alpha = [], eta = [], algorithms_list=[], batch_size=0, kappa = [], dataset = ""):

    Numb_Algs = len(algorithms_list)
    glob_acc, train_acc, train_loss = get_training_data_value( num_users=num_users, loc_ep1=loc_ep1, Numb_Glob_Iters=Numb_Glob_Iters, lamb=lamb, learning_rate=learning_rate, alpha =alpha, eta =eta, algorithms_list=algorithms_list, batch_size=batch_size, kappa=kappa, dataset= dataset)
    for i in range(Numb_Algs):
        print(algorithms_list[i], "loss:", glob_acc[i].max())
    
    plt.figure(1)
    linestyles = ['-','-','-', '-']
    markers = ["o","v","s","*","x","P"]
    algs_lbl = ["DONE","DONE", "DONE", "DONE"]
    #plt.figure(figsize=(6,12))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    #fig, (ax1, ax2) = plt.subplots(1, 2)

    num_al = len(algs_lbl)

    for i in range(num_al):
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$R = $' + str(loc_ep1[i]) + ', $\\alpha = $' + str(alpha[i]) ,marker = markers[i],markevery=0.2, markersize=7)

    #fig.hlines(y=0.035,xmin=0, xmax=200, linestyle='--',label = "optimal solution", color= "m" )
    ax1.legend(loc='upper right')
    ax1.set_ylim([0.1,0.6])
    ax1.grid(True)
    ax1.set_title('Fixed '+'$\\alpha$')

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        ax2.plot(train_loss[i+num_al, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$R = $' + str(loc_ep1[i+ num_al]) + ', $\\alpha = $' + str(alpha[i+ num_al]) ,marker = markers[i],markevery=0.2, markersize=7)

    ax2.set_ylim([0.1,0.6])

    #plt.title('$\\kappa = $' + str(kappa))
    #fig.set_title('Linear Synthetic')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    ax2.set_title('Fixed R')
    #ax1.set_ylim([0.045, 0.2])
    ax.set_xlabel('Global rounds ' + '$T$')
    ax.set_ylabel('Training Loss', labelpad = 10)
    #plt.xticks(np.arange(0.045, 2, 0.1))
    plt.savefig('Human_R_alpha.pdf', bbox_inches='tight')
    plt.savefig('Human_R_alpha.png', bbox_inches='tight')
    
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    #fig, (ax1, ax2) = plt.subplots(1, 2)

    num_al = len(algs_lbl)

    for i in range(num_al):
        ax1.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": " + '$R = $' + str(
            loc_ep1[i]) + ', $\\alpha = $' + str(alpha[i]), marker=markers[i], markevery=0.2, markersize=7)

    #fig.hlines(y=0.035,xmin=0, xmax=200, linestyle='--',label = "optimal solution", color= "m" )
    ax1.legend(loc='lower right')
    ax1.set_ylim([0.86, 0.97])
    ax1.grid(True)
    ax1.set_title('Fixed '+'$\\alpha$')

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        ax2.plot(glob_acc[i+num_al, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": " + '$R = $' + str(
            loc_ep1[i + num_al]) + ', $\\alpha = $' + str(alpha[i + num_al]), marker=markers[i], markevery=0.2, markersize=7)

    ax2.set_ylim([0.86, 0.97])

    #plt.title('$\\kappa = $' + str(kappa))
    #fig.set_title('Linear Synthetic')
    ax2.legend(loc='lower right')
    ax2.grid(True)
    ax2.set_title('Fixed R')
    #ax1.set_ylim([0.045, 0.2])
    ax.set_xlabel('Global rounds ' + '$T$')
    ax.set_ylabel('Testing Accuracy', labelpad=10)
    #plt.xticks(np.arange(0.045, 2, 0.1))
    plt.savefig('Human_R_alpha_accu.pdf', bbox_inches='tight')
    plt.savefig('Human_R_alpha_accu.png', bbox_inches='tight')


def plot_summary_nist_R_and_alpha(num_users=[], loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], alpha = [], eta = [], algorithms_list=[], batch_size=0, kappa = [], dataset = ""):

    Numb_Algs = len(algorithms_list)
    glob_acc, train_acc, train_loss = get_training_data_value( num_users=num_users, loc_ep1=loc_ep1, Numb_Glob_Iters=Numb_Glob_Iters, lamb=lamb, learning_rate=learning_rate, alpha =alpha, eta =eta, algorithms_list=algorithms_list, batch_size=batch_size, kappa=kappa, dataset= dataset)
    for i in range(Numb_Algs):
        print(algorithms_list[i], "loss:", glob_acc[i].max())
    
    plt.figure(1)
    linestyles = ['-','-','-', '-']
    markers = ["o","v","s","*","x","P"]
    algs_lbl = ["DONE","DONE", "DONE", "DONE"]
    #plt.figure(figsize=(6,12))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    #fig, (ax1, ax2) = plt.subplots(1, 2)

    num_al = len(algs_lbl)

    for i in range(num_al):
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$R = $' + str(loc_ep1[i]) + ', $\\alpha = $' + str(alpha[i]) ,marker = markers[i],markevery=0.2, markersize=7)

    #fig.hlines(y=0.035,xmin=0, xmax=200, linestyle='--',label = "optimal solution", color= "m" )
    ax1.legend(loc='upper right')
    ax1.set_ylim([0.62,2])
    ax1.grid(True)
    ax1.set_title('Fixed '+'$\\alpha$')

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        ax2.plot(train_loss[i+num_al, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$R = $' + str(loc_ep1[i+ num_al]) + ', $\\alpha = $' + str(alpha[i+ num_al]) ,marker = markers[i],markevery=0.2, markersize=7)

    #ax2.set_ylim([0.65,1.6])
    ax2.set_ylim([0.62,2])

    #plt.title('$\\kappa = $' + str(kappa))
    #fig.set_title('Linear Synthetic')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    ax2.set_title('Fixed R')
    #ax1.set_ylim([0.045, 0.2])
    ax.set_xlabel('Global rounds ' + '$T$')
    ax.set_ylabel('Training Loss', labelpad = 10)
    #plt.xticks(np.arange(0.045, 2, 0.1))
    plt.savefig('Nist_R_alpha.pdf', bbox_inches='tight')
    plt.savefig('Nist_R_alpha.png', bbox_inches='tight')
    
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    #fig, (ax1, ax2) = plt.subplots(1, 2)

    num_al = len(algs_lbl)

    for i in range(num_al):
        ax1.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": " + '$R = $' + str(
            loc_ep1[i]) + ', $\\alpha = $' + str(alpha[i]), marker=markers[i], markevery=0.2, markersize=7)

    #fig.hlines(y=0.035,xmin=0, xmax=200, linestyle='--',label = "optimal solution", color= "m" )
    ax1.legend(loc='lower right')
    ax1.set_ylim([0.5, 0.81])
    ax1.grid(True)
    ax1.set_title('Fixed '+'$\\alpha$')

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        ax2.plot(glob_acc[i+num_al, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": " + '$R = $' + str(
            loc_ep1[i + num_al]) + ', $\\alpha = $' + str(alpha[i + num_al]), marker=markers[i], markevery=0.2, markersize=7)

    #ax2.set_ylim([0.63, 0.808])
    ax2.set_ylim([0.5, 0.81])

    #plt.title('$\\kappa = $' + str(kappa))
    #fig.set_title('Linear Synthetic')
    ax2.legend(loc='lower right')
    ax2.grid(True)
    ax2.set_title('Fixed R')
    #ax1.set_ylim([0.045, 0.2])
    ax.set_xlabel('Global rounds ' + '$T$')
    ax.set_ylabel('Testing Accuracy', labelpad=10)
    #plt.xticks(np.arange(0.045, 2, 0.1))
    plt.savefig('Nist_R_alpha_accu.pdf', bbox_inches='tight')
    plt.savefig('Nist_R_alpha_accu.png', bbox_inches='tight')

def plot_summary_mnist_edge(num_users=[], loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], alpha = [], eta = [], algorithms_list=[], batch_size=0, kappa = [], dataset = ""):

    Numb_Algs = len(algorithms_list)
    dataset = dataset

    algs_lbl = algorithms_list.copy()
    glob_acc, train_acc, train_loss = get_training_data_value( num_users=num_users, loc_ep1=loc_ep1, Numb_Glob_Iters=Numb_Glob_Iters, lamb=lamb, learning_rate=learning_rate, alpha =alpha, eta =eta, algorithms_list=algorithms_list, batch_size=batch_size, dataset= dataset)
    
    for i in range(Numb_Algs):
        print(algorithms_list[i], "acc:", glob_acc[i].max())
        print(algorithms_list[i], "loss:", train_loss[i].min())

    linestyles = ['-', '-', '-', '-']
    markers = ["o","v","s","*","x","P"]
    algs_lbl = ["DONE", "DONE", "DONE", "DONE"]
    num_users = ["0.4N","0.6N","0.8N","N"]
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')

    num_al = 4

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$S = $' + str(num_users[i]) ,marker = markers[i],markevery=0.2, markersize=7)
    ax1.legend(loc='upper right')
    ax1.set_ylim([0.21, 0.52])
    ax1.grid(True)
    ax1.set_title('Training Loss')

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax2.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$S = $' + str(num_users[i]) ,marker = markers[i],markevery=0.2, markersize=7)

    ax2.set_ylim([0.86, 0.922])
    ax2.grid(True)
    ax2.set_title('Testing Accuracy')
    ax.set_xlabel('Global rounds ' + '$T$')
    plt.savefig(dataset + '_edge.pdf', bbox_inches='tight')
    plt.savefig(dataset +  '_edge.png', bbox_inches='tight')

def plot_summary_human_edge(num_users=[], loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], alpha = [], eta = [], algorithms_list=[], batch_size=0, kappa = [], dataset = ""):

    Numb_Algs = len(algorithms_list)
    dataset = dataset

    algs_lbl = algorithms_list.copy()
    glob_acc, train_acc, train_loss = get_training_data_value( num_users=num_users, loc_ep1=loc_ep1, Numb_Glob_Iters=Numb_Glob_Iters, lamb=lamb, learning_rate=learning_rate, alpha =alpha, eta =eta, algorithms_list=algorithms_list, batch_size=batch_size, dataset= dataset)
    
    for i in range(Numb_Algs):
        print(algorithms_list[i], "acc:", glob_acc[i].max())
        print(algorithms_list[i], "loss:", train_loss[i].min())

    linestyles = ['-', '-', '-', '-']
    markers = ["o","v","s","*","x","P"]
    algs_lbl = ["DONE", "DONE", "DONE", "DONE"]
    num_users = ["0.4N","0.6N","0.8N","N"]
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    
    num_al = 4

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$S = $' + str(num_users[i]) ,marker = markers[i],markevery=0.2, markersize=7)
    ax1.legend(loc='upper right')
    ax1.set_ylim([0.1, 0.6])
    ax1.grid(True)
    ax1.set_title('Training Loss')

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax2.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$S = $' + str(num_users[i]) ,marker = markers[i],markevery=0.2, markersize=7)

    ax2.set_ylim([0.86, 0.97])
    ax2.grid(True)
    ax2.set_title('Testing Accuracy')
    ax.set_xlabel('Global rounds ' + '$T$')
    plt.savefig(dataset + '_edge.pdf', bbox_inches='tight')
    plt.savefig(dataset +  '_edge.png', bbox_inches='tight')

def plot_summary_nist_edge(num_users=[], loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], alpha = [], eta = [], algorithms_list=[], batch_size=0, kappa = [], dataset = ""):

    Numb_Algs = len(algorithms_list)
    dataset = dataset

    algs_lbl = algorithms_list.copy()
    glob_acc, train_acc, train_loss = get_training_data_value( num_users=num_users, loc_ep1=loc_ep1, Numb_Glob_Iters=Numb_Glob_Iters, lamb=lamb, learning_rate=learning_rate, alpha =alpha, eta =eta, algorithms_list=algorithms_list, batch_size=batch_size, dataset= dataset)
    
    for i in range(Numb_Algs):
        print(algorithms_list[i], "acc:", glob_acc[i].max())
        print(algorithms_list[i], "loss:", train_loss[i].min())

    linestyles = ['-', '-', '-', '-']
    markers = ["o","v","s","*","x","P"]
    algs_lbl = ["DONE", "DONE", "DONE", "DONE"]
    num_users = ["0.4N","0.6N","0.8N","N"]
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')

    num_al = 4

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$S = $' + str(num_users[i]) ,marker = markers[i],markevery=0.2, markersize=7)
    ax1.legend(loc='upper right')
    ax1.set_ylim([0.62, 2])
    ax1.grid(True)
    ax1.set_title('Training Loss')

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax2.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$S = $' + str(num_users[i]) ,marker = markers[i],markevery=0.2, markersize=7)

    ax2.set_ylim([0.5, 0.81])
    ax2.grid(True)
    ax2.set_title('Testing Accuracy')
    ax.set_xlabel('Global rounds ' + '$T$')
    plt.savefig(dataset + '_edge.pdf', bbox_inches='tight')
    plt.savefig(dataset +  '_edge.png', bbox_inches='tight')

def plot_summary_linear_kappa(num_users=[], loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], alpha = [], eta = [], algorithms_list=[], batch_size=0, kappa = [], dataset = ""):

    Numb_Algs = len(algorithms_list)
    glob_acc, train_acc, train_loss = get_training_data_value( num_users=num_users, loc_ep1=loc_ep1, Numb_Glob_Iters=Numb_Glob_Iters, lamb=lamb, learning_rate=learning_rate, alpha =alpha, eta =eta, algorithms_list=algorithms_list, batch_size=batch_size, kappa = kappa, dataset= dataset)
    for i in range(Numb_Algs):
        print(algorithms_list[i], "loss:", glob_acc[i].max())
    kappa = ["$10$", "$10^2$", "$10^3$", "$10^4$"]
    plt.figure(1)
    linestyles = ['-','-','-', '-']
    markers = ["o","v","s","*","x","P"]
    algs_lbl = ["DONE","DONE", "DONE", "DONE"]
    #plt.figure(figsize=(6,12))
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    #fig, (ax1, ax2) = plt.subplots(1, 2)

    num_al = len(algs_lbl)

    for i in range(num_al):
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$R = $' + str(loc_ep1[i]) + ', $\\kappa = $' + kappa[i] ,marker = markers[i],markevery=0.2, markersize=7)

    #fig.hlines(y=0.035,xmin=0, xmax=200, linestyle='--',label = "optimal solution", color= "m" )
    ax1.legend(loc='upper right')
    ax1.set_ylim([0.0485, 0.1])
    ax1.grid(True)
   # ax1.set_title('Fixed '+'$\\alpha$')

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        ax2.plot(train_loss[i+num_al, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$R = $' + str(loc_ep1[i+ num_al]) + ', $\\kappa = $' + kappa[i] ,marker = markers[i], markevery=0.2, markersize=7)

    ax2.set_ylim([0.0485, 0.1])
    ax2.legend(loc='upper right')
    ax2.grid(True)

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        ax3.plot(train_loss[i+num_al*2, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$R = $' + str(loc_ep1[i+ num_al * 2]) + ', $\\kappa = $' + kappa[i] ,marker = markers[i], markevery=0.2, markersize=7)
    ax3.set_ylim([0.0485, 0.1])
    ax3.legend(loc='upper right')
    ax3.grid(True)

    ax.set_xlabel('Global rounds ' + '$T$')
    ax.set_ylabel('Training Loss', labelpad = 10)
    plt.savefig('Linear_synthetic_train_kappa.pdf', bbox_inches='tight')
    plt.savefig('Linear_synthetic_train_kappa.png', bbox_inches='tight')
    Numb_Algs = len(algorithms_list)


    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    #fig, (ax1, ax2) = plt.subplots(1, 2)
    
    num_al = len(algs_lbl)

    for i in range(num_al):
        ax1.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": " + '$R = $' + str(
            loc_ep1[i]) + ', $\\kappa = $' + kappa[i], marker=markers[i], markevery=0.2, markersize=7)
    ax1.legend(loc='upper right')
    ax1.set_ylim([0.0485, 0.1])
    ax1.grid(True)

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        ax2.plot(glob_acc[i+num_al, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": " + '$R = $' + str(
            loc_ep1[i + num_al]) + ', $\\kappa = $' + kappa[i], marker=markers[i], markevery=0.2, markersize=7)
    ax2.set_ylim([0.0485, 0.1])
    ax2.legend(loc='upper right')
    ax2.grid(True)

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        ax3.plot(glob_acc[i+num_al*2, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": " + '$R = $' + str(
            loc_ep1[i + num_al*2]) + ', $\\kappa = $' + kappa[i], marker=markers[i], markevery=0.2, markersize=7)
    ax3.set_ylim([0.0485, 0.1])
    ax3.legend(loc='upper right')
    ax3.grid(True)

    ax.set_xlabel('Global rounds ' + '$T$')
    ax.set_ylabel('Testing Loss', labelpad=10)
    plt.savefig('Linear_synthetic_kappa_test_loss.pdf', bbox_inches='tight')
    plt.savefig('Linear_synthetic_kappa_test_loss.png', bbox_inches='tight')


def plot_summary_nist_batch(num_users=[], loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], alpha = [], eta = [], algorithms_list=[], batch_size=0, kappa = [], dataset = ""):

    Numb_Algs = len(algorithms_list)
    dataset = dataset
    algs_lbl = algorithms_list.copy()
    glob_acc, train_acc, train_loss = get_training_data_value( num_users=num_users, loc_ep1=loc_ep1, Numb_Glob_Iters=Numb_Glob_Iters, lamb=lamb, learning_rate=learning_rate, alpha =alpha, eta =eta, algorithms_list=algorithms_list, batch_size=batch_size, dataset= dataset)
    for i in range(Numb_Algs):
        print(algorithms_list[i], "acc:", glob_acc[i].max())
        print(algorithms_list[i], "loss:", train_loss[i].min())

    linestyles = ['-', '-', '-', '-']
    markers = ["o","v","s","*","x","P"]
    algs_lbl = ["DONE", "DONE", "DONE", "DONE"]

   #plt.figure(figsize=(6,12))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    num_al = 4
    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+ '$B = $' + str(stringbatch) + ', $R = $' + str(loc_ep1[i]) + ', $\\alpha = $' + str(alpha[i]),marker = markers[i],markevery=0.2, markersize=7)
    ax1.legend(loc='upper right')
    ax1.set_ylim([0.62,2])
    ax1.grid(True)
    ax1.set_title('Training Loss')
    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax2.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$B = $' + str(stringbatch) + ', $R = $' + str(loc_ep1[i]) + ', $\\alpha = $' + str(alpha[i]),marker = markers[i],markevery=0.2, markersize=7)

    ax2.set_ylim([0.5, 0.81])
    ax2.grid(True)
    ax2.set_title('Testing Accuracy')
    ax.set_xlabel('Global rounds ' + '$T$')
    plt.savefig('Nist_batch.pdf', bbox_inches='tight')

def plot_summary_mnist_batch(num_users=[], loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], alpha = [], eta = [], algorithms_list=[], batch_size=0, kappa = [], dataset = ""):

    Numb_Algs = len(algorithms_list)
    dataset = dataset
    algs_lbl = algorithms_list.copy()
    glob_acc, train_acc, train_loss = get_training_data_value( num_users=num_users, loc_ep1=loc_ep1, Numb_Glob_Iters=Numb_Glob_Iters, lamb=lamb, learning_rate=learning_rate, alpha =alpha, eta =eta, algorithms_list=algorithms_list, batch_size=batch_size, dataset= dataset)
    for i in range(Numb_Algs):
        print(algorithms_list[i], "acc:", glob_acc[i].max())
        print(algorithms_list[i], "loss:", train_loss[i].min())

    linestyles = ['-', '-', '-', '-']
    markers = ["o","v","s","*","x","P"]
    algs_lbl = ["DONE", "DONE", "DONE", "DONE"]

   #plt.figure(figsize=(6,12))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    num_al = 4
    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+ '$B = $' + str(stringbatch) + ', $R = $' + str(loc_ep1[i]) + ', $\\alpha = $' + str(alpha[i]),marker = markers[i],markevery=0.2, markersize=7)
    ax1.legend(loc='upper right')
    ax1.set_ylim([0.21, 0.52])
    ax1.grid(True)
    ax1.set_title('Training Loss')
    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax2.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$B = $' + str(stringbatch) + ', $R = $' + str(loc_ep1[i]) + ', $\\alpha = $' + str(alpha[i]),marker = markers[i],markevery=0.2, markersize=7)

    ax2.set_ylim([0.86, 0.922])
    ax2.grid(True)
    ax2.set_title('Testing Accuracy')
    ax.set_xlabel('Global rounds ' + '$T$')
    plt.savefig('Mnist_batch.pdf', bbox_inches='tight')

def plot_summary_human_batch(num_users=[], loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], alpha = [], eta = [], algorithms_list=[], batch_size=0, kappa = [], dataset = ""):

    Numb_Algs = len(algorithms_list)
    dataset = dataset
    algs_lbl = algorithms_list.copy()
    glob_acc, train_acc, train_loss = get_training_data_value( num_users=num_users, loc_ep1=loc_ep1, Numb_Glob_Iters=Numb_Glob_Iters, lamb=lamb, learning_rate=learning_rate, alpha =alpha, eta =eta, algorithms_list=algorithms_list, batch_size=batch_size, dataset= dataset)
    for i in range(Numb_Algs):
        print(algorithms_list[i], "acc:", glob_acc[i].max())
        print(algorithms_list[i], "loss:", train_loss[i].min())

    linestyles = ['-', '-', '-', '-']
    markers = ["o","v","s","*","x","P"]
    algs_lbl = ["DONE", "DONE", "DONE", "DONE"]

   #plt.figure(figsize=(6,12))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    num_al = 4
    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+ '$B = $' + str(stringbatch) + ', $R = $' + str(loc_ep1[i]) + ', $\\alpha = $' + str(alpha[i]),marker = markers[i],markevery=0.2, markersize=7)
    ax1.legend(loc='upper right')
    ax1.set_ylim([0.1,0.6])
    ax1.grid(True)
    ax1.set_title('Training Loss')
    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax2.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + ": "+  '$B = $' + str(stringbatch) + ', $R = $' + str(loc_ep1[i]) + ', $\\alpha = $' + str(alpha[i]),marker = markers[i],markevery=0.2, markersize=7)

    ax2.set_ylim([0.86, 0.97])
    ax2.grid(True)
    ax2.set_title('Testing Accuracy')
    ax.set_xlabel('Global rounds ' + '$T$')
    plt.savefig('Human_batch.pdf', bbox_inches='tight')


def plot_summary_mnist_algorithm(num_users=[], loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], alpha = [], eta = [], algorithms_list=[], batch_size=0, kappa = [], dataset = ""):

    Numb_Algs = len(algorithms_list)
    glob_acc, train_acc, train_loss = get_training_data_value( num_users=num_users, loc_ep1=loc_ep1, Numb_Glob_Iters=Numb_Glob_Iters, lamb=lamb, learning_rate=learning_rate, alpha =alpha, eta =eta, algorithms_list=algorithms_list, batch_size=batch_size, kappa=kappa, dataset= dataset)
    for i in range(Numb_Algs):
        print(algorithms_list[i], "loss:", glob_acc[i].max())
    
    plt.figure(1)
    linestyles = ['-','-','-', '-','-', '-']
    markers = ["o","v","s","*","x","P"]
    algs_lbl = ["DONE", "NEWTON", "FEDL", "DANE", "GT", "GD"]
    #plt.figure(figsize=(6,12))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    #fig, (ax1, ax2) = plt.subplots(1, 2)

    num_al = len(algs_lbl)

    for i in range(num_al):
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] ,marker = markers[i],markevery=0.2, markersize=7)

    #fig.hlines(y=0.035,xmin=0, xmax=200, linestyle='--',label = "optimal solution", color= "m" )
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Training Loss', labelpad=10)
    ax1.set_ylim([0.21, 0.52])
    ax1.grid(True)
    #ax1.set_title('Fixed '+'$\\alpha$')

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        ax2.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=algs_lbl[i], marker=markers[i], markevery=0.2, markersize=7)

    ax2.set_ylim([0.86, 0.922])

    #plt.title('$\\kappa = $' + str(kappa))
    #fig.set_title('Linear Synthetic')
    ax2.legend(loc='lower right')
    ax2.grid(True)
    #ax1.set_ylim([0.045, 0.2])
    ax.set_xlabel('Global rounds ' + '$T$')
    ax2.set_ylabel('Testing Accuracy', labelpad=10)
    #plt.xticks(np.arange(0.045, 2, 0.1))
    plt.savefig('MNIST_Algorithm.pdf', bbox_inches='tight')

def plot_summary_nist_algorithm(num_users=[], loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], alpha = [], eta = [], algorithms_list=[], batch_size=0, kappa = [], dataset = ""):

    Numb_Algs = len(algorithms_list)
    glob_acc, train_acc, train_loss = get_training_data_value( num_users=num_users, loc_ep1=loc_ep1, Numb_Glob_Iters=Numb_Glob_Iters, lamb=lamb, learning_rate=learning_rate, alpha =alpha, eta =eta, algorithms_list=algorithms_list, batch_size=batch_size, kappa=kappa, dataset= dataset)
    for i in range(Numb_Algs):
        print(algorithms_list[i], "loss:", glob_acc[i].max())
    
    plt.figure(1)
    linestyles = ['-','-','-', '-','-', '-']
    markers = ["o","v","s","*","x","P"]
    algs_lbl = ["DONE", "NEWTON", "FEDL", "DANE", "GT", "GD"]
    #plt.figure(figsize=(6,12))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    #fig, (ax1, ax2) = plt.subplots(1, 2)

    num_al = len(algs_lbl)

    for i in range(num_al):
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] ,marker = markers[i],markevery=0.2, markersize=7)

    #fig.hlines(y=0.035,xmin=0, xmax=200, linestyle='--',label = "optimal solution", color= "m" )
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Training Loss', labelpad=10)
    ax1.set_ylim([0.62, 2])
    ax1.grid(True)
    #ax1.set_title('Fixed '+'$\\alpha$')

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        ax2.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=algs_lbl[i], marker=markers[i], markevery=0.2, markersize=7)

    ax2.set_ylim([0.4, 0.82])

    #plt.title('$\\kappa = $' + str(kappa))
    #fig.set_title('Linear Synthetic')
    ax2.legend(loc='lower right')
    ax2.grid(True)
    #ax1.set_ylim([0.045, 0.2])
    ax.set_xlabel('Global rounds ' + '$T$')
    ax2.set_ylabel('Testing Accuracy', labelpad=10)
    #plt.xticks(np.arange(0.045, 2, 0.1))
    plt.savefig('NIST_Algorithm.pdf', bbox_inches='tight')

def plot_summary_human_algorithm(num_users=[], loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], alpha = [], eta = [], algorithms_list=[], batch_size=0, kappa = [], dataset = ""):

    Numb_Algs = len(algorithms_list)
    glob_acc, train_acc, train_loss = get_training_data_value( num_users=num_users, loc_ep1=loc_ep1, Numb_Glob_Iters=Numb_Glob_Iters, lamb=lamb, learning_rate=learning_rate, alpha =alpha, eta =eta, algorithms_list=algorithms_list, batch_size=batch_size, kappa=kappa, dataset= dataset)
    for i in range(Numb_Algs):
        print(algorithms_list[i], "loss:", glob_acc[i].max())
    
    plt.figure(1)
    linestyles = ['-','-','-', '-','-', '-']
    markers = ["o","v","s","*","x","P"]
    algs_lbl = ["DONE", "NEWTON", "FEDL", "DANE", "GT", "GD"]
    #plt.figure(figsize=(6,12))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    #fig, (ax1, ax2) = plt.subplots(1, 2)

    num_al = len(algs_lbl)

    for i in range(num_al):
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] ,marker = markers[i],markevery=0.2, markersize=7)

    #fig.hlines(y=0.035,xmin=0, xmax=200, linestyle='--',label = "optimal solution", color= "m" )
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Training Loss', labelpad=10)
    ax1.set_ylim([0.1, 0.6])
    ax1.grid(True)
    #ax1.set_title('Fixed '+'$\\alpha$')

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        ax2.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=algs_lbl[i], marker=markers[i], markevery=0.2, markersize=7)

    ax2.set_ylim([0.86, 0.97])

    #plt.title('$\\kappa = $' + str(kappa))
    #fig.set_title('Linear Synthetic')
    ax2.legend(loc='lower right')
    ax2.grid(True)
    #ax1.set_ylim([0.045, 0.2])
    ax.set_xlabel('Global rounds ' + '$T$')
    ax2.set_ylabel('Testing Accuracy', labelpad=10)
    #plt.xticks(np.arange(0.045, 2, 0.1))
    plt.savefig('Human_Algorithm.pdf', bbox_inches='tight')
