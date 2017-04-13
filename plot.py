import numpy as np
from collections import defaultdict
import matplotlib.pylab as plt
import matplotlib.cm as cm
import sys
import os

#plt.rcParams.update({'font.size': 16})
#plt.close('all')

#Direct input 
params = {'text.usetex' : True,
          'font.size' : 15,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params) 
plt.rcParams['xtick.labelsize'] = 17
plt.rcParams['ytick.labelsize'] = 17
plt.rcParams['axes.labelsize'] = 17
plt.rcParams['axes.titlesize'] = 17



def parse_log(file_):
    data = defaultdict(list)
    with open(file_) as f:
        for line in f:
            line = line.strip()
            line = line.split()
            if len(line) == 2:
                try:
                    data[line[0][:-1]].append(float(line[1]))
                except ValueError:
                    continue
    return data


def plot(logs, what='per', title='DRAW', xlim=None, ylim=None):
    fig = plt.figure()
    plt.title(title, fontsize=17)
    colors = ['b', 'r', 'g', 'k', 'y', 'c', 'm', 'sienna', 'darkslategray',  'slateblue', 'darkmagenta', 'crimson', 'darkgoldenrod', 'lime']
    for i, log in enumerate(logs):
        if isinstance(log, tuple):
            name = log[1]
            log = log[0]
        else:
            name = os.path.basename(log)
        data = parse_log(log)
        test = data['test_nll_bound'][1:] # TODO  # :-1]
        print log, np.min(test), np.argmin(test)
        c = colors[i%len(colors)]
        plt.plot(test[:], c=c, lw=2, label=name)
        plt.ylabel('NLL', fontsize=17)
        plt.legend()
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.grid()
    plt.xlabel('Epochs', fontsize=17)
    #plt.close('all')
    plt.savefig('draw.pdf', dpi=1000, bbox_inches='tight')


logs = [#('/Tmp/laurent/draw/baseline_bs=128-20160912-114307/log', 'Baseline'),
        #('/Tmp/laurent/draw/new_baseline_bs=128-20160912-183603/log', 'Baseline New'),
        #('/Tmp/laurent/draw/norm_bs=128-20160912-184659/log', 'Weight Norm'),
        #('/Tmp/laurent/draw/norm_bs=128,rec_gamma=0.1-20160912-185946/log', 'Norm Prop 0.1'),
        #('/Tmp/laurent/draw/norm_bs=128,rec_gamma=1.0-20160912-191720/log', 'Norm Prop 1.0'),
        ('/Tmp/laurent/draw/baseline_bs=128,lr=0.001-20160913-111227/log', 'Baseline New 0.001'),
        ('/Tmp/laurent/draw/norm_bs=128,lr=0.001-20160913-113304/log', 'Weight Norm 0.001'),
        ('/Tmp/laurent/draw/norm_bs=128,lr=0.001,rec_gamma=0.1-20160913-113359/log', 'Norm Prop 0.1 0.001'),
        ('/Tmp/laurent/draw/norm_bs=128,lr=0.001,rec_gamma=1.0-20160913-113458/log', 'Norm Prop 1.0 0.001'),
        ('/Tmp/laurent/draw/norm_bs=128,lr=0.001,rec_gamma=0.1,gamma=1.0-20160913-151218/log', 'Norm Prop 0.1 1.0 0.001'),
        ('/Tmp/laurent/draw/norm_bs=128,lr=0.001,rec_gamma=1.0,gamma=1.0-20160913-151724/log', 'Norm Prop 1 1.0 0.001'),
        ('/Tmp/laurent/draw/norm_bs=128,rec_gamma=0.1,gamma=1.0-20160913-154425/log', 'Norm Prop 0.1 1.0'), #Might actually be gamma=auto
        ('/Tmp/laurent/draw/norm_bs=128,rec_gamma=1.0,gamma=1.0-20160913-181618/log', 'Norm Prop 1.0 1.0'),
        ('/Tmp/laurent/draw/norm_bs=128,rec_gamma=0.1,gamma=1.0,no_train_gamma_lstm-20160914-123123/log', 'Norm Prop 0.1 1.0 no'),
        ('/Tmp/laurent/draw/norm_bs=128,rec_gamma=1.0,gamma=1.0,no_train_gamma_lstm-20160914-122819/log', 'Norm Prop 1.0 1.0 no'),
        ('/Tmp/laurent/draw/norm_bs=128,rec_gamma=auto,gamma=1.0,no_train_gamma_lstm-20160914-123146/log', 'Norm Prop auto 1.0 no'),
        ('/Tmp/laurent/draw/norm_bs=128,lr=0.001,rec_gamma=1.0,gamma=1.0,no_train_gamma_lstm,no_c_gamma-20160914/log', 'Norm Prop 1.0 1.0 no no 0.001'),
        #('/Tmp/laurent/draw/norm_bs=128,lr=0.001,rec_gamma=auto,gamma=1.0,no_train_gamma_lstm,no_c_gamma-20160914/log', 'Norm Prop auto 1.0 no no 0.001'),
        ('/Tmp/laurent/draw/norm_bs=128,rec_gamma=1.0,gamma=1.0,no_train_gamma_lstm-20160914/log', 'Norm Prop 1.0 1.0 no'),
        ('/Tmp/laurent/draw/norm_bs=128,rec_gamma=1.0,gamma=1.0,no_train_gamma_lstm-20160914/log', 'Norm Prop 1.0 1.0 no'),
        ('/Tmp/laurent/draw/norm_bs=128,rec_gamma=1.0,gamma=1.0,no_train_gamma_lstm-20160914/log', 'Norm Prop 1.0 1.0 no'),
        #('/Tmp/laurent/draw/norm_bs=128,rec_gamma=auto,gamma=1.0,no_train_gamma_lstm-20160914/log', 'Norm Prop auto auto no'),
        ]
#plot(logs, title='DRAW', xlim=[0,100], ylim=[80,210])

logs = [('/Tmp/laurent/draw/baseline_bs=128,lr=0.001-20160913-111227/log', 'Baseline'),
        #('/Tmp/laurent/draw/new/baseline_bs=128,lr=0.001-20160915/log', 'BL 0.001'), # BAD
        #('/Tmp/laurent/draw/new/ln_bs=128,lr=0.001,c_gamma=1.0-20160915/log', 'LN 0.001'),
        ('/Tmp/laurent/draw/new/ln_bs=128,lr=0.001,c_gamma=1.0,run=2-20160916/log', 'LN'),
        #('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=1.0,c_gamma=1.0-20160915/log', 'NP 1.0 1.0'),
        #('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=0.1,c_gamma=1.0-20160915/log', 'NP 0.1 1.0'), #BAD
        #('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=1.0,c_gamma=0.1-20160915/log', 'NP 1.0 0.1'),
        #('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=0.1,c_gamma=0.1-20160915/log', 'NP 0.1 0.1'), #BAD
        #('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=0.5,c_gamma=0.1-20160915/log', 'NP 0.5 0.1'),
        #('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=0.5,c_gamma=1.0-20160915/log', 'NP 0.5 1.0'),
        #('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=0.2,c_gamma=0.2-20160916/log', 'NP 0.2 0.2'),
        #('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=0.4,c_gamma=0.4-20160916/log', 'NP 0.4 0.4'),
        ('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=0.5,c_gamma=0.5-20160915/log', 'NP 0.5 0.5'), # BEST
        #('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=0.6,c_gamma=0.6,run=2-20160916/log', 'NP 0.6 0.6'),
        #('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=0.8,c_gamma=0.8-20160916/log', 'NP 0.8 0.8'),
        #('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=0.8,c_gamma=0.4-20160916/log', 'NP 0.8 0.4'),
        #('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=0.4,c_gamma=0.8-20160916/log', 'NP 0.4 0.8'),
        #('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=0.6,c_gamma=0.3-20160917/log', 'NP 0.6 0.3'),
        #('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=0.3,c_gamma=0.6,run=2-20160916/log', 'NP 0.3 0.6'),
        ('/Tmp/laurent/draw/new/baseline_bs=128,lr=0.001,rec_gamma=0.5,c_gamma=1.0,ortho-20160922/log', 'Baseline Ortho'),
        ('/Tmp/laurent/draw/new/ln_bs=128,lr=0.001,ortho-20160922/log', 'LN Ortho'),
        ('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=0.5,c_gamma=0.5,ortho-20160922/log', 'NP Ortho 0.5'),
        ('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=0.1,c_gamma=0.1,ortho-20160923/log', 'NP Ortho 0.1'),
        ('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=1.0,c_gamma=1.0,ortho-20160923/log', 'NP Ortho 1.0'),
        ('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=0.3,c_gamma=0.3,ortho-20160923/log', 'NP Ortho 0.3'),
       ]
#plot(logs, title='DRAW', xlim=[0,100], ylim=[80,240])

#d = '/Tmp/laurent/draw/new'
#logs = [os.path.join(d, x, 'log') for x in os.listdir(d) if 'scale' in x]
#logs.insert(0, ('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=0.5,c_gamma=0.5-20160915/log', 'BL'))
#plot(logs, title='Grid', ylim=[80, 250])


logs = [#('/Tmp/laurent/draw/baseline_bs=128,lr=0.001-20160913-111227/log', 'Baseline'),
        ('/Tmp/laurent/draw/new/ln_bs=128,lr=0.001,c_gamma=1.0,run=2-20160916/log', 'LN'),
        ('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=0.5,c_gamma=0.5-20160915/log', 'NP 0.5 0.5'), # BEST
        #('/Tmp/laurent/draw/new/baseline_bs=128,lr=0.001,rec_gamma=0.5,c_gamma=1.0,ortho-20160922/log', 'Baseline Ortho'),
        #('/Tmp/laurent/draw/new/baseline_bs=128,lr=0.001,ortho,forget_bias-20160929/log', 'Baseline Ortho FB'),
        #('/Tmp/laurent/draw/new/baseline_bs=128,lr=0.001,forget_bias-20160929/log', 'Baseline FB'),
        ('/Tmp/laurent/draw/new/ln_bs=128,lr=0.001,ortho-20160922/log', 'LN Ortho'),
        ('/Tmp/laurent/draw/new/ln_bs=128,lr=0.001,ortho,forget_bias-20160929/log', 'LN Ortho FB'),
        ('/Tmp/laurent/draw/new/ln_bs=128,lr=0.001,forget_bias-20160929/log', 'LN FB'),
        ('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=0.5,c_gamma=0.5,bl,forget_bias-20160929/log', 'NP BL 0.5'),
        ('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=0.1,c_gamma=0.1,bl,forget_bias-20160929/log', 'NP BL 0.1'),
        ('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=1.0,c_gamma=1.0,bl,forget_bias-20160929/log', 'NP BL 1.0'),
        ('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=0.3,c_gamma=0.3,bl,forget_bias-20160929/log', 'NP BL 0.3'),
       ]
#plot(logs, title='DRAW', xlim=[0,100], ylim=[80,240])

logs = [#('/Tmp/laurent/draw/baseline_bs=128,lr=0.001-20160913-111227/log', 'Baseline'),
        #('/Tmp/laurent/draw/new/ln_bs=128,lr=0.001,c_gamma=1.0,run=2-20160916/log', 'LN'),
        ('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=0.5,c_gamma=0.5-20160915/log', 'NP 0.5 0.5'), # BEST
        #('/Tmp/laurent/draw/new/norm_bs=128,lr=0.01,rec_gamma=0.1,c_gamma=0.1,force_norm-20160930/log', 'NP FN 0.1'),
        ('/Tmp/laurent/draw/new/norm_bs=128,lr=0.01,lr_decay=1e-6,rec_gamma=0.5,c_gamma=0.5,force_norm-20161001/log', 'NP FN 0.5 1e-6'),
        ('/Tmp/laurent/draw/new/norm_bs=128,lr=0.01,lr_decay=1e-7,rec_gamma=0.5,c_gamma=0.5,force_norm-20161001/log', 'NP FN 0.5 1e-7'),
        #('/Tmp/laurent/draw/new/norm_bs=128,lr=0.01,lr_decay=1e-8,rec_gamma=0.5,c_gamma=0.5,force_norm-20161001/log', 'NP FN 0.5 1e-8'),
        #('/Tmp/laurent/draw/new/norm_bs=128,lr=0.01,lr_decay=1e-9,rec_gamma=0.5,c_gamma=0.5,force_norm-20161001/log', 'NP FN 0.5 1e-9'),
        ('/Tmp/laurent/draw/new/norm_bs=128,lr=0.01,lr_decay=5e-3,rec_gamma=0.5,c_gamma=0.5,force_norm,new_decay-20161001/log', 'NP FN 0.5 5e-3'),
        ('/Tmp/laurent/draw/new/norm_bs=128,lr=0.01,lr_decay=1e-3,rec_gamma=0.5,c_gamma=0.5,force_norm,new_decay-20161001/log', 'NP FN 0.5 1e-3'),
        ('/Tmp/laurent/draw/new/norm_bs=128,lr=0.01,lr_decay=5e-4,rec_gamma=0.5,c_gamma=0.5,force_norm,new_decay-20161001/log', 'NP FN 0.5 5e-4'),
        ('/Tmp/laurent/draw/new/norm_bs=128,lr=0.01,lr_decay=1e-4,rec_gamma=0.5,c_gamma=0.5,force_norm,new_decay-20161001/log', 'NP FN 0.5 1e-4'),
        ('/Tmp/laurent/draw/new/norm_bs=128,lr=0.01,lr_decay=5e-5,rec_gamma=0.5,c_gamma=0.5,force_norm,new_decay-20161001/log', 'NP FN 0.5 5e-5'),
       ]
#plot(logs, title='DRAW', xlim=[0,100], ylim=[80,240])


logs = [#('/Tmp/laurent/draw/baseline_bs=128,lr=0.001-20160913-111227/log', 'Baseline'),
        ('/Tmp/laurent/draw/new/baseline_bs=128,lr=0.01,lr_decay=1e-3,ortho-20161002/log', 'Baseline'),
        ('/data/lisatmp3/laurent/draw/baseline,lr=0.001,lr_decay=1e-4,init=baseline-20161024/log', 'Baseline New BL'),
        ('/data/lisatmp3/laurent/draw/baseline,lr=0.01,lr_decay=1e-3,init=ortho-20161024/log', 'Baseline New Orth'),
        #('/Tmp/laurent/draw/new/baseline_bs=128,lr=0.01,lr_decay=1e-3,baseline-20161002/log', 'BL bl'),
        ('/Tmp/laurent/draw/new/ln_bs=128,lr=0.001,c_gamma=1.0,run=2-20160916/log', 'Layer Norm'),
        #('/Tmp/laurent/draw/new/ln_bs=128,lr=0.01,lr_decay=1e-3,ortho-20161002/log', 'LN ortho'),
        #('/Tmp/laurent/draw/new/ln_bs=128,lr=0.01,lr_decay=1e-3,baseline-20161002/log', 'LN bl'),
        #('/Tmp/laurent/draw/new/norm_bs=128,lr=0.001,rec_gamma=0.5,c_gamma=0.5-20160915/log', 'NP 0.5 0.5'), # BEST
        #('/Tmp/laurent/draw/new/norm_bs=128,lr=0.01,rec_gamma=0.1,c_gamma=0.1,force_norm-20160930/log', 'NP FN 0.1'),
        ('/Tmp/laurent/draw/new/norm_bs=128,lr=0.01,lr_decay=1e-3,rec_gamma=0.5,c_gamma=0.5,force_norm,new_decay-20161001/log', 'Norm Prop'),#'NP FN 0.5 1e-3'),
       ]
#plot(logs, title='DRAW', xlim=[0,120], ylim=[80,200])

path = '/data/lisatmp3/laurent/draw'

logs = [os.path.join(path, x, 'log') for x in sorted(os.listdir(path))]

logs = [
        ('/data/lisatmp3/laurent/draw/baseline,lr=0.01,lr_decay=1e-3,init=ortho-20161024/log', 'Baseline'),
        ('/data/lisatmp3/laurent/draw/norm,lr=0.01,lr_decay=1e-3,force,gamma=0.5,init=ortho-20161027/log', 'Norm Prop'),
        ('/data/lisatmp3/laurent/draw/wn,lr=0.01,lr_decay=1e-3,force,gamma=1.0,init=ortho-20161027/log', 'Weight Norm')
       ]


plot(logs, title='DRAW', xlim=[0,100], ylim=[80,100])
plt.show()
