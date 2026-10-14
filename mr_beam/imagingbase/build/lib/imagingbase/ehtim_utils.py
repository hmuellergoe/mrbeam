'''
This file contains slightly modified code of the ehtim (https://github.com/achael/eht-imaging) library to keep MrBeam executable with ehtim.
'''

import numpy as np
import numpy.random as npr
import pandas as pd

from ehtim.statistics.dataframes import make_df, get_bins_labels, df_to_rec
import datetime as datetime
from astropy.time import Time

def coh_avg_vis(obs,dt=0,scan_avg=False,return_type='rec',err_type='predicted',num_samples=int(1e3), seed=12):
    """coherently averages visibilities
    Args:
        obs: ObsData object
        dt (float): integration time in seconds
        return_type (str): 'rec' for numpy record array (as used by ehtim), 'df' for data frame
        err_type (str): 'predicted' for modeled error, 'measured' for bootstrap empirical variability estimator
        num_samples: 'bootstrap' resample set size for measured error
        scan_avg (bool): should scan-long averaging be performed. If True, overrides dt
    Returns:
        vis_avg: coherently averaged visibilities
    """
    if (dt<=0)&(scan_avg==False):
        return obs.data
    else:
        vis = make_df(obs)
        if scan_avg==False:
            #TODO
            #we don't have to work on datetime products at all
            #change it to only use 'time' in mjd
            t0 = datetime.datetime(1960,1,1) 
            vis['round_time'] = list(map(lambda x: np.floor((x- t0).total_seconds()/float(dt)),vis.datetime))  
            grouping=['tau1','tau2','polarization','band','baseline','t1','t2','round_time']
        else:
            bins, labs = get_bins_labels(obs.scans)
            vis['scan'] = list(pd.cut(vis.time, bins,labels=labs))
            grouping=['tau1','tau2','polarization','band','baseline','t1','t2','scan']
        #column just for counting the elements
        vis['number'] = 1
        aggregated = {'datetime': np.min, 'time': np.min,
        'number': lambda x: len(x), 'u':np.mean, 'v':np.mean,'tint': np.sum}

        if err_type not in ['measured', 'predicted']:
            print("Error type can only be 'predicted' or 'measured'! Assuming 'predicted'.")
            err_type='predicted'

        if obs.polrep=='stokes':
            vis1='vis'; vis2='qvis'; vis3='uvis'; vis4='vvis'
            sig1='sigma'; sig2='qsigma'; sig3='usigma'; sig4='vsigma'
        elif obs.polrep=='circ':
            vis1='rrvis'; vis2='llvis'; vis3='rlvis'; vis4='lrvis'
            sig1='rrsigma'; sig2='llsigma'; sig3='rlsigma'; sig4='lrsigma'

        #AVERAGING-------------------------------    
        if err_type=='measured':
            vis['dummy'] = vis[vis1]
            vis['qdummy'] = vis[vis2]
            vis['udummy'] = vis[vis3]
            vis['vdummy'] = vis[vis4]
            meanF = lambda x: np.nanmean(np.asarray(x))
            meanerrF = lambda x: bootstrap(np.abs(x), np.mean, num_samples=num_samples,wrapping_variable=False, seed=seed)
            aggregated[vis1] = meanF
            aggregated[vis2] = meanF
            aggregated[vis3] = meanF
            aggregated[vis4] = meanF
            aggregated['dummy'] = meanerrF
            aggregated['udummy'] = meanerrF
            aggregated['vdummy'] = meanerrF
            aggregated['qdummy'] = meanerrF
       
        elif err_type=='predicted':
            meanF = lambda x: np.nanmean(np.asarray(x))
            #meanerrF = lambda x: bootstrap(np.abs(x), np.mean, num_samples=num_samples,wrapping_variable=False)
            def meanerrF(x):
                x = np.asarray(x)
                x = x[x==x]
                
                if len(x)>0: ret = np.sqrt(np.sum(x**2)/len(x)**2)
                else: ret = np.nan +1j*np.nan
                return ret
              
            aggregated[vis1] = meanF
            aggregated[vis2] = meanF
            aggregated[vis3] = meanF
            aggregated[vis4] = meanF
            aggregated[sig1] = meanerrF
            aggregated[sig2] = meanerrF
            aggregated[sig3] = meanerrF
            aggregated[sig4] = meanerrF

        #ACTUAL AVERAGING
        vis_avg = vis.groupby(grouping).agg(aggregated).reset_index()
        
        if err_type=='measured':
            vis_avg[sig1] = [0.5*(x[1][1]-x[1][0]) for x in list(vis_avg['dummy'])]
            vis_avg[sig2] = [0.5*(x[1][1]-x[1][0]) for x in list(vis_avg['qdummy'])]
            vis_avg[sig3] = [0.5*(x[1][1]-x[1][0]) for x in list(vis_avg['udummy'])]
            vis_avg[sig4] = [0.5*(x[1][1]-x[1][0]) for x in list(vis_avg['vdummy'])]

        vis_avg['amp'] = list(map(np.abs,vis_avg[vis1]))
        vis_avg['phase'] = list(map(lambda x: (180./np.pi)*np.angle(x),vis_avg[vis1]))
        vis_avg['snr'] = vis_avg['amp']/vis_avg[sig1]

        if scan_avg==False:
            #round datetime and time to the begining of the bucket and add half of a bucket time
            half_bucket = dt/2.
            vis_avg['datetime'] =  list(map(lambda x: t0 + datetime.timedelta(seconds= int(dt*x) + half_bucket), vis_avg['round_time']))
            vis_avg['time']  = list(map(lambda x: (Time(x).mjd-obs.mjd)*24., vis_avg['datetime']))
        else:
            #drop values that couldn't be matched to any scan
            vis_avg.drop(list(vis_avg[vis_avg.scan<0].index.values),inplace=True)
        if err_type=='measured':
            vis_avg.drop(labels=['udummy','vdummy','qdummy','dummy'],axis='columns',inplace=True)      
        if return_type=='rec':
            if obs.polrep=='stokes':
                return df_to_rec(vis_avg,'vis')
            elif obs.polrep=='circ':
                return df_to_rec(vis_avg,'vis_circ')
        elif return_type=='df':
            return vis_avg

def bootstrap(data, statistic, num_samples=int(1e3), alpha='1sig',wrapping_variable=False, seed=12):
    """bootstrap estimate of 100.0*(1-alpha) confidence interval for a given statistic
        Args:
            data: vector of data to estimate bootstrap statistic on
            statistic: function representing the statistic to be evaluated
            num_samples: number of bootstrap (re)samples
            alpha: parameter of the confidence interval, '1s' gets an analog of 1 sigma confidence for a normal variable
            wrapping_variable: True for circular variables, attempts to avoid problem related to estimating variability of wrapping variable

        Returns:
            bootstrap_value: bootstrap-estimated value of the statistic
            bootstrap_CI: bootstrap-estimated confidence interval
    """
    if alpha=='1sig':
        alpha=0.3173
    elif alpha=='2sig':
        alpha=0.0455
    elif alpha=='3sig':
        alpha=0.0027
    stat = np.zeros(num_samples)
    data = np.asarray(data)
    if wrapping_variable==True:
        m=statistic(data)
    else:
        m=0
    data = data-m
    n = len(data)
    npr.seed = seed
    idx = npr.randint(0, n, (num_samples, n))
    samples = data[idx]
    for cou in range(num_samples):
        stat[cou] = statistic(samples[cou,:])
    stat = np.sort(stat)
    bootstrap_value = np.median(stat)+m
    bootstrap_CI = [stat[int((alpha/2.0)*num_samples)]+m, stat[int((1-alpha/2.0)*num_samples)]+m]
    return bootstrap_value, bootstrap_CI  