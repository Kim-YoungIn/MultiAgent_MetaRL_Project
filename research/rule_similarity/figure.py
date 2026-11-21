"""Plot the D-031 transfer association and the structured-game sensitivity."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment.statistics import _rank


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('d031',type=Path)
    parser.add_argument('structured',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    fig,(left,right)=plt.subplots(1,2,figsize=(10.7,4.2),layout='constrained')
    family=defaultdict(list)
    for row in read_csv(args.d031/'games.csv'):
        family[row['family']].append(row)
    for name, rows in sorted(family.items()):
        dist=[float(r['strategy_similarity']) for r in rows]
        gain=np.array([float(r['meta_minus_reward']) for r in rows])
        left.scatter(np.array(_rank(dist))/(len(dist)-1),gain-gain.mean(),s=23,alpha=.75,label=name)
    left.set_title('Learned exploration in D-031')
    left.set_xlabel('Within-family strategy similarity rank')
    left.set_ylabel('Centered adaptation AUC difference (M − P)')
    left.legend(frameon=False,fontsize=8)
    groups=defaultdict(list)
    for row in read_csv(args.structured/'games.csv'):
        groups[row['block'],row['family']].append(row)
    means=defaultdict(list)
    for (block,_),rows in groups.items():
        values=np.array([[float(r['combined_similarity']),float(r['retained_gain']),
                          float(r['meta_extra_gain'])] for r in rows])
        order=np.argsort(values[:,0],kind='stable')
        bins=np.array_split(values[order,1:],4)
        means[block].append(np.array([b.mean(0)-values[:,1:].mean(0) for b in bins]))
    samples=np.array([np.mean(families,axis=0) for families in means.values()])
    draws=np.random.default_rng(81725).integers(len(samples),size=(10000,len(samples)))
    bounds=np.quantile(samples[draws].mean(1),[.025,.975],axis=0)
    x=np.arange(4)
    for i,(name,color) in enumerate([('Meta update gain','#156e72'),('Extra gain over pooled updates','#a24b39')]):
        right.plot(x,samples.mean(0)[:,i],marker='o',color=color,label=name)
        right.fill_between(x,bounds[0,:,i],bounds[1,:,i],color=color,alpha=.12)
    right.set_xticks(x,['Lowest','Q2','Q3','Highest'])
    right.set_xlabel('Combined similarity quartile within family and block')
    right.set_ylabel('Centered one-step payoff gain after 8 probes')
    right.set_title('Learning opponents in structured games')
    right.legend(frameon=False,fontsize=8)
    for ax in (left,right):
        ax.axhline(0.,color='#888888',lw=.7)
        ax.grid(axis='y',alpha=.15)
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(args.output,dpi=200)
    plt.close(fig)


if __name__=='__main__':
    main()
