"""Plot absolute rewards and the three prespecified contrasts."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = ('#176c70','#a74837','#58577c')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('run',type=Path)
    cli.add_argument('output',type=Path)
    args = cli.parse_args()
    groups = defaultdict(list)
    fields = ('opening_reward','terminal_reward','total')
    with (args.run/'cells.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            if row['family']=='all':
                groups[row['opener']].append([float(row[key]) for key in fields])
    names = ('return','terminal_only','gain','myopic','donor','uniform')
    tags = ('Cumulative return','Terminal payoff','Gain only','Immediate reward','Shuffled history','Uniform')
    means = np.asarray([np.mean(groups[name],axis=0) for name in names])
    effects = json.loads((args.run/'summary.json').read_text())['primary']
    fig,(left,right) = plt.subplots(1,2,figsize=(11.5,4.4),layout='constrained')
    y=np.arange(len(names))
    for i,(field,label) in enumerate(zip(fields,('First reward','Terminal reward','Sum'),strict=True)):
        left.scatter(means[:,i],y+(i-1)*.16,color=COLORS[i],s=25,marker=('o','s','D')[i],label=label)
    left.set_yticks(y,tags)
    left.invert_yaxis()
    left.set_xlabel('Mean reward on new games')
    left.set_title('Two live decisions')
    left.legend(frameon=False, fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(.5, -.19))
    for i,e in enumerate(effects):
        val = e['estimate']
        right.errorbar(val,i,xerr=[[val-e['lower']],[e['upper']-val]],fmt='o',
                       capsize=3,color=COLORS[0])
    right.set_yticks([0,1,2],['Return − gain only','Return − immediate','Reward use after rewind'])
    right.set_ylim(2.65,-.65)
    right.set_title('Prespecified comparisons')
    right.set_xlabel('Paired difference, 98.33% block intervals')
    for ax in (left,right):
        ax.axvline(0.,color='#666666',linewidth=.7)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x',alpha=.15)
    fig.savefig(args.output,dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
