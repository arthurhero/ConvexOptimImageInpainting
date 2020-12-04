import matplotlib.pyplot as plt 
import numpy as np

def draw_line_plot(x,y,title,filename,xlabel,ylabel,z=None,yleg=None,zleg=None):
    '''
    draw line plot y,z vs x
    x - x data points
    y - y data points
    title - plot title
    filename - save to filename
    z - optinal another line
    yleg - y legend
    '''
    fig = plt.figure()
    fig.suptitle(title)
    fig.subplots_adjust(hspace=0.5)
    plot1 = fig.add_subplot(111)
    plot1.set_xlabel(xlabel)
    plot1.set_ylabel(ylabel)
    if z is None:
        plot1.plot(x, y)
    else:
        plot1.plot(x,y,'r.-',label=yleg)
        plot1.plot(x,z,'b.-',label=zleg)
        plot1.legend()
    plt.savefig(filename)
    plt.close()

def read_file_and_plot(filename):
    lines = [line.rstrip('\n') for line in open(filename)]
    mc_l1_dict = dict() # key: (percent, num_hole), value: l1 diff
    fo_l1_dict = dict()
    percents = set()
    num_holes = set()

    percent=""
    num_hole=""

    for i in range(len(lines)):
        line = lines[i]
        if "pencent" in line:
            percent = line.split(':')[1].strip()
            percents.add(percent)
        if "num_hole" in line:
            num_hole = line.split(':')[1].strip()
            num_holes.add(num_hole)
        if "mc_l1" in line:
            mc_l1 = float(line.split(':')[1])
            mc_l1_dict[(percent,num_hole)] = mc_l1
        if "fo_l1" in line:
            fo_l1 = float(line.split(':')[1])
            fo_l1_dict[(percent,num_hole)] = fo_l1
    for p in percents:
        xs = list()
        ys = list()
        zs = list()
        for nh in num_holes:
            xs.append(nh)
            ys.append(mc_l1_dict[(p,nh)])
            zs.append(fo_l1_dict[(p,nh)])
        draw_line_plot(xs,ys,"l1 Diff vs number of holes under "+p+" pixel loss","p"+p+".jpg",'Number of Square Holes',"L1 Diff (per pixel avg)",zs,'Matrix Completion','Cosine Compressive Sensing')


if __name__ == '__main__':
    read_file_and_plot("log.txt")
