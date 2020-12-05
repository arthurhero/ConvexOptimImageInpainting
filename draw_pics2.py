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
    mc_sums=np.zeros((10,))
    fo_sums=np.zeros((10,))
    count=0.0

    for i in range(len(lines)):
        line = lines[i]
        if "img" in line:
            count+=1
            idx = int(line.split(':')[1])
            mc_sums[idx] += float(lines[i+1].split(':')[1])
            fo_sums[idx] += float(lines[i+2].split(':')[1])

    mc_sums/=(count/10.0)
    fo_sums/=(count/10.0)
    draw_line_plot(list(range(1,11)),mc_sums,"Average l1 Diff of Two Methods for Each Image","imgs.jpg",'Image Index',"L1 Diff (per pixel avg)",fo_sums,'Matrix Completion','Cosine Compressive Sensing')


if __name__ == '__main__':
    read_file_and_plot("log.txt")
