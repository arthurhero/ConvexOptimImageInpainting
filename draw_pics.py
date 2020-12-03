import matplotlib.pyplot as plt 
import numpy as np

def draw_line_plot(x,y,title,filename,xlabel,ylabel,z=None,xleg=None,zleg=None):
    '''
    draw line plot x,z vs y
    x - x data points
    y - y data points
    title - plot title
    filename - save to filename
    z - optinal another line
    xleg - x legend
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
        plot1.plot(x,y,'r.-',label=xleg)
        plot1.plot(x,z,'b.-',label=zleg)
        plot1.legend()
    plt.savefig(filename)
    plt.close()

def read_file_and_plot(filename):
    lines = [line.rstrip('\n') for line in open(filename)]


if __name__ == '__main__':
    read_file_and_plot("log.txt")
