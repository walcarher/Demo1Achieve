#MIT License

# This software has been heavily inspired from: JetsonHacks YouTube videos, examples and GitHub Code
# Please refer to : https://github.com/jetsonhacks/gpuGraphTX

# Modifications to include instant bar graphs and also CPU average usage history
# by: Walther Carballo Hernandez
# Please refer to: https://github.com/walcarher
# Modifications Copyright (c) 2019 Institut Pascal

#Copyright (c) 2018 Jetsonhacks

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# This files are to be found in the L4T version of TX2, this may vary in the future
cpuLoadFile = '/proc/stat'
usage = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
total_last = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
idle_last = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
fig = plt.figure(figsize=(7,3))
plt.subplots_adjust(top=0.85, bottom=0.30)
fig.set_facecolor('#F2F1F0')
fig.canvas.set_window_title('CPUs Usage History Monitor')
# Subplot for the GPU activity
cpuAx = plt.subplot2grid((1,1), (0,0), rowspan=2, colspan=1)

# For the comparison
cpuLine, = cpuAx.plot([],[])

# The line points in x,y list form
cpuy_list = deque([0]*240)
cpux_list = deque(np.linspace(60,0,num=240))

fill_lines=0

def initGraph():
    global cpuAx
    global cpuLine
    global fill_lines


    cpuAx.set_xlim(60, 0)
    cpuAx.set_ylim(-5, 105)
    cpuAx.set_title('CPUs Average History')
    cpuAx.set_ylabel('CPU Average Usage (%)')
    cpuAx.set_xlabel('Samples');
    cpuAx.grid(color='gray', linestyle='dotted', linewidth=1)

    cpuLine.set_data([],[])
    cpuLine.set_color('blue')
    fill_lines=cpuAx.fill_between(cpuLine.get_xdata(),50,0)

    return [cpuLine] + [fill_lines]

def updateGraph(frame):
    global fill_lines
    global cpuy_list
    global cpux_list
    global cpuLine
    global cpuAx

 
    # Now draw the GPU usage
    cpuy_list.popleft()
    with open(cpuLoadFile, 'r') as cpuFile:
    	cpuFile.readline() # Discard global CPU top line
    	for i in range(len(total_last)):
    		cpu = cpuFile.readline().split()
    		total = np.sum(map(float,cpu[1:9]))
    		idle = float(cpu[4])	
    		delta_total = total - total_last[i]
    		delta_idle = idle - idle_last[i]
    		usage[i] = (1000*(delta_total-delta_idle)/delta_total+5)/10
    		total_last[i] = total
    		idle_last[i] = idle
    # The GPU load is stored as a percentage * 10, e.g 256 = 25.6%
    cpuy_list.append(np.sum(usage)/6)
    cpuLine.set_data(cpux_list,cpuy_list)
    fill_lines.remove()
    fill_lines=cpuAx.fill_between(cpux_list,0,cpuy_list, facecolor='blue', alpha=0.50)

    return [cpuLine] + [fill_lines]


# Keep a reference to the FuncAnimation, so it does not get garbage collected
animation = FuncAnimation(fig, updateGraph, frames=200,
                    init_func=initGraph,  interval=100, blit=True)

plt.show()
