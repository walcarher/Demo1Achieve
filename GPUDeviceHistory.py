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
gpuLoadFile = '/sys/devices/gpu.0/load'

fig = plt.figure(figsize=(7,3))
plt.subplots_adjust(top=0.85, bottom=0.30)
fig.set_facecolor('#F2F1F0')
fig.canvas.set_window_title('GPU Usage History Monitor')

# Subplot for the GPU activity
gpuAx = plt.subplot2grid((1,1), (0,0), rowspan=2, colspan=1)

# For the comparison
gpuLine, = gpuAx.plot([],[])

# The line points in x,y list form
gpuy_list = deque([0]*240)
gpux_list = deque(np.linspace(60,0,num=240))

fill_lines=0

def initGraph():
    global gpuAx
    global gpuLine
    global fill_lines


    gpuAx.set_xlim(60, 0)
    gpuAx.set_ylim(-5, 105)
    gpuAx.set_title('GPU History')
    gpuAx.set_ylabel('GPU Usage (%)')
    gpuAx.set_xlabel('Samples');
    gpuAx.grid(color='gray', linestyle='dotted', linewidth=1)

    gpuLine.set_data([],[])
    gpuLine.set_color('green')
    fill_lines=gpuAx.fill_between(gpuLine.get_xdata(),50,0)

    return [gpuLine] + [fill_lines]

def updateGraph(frame):
    global fill_lines
    global gpuy_list
    global gpux_list
    global gpuLine
    global gpuAx

 
    # Now draw the GPU usage
    gpuy_list.popleft()
    with open(gpuLoadFile, 'r') as gpuFile:
      fileData = gpuFile.read()
    # The GPU load is stored as a percentage * 10, e.g 256 = 25.6%
    gpuy_list.append(int(fileData)/10)
    gpuLine.set_data(gpux_list,gpuy_list)
    fill_lines.remove()
    fill_lines=gpuAx.fill_between(gpux_list,0,gpuy_list, facecolor='green', alpha=0.50)

    return [gpuLine] + [fill_lines]


# Keep a reference to the FuncAnimation, so it does not get garbage collected
animation = FuncAnimation(fig, updateGraph, frames=200,
                    init_func=initGraph,  interval=100, blit=True)

plt.show()
