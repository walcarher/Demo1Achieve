#MIT License

# This software has been heavily inspired from: JetsonHacks YouTube videos, examples and GitHub Code
# Please refer to : https://github.com/jetsonhacks/gpuGraphTX

# Modifications to include instant bar graphs and also CPU average usage history
# by: Walther Carballo Hernandez
# Please refer to: https://github.com/walcarher
# Modifications Copyright (c) 2019-2021 Institut Pascal

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

# This files are to be found in the L4T version of TX2 and on C10GX CHIMERA board, this may vary in the future
cpuLoadFile = '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_power1_input' # CPU power channel
gpuLoadFile = '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/in_power0_input' # GPU power channel
fpgaLoadFile = '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/in_power0_input' # FPGA core power channel

fig = plt.figure(figsize=(7,3))
plt.subplots_adjust(top=0.85, bottom=0.30)
fig.set_facecolor('#F2F1F0')
fig.canvas.set_window_title('CPU-FPGA-GPU Power Monitor')

# Subplot for the CPU-FPGA-GPU activity
ax = plt.subplot2grid((1,1), (0,0), rowspan=2, colspan=1)

# For the comparison
cpuLine, = ax.plot([],[])
gpuLine, = ax.plot([],[])
fpgaLine, = ax.plot([],[])

# The line points in x,y list form
cpuy_list = deque([0]*240)
cpux_list = deque(np.linspace(60,0,num=240))
gpuy_list = deque([0]*240)
gpux_list = deque(np.linspace(60,0,num=240))
fpgay_list = deque([0]*240)
fpgax_list = deque(np.linspace(60,0,num=240))

cpu_fill_lines=0
gpu_fill_lines=0
fpga_fill_lines=0

def initGraph():
    global ax
    global cpuLine
    global gpuLine
    global fpgaLine
    global cpu_fill_lines
    global gpu_fill_lines
    global fpga_fill_lines

    ax.set_xlim(60, 0)
    ax.set_ylim(-10, 10000)
    ax.set_title('CPU-FPGA-GPU Power Plot')
    ax.set_ylabel('Power (mW)')
    ax.set_xlabel('Samples');
    ax.grid(color='gray', linestyle='dotted', linewidth=1)

    cpuLine.set_data([],[])
    cpuLine.set_color('gray')
    cpuLine.set_label('CPU')
    cpu_fill_lines=ax.fill_between(cpuLine.get_xdata(),50,0)
    gpuLine.set_data([],[])
    gpuLine.set_color('green')
    gpuLine.set_label('GPU')
    gpu_fill_lines=ax.fill_between(gpuLine.get_xdata(),50,0)
    fpgaLine.set_data([],[])
    fpgaLine.set_color('blue')
    fpgaLine.set_label('FPGA')
    fpga_fill_lines=ax.fill_between(fpgaLine.get_xdata(),50,0)

    ax.legend(loc='upper left')

    return [cpuLine] + [cpu_fill_lines] + [fpgaLine] + [fpga_fill_lines] + [gpuLine] + [gpu_fill_lines]

def updateGraph(frame):
    global cpu_fill_lines
    global cpuy_list
    global cpux_list
    global cpuLine
    global gpu_fill_lines
    global gpuy_list
    global gpux_list
    global gpuLine
    global fpga_fill_lines
    global fpgay_list
    global fpgax_list
    global fpgaLine
    global ax

    # Draw average CPU power consumption
    cpuy_list.popleft()
    try:
        with open(cpuLoadFile, 'r') as cpuFile:
          fileData = cpuFile.read()
    except IOError:
        fileData = '0'
    # The CPU power is in mW
    cpuy_list.append(int(fileData))
    cpuLine.set_data(cpux_list,cpuy_list)
    cpu_fill_lines.remove()
    cpu_fill_lines=ax.fill_between(cpux_list,0,cpuy_list, facecolor='gray', alpha=0.25)
    # Draw average GPU power consumption
    gpuy_list.popleft()
    try:
        with open(gpuLoadFile, 'r') as gpuFile:
          fileData = gpuFile.read()
    except IOError:
        fileData = '0'
    # The GPU power is in mW
    gpuy_list.append(int(fileData))
    gpuLine.set_data(gpux_list,gpuy_list)
    gpu_fill_lines.remove()
    gpu_fill_lines=ax.fill_between(gpux_list,0,gpuy_list, facecolor='green', alpha=0.25)
    # Draw average FPGA power consumption
    fpgay_list.popleft()
    try:
        with open(fpgaLoadFile, 'r') as fpgaFile:
          fileData = fpgaFile.read()
    except IOError:
        fileData = '0'
    # The FPGA power is in mW
    fpgay_list.append(int(fileData))
    fpgaLine.set_data(fpgax_list,fpgay_list)
    fpga_fill_lines.remove()
    fpga_fill_lines=ax.fill_between(fpgax_list,0,fpgay_list, facecolor='blue', alpha=0.25)

    return [cpuLine] + [cpu_fill_lines] + [fpgaLine] + [fpga_fill_lines] + [gpuLine] + [gpu_fill_lines]

# Keep a reference to the FuncAnimation, so it does not get garbage collected
animation = FuncAnimation(fig, updateGraph, frames=200,
                    init_func=initGraph,  interval=100, blit=True)

plt.show()
