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

devices = ('CPU0', 'CPU1', 'CPU2', 'CPU3', 'CPU4', 'CPU5', 'GPU')
usage = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
total_last = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
idle_last = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
time_interval = 50 # Miliseconds update
x_axis = np.arange(len(devices))
# This files are to be found in the L4T version of TX2, this may vary in the future
gpuUsageFile = '/sys/devices/gpu.0/load'
cpuUsageFile = '/proc/stat'
# Plot characteristics
fig, ax = plt.subplots(num='Device Usage Monitor', figsize=(7,3))
ax.set_title('Device Usage')
plt.ylabel('Usage (%)')

def initGraph():
	ax.set_ylim(0,100)
	ln = plt.bar(x_axis[0:7], usage[0:7], width=1.0, color='b', align='center', alpha=0.5)
	ln[6].set_color('g')	
	ln[6].set_alpha(0.5)	
	plt.xticks(x_axis, devices)
	return ln

def updateGraph(frame):
	with open(cpuUsageFile,'r') as cpuFile:
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
	with open(gpuUsageFile, 'r') as gpuFile:
		usage[6] = float(gpuFile.read())/10
	ln = plt.bar(x_axis[0:7], usage[0:7], width=1.0, color='b', align='center', alpha=0.8)	
	ln[6].set_color('g')	
	return ln


animation = FuncAnimation(fig, updateGraph, interval=time_interval, init_func=initGraph, blit=True)

plt.show()
