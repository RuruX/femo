import numpy as np
from matplotlib import pyplot as plt



#--------------------- time plot for optimization -----------------

f1 = plt.figure(1)
N = 4


t1 = (490.67,331.00,210.67,164.33)  # time for solve_nonlinear
t2 = (185.33,142.00,124.33,132.33)  # time for solve_linear
t3 = (563.00,368.67,196.33,135.67)   # time for linearize
t4 = (5.92,7.50,10.94,17.51)     # time for mpi.comm.bcast
t5 = (788.83,870.50,848.77,995.93)  # time for convertToCSR
t6 = (2701.25,2731.33,2733.63,2900.22)     # time for others

t12 = np.array(t1)+np.array(t2)
t123 = np.array(t1)+np.array(t2)+np.array(t3)
t1234 = np.array(t1)+np.array(t2)+np.array(t3)+np.array(t4)
t12345 = np.array(t1)+np.array(t2)+np.array(t3)+np.array(t4)+np.array(t5)
num_pc = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence


from matplotlib import cm
cmap = cm.get_cmap('Blues')
colors = cmap(np.linspace(0.1,1,4))
p1 = plt.bar(num_pc, t1, width, color=colors[0])
p2 = plt.bar(num_pc, t2, width, bottom=t1, color=colors[1])
p3 = plt.bar(num_pc, t3, width, bottom=t12, color=colors[2])
p4 = plt.bar(num_pc, t4, width, bottom=t123, color=colors[3])
p5 = plt.bar(num_pc, t5, width, bottom=t1234, color='slategray')
p6 = plt.bar(num_pc, t6, width, bottom=t12345, color='lightgray')


plt.ylabel('Time (s)')
plt.xlabel('Number of processors')
plt.title('Time Performance for Parallelization')
plt.xticks(num_pc, ('1', '2', '4', '8'))
plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0]), ('solve_nonlinear', 
    'solve_linear', 'linearize', 'mpi.comm.bcast', 'convertToCSR', 'others'))


#--------------------- time plot for optimization -----------------
#---------------------(factorized with number of iterations)-----------------
f2 = plt.figure()

N = 4

t1 = (308.40,187.64,124.88,104.27)
t2 = (116.49,80.50,73.70,83.97)
t3 = (353.87,208.99,116.38,86.08)
t4 = (3.72,4.25,6.48,11.11)
t5 = (495.81,493.48,503.12,631.94)
t6 = (1697.83,1548.37,1620.41,1840.24)
t12 = np.array(t1)+np.array(t2)
t123 = np.array(t1)+np.array(t2)+np.array(t3)
t1234 = np.array(t1)+np.array(t2)+np.array(t3)+np.array(t4)
t12345 = np.array(t1)+np.array(t2)+np.array(t3)+np.array(t4)+np.array(t5)
num_pc = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence


from matplotlib import cm
cmap = cm.get_cmap('Blues')
colors = cmap(np.linspace(0.1,1,4))
p1 = plt.bar(num_pc, t1, width, color=colors[0])
p2 = plt.bar(num_pc, t2, width, bottom=t1, color=colors[1])
p3 = plt.bar(num_pc, t3, width, bottom=t12, color=colors[2])
p4 = plt.bar(num_pc, t4, width, bottom=t123, color=colors[3])
p5 = plt.bar(num_pc, t5, width, bottom=t1234, color='slategray')
p6 = plt.bar(num_pc, t6, width, bottom=t12345, color='lightgray')

plt.ylabel('Time (s)')
plt.xlabel('Number of processors')
plt.title('Time per 1000 iteration for Parallelization')
plt.xticks(num_pc, ('1', '2', '4', '8'))
plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0]), ('solve_nonlinear', 
    'solve_linear', 'linearize', 'mpi.comm.bcast', 'convertToCSR', 'others'))


#---------------------- Time plot for each method --------------------------------
f3 = plt.figure()

# solve_nonlinear() 
p1 = plt.bar(num_pc-width/2, t1, width/2, color=colors[0])

# solve_linear() 
p2 = plt.bar(num_pc, t2, width/2, color=colors[1])

# linearize() 
p3 = plt.bar(num_pc+width/2, t3, width/2, color=colors[2])

plt.ylabel('Time (s)') 
plt.xlabel('Number of processors')
plt.title('Time for each method per 1000 iteration')
plt.xticks(num_pc, ('1', '2', '4', '8'))
plt.legend((p1[0], p2[0], p3[0]), ('solve_nonlinear', 'solve_linear', 'linearize'))

plt.show()

