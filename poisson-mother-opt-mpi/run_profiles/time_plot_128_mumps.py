import numpy as np
from matplotlib import pyplot as plt



#--------------------- time plot for optimization -----------------

f1 = plt.figure(1)
N = 4

t1 = (103.00, 35.03, 33.63, 25.77)
t2 = (101.00, 47.53, 64.73, 58.53)
t3 = (96.93, 32.27, 23.57, 17.83)
t4 = (0., 159.67, 161.33, 179.)
t5 = (131.33, 70.90, 84.67, 82.67)
t6 = (164.73, 1.27, 2.73, 4.87)

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

t1 = (95.81, 61.25, 50.96, 47.02)
t2 = (93.95, 83.10, 98.08, 106.81)
t3 = (90.17, 56.41, 35.71, 32.54)
t4 = (0.00, 279.14, 244.44, 326.64)
t5 = (122.17, 123.95, 128.28, 150.85)
t6 = (153.24, 2.21, 4.14, 8.88)
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

