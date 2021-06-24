import numpy as np
from matplotlib import pyplot as plt

#--------------------- time plot for nonlinear solver -----------------
#num_el = np.array([128, 256, 512, 1024, 2048])
#num_pc = [1, 2, 4, 8]
#xi = list(range(len(num_pc)))
#print(xi)
## for num_el = 2048
#time = np.array([117.01, 79.27, 49.19, 45.72])
#time_fraction = time[0]/time

#plt.plot(xi, time_fraction, 'ro')
##plt.xscale('log')
#plt.xticks(xi, num_pc)
#plt.xlabel('number of processors')
#plt.ylabel('speed up fraction')
#plt.show()


#--------------------- time plot for optimization -----------------
N = 4

t1 = (131.33, 70.90, 84.67, 82.67)
t2 = (103.00, 35.03, 33.63, 25.77)
t3 = (101.00, 47.53, 64.73, 58.53)
t4 = (96.93, 32.27, 23.57, 17.83)
t5 = (0., 159.67, 161.33, 179.)
t6 = (164.73, 1.27, 2.73, 4.87)

t12 = np.array(t1)+np.array(t2)
t123 = np.array(t1)+np.array(t2)+np.array(t3)
t1234 = np.array(t1)+np.array(t2)+np.array(t3)+np.array(t4)
t12345 = np.array(t1)+np.array(t2)+np.array(t3)+np.array(t4)+np.array(t5)
num_pc = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

#p1 = plt.bar(num_pc, t1, width, color='goldenrod')
#p2 = plt.bar(num_pc, t2, width, bottom=t1, color='slategray')
#p3 = plt.bar(num_pc, t3, width, bottom=t12, color='lightblue')
#p4 = plt.bar(num_pc, t4, width, bottom=t123, color='steelblue')
#p5 = plt.bar(num_pc, t5, width, bottom=t1234, color='lightsteelblue')
#p6 = plt.bar(num_pc, t6, width, bottom=t12345, color='darkgrey')



from matplotlib import cm
cmap = cm.get_cmap('Blues')
colors = cmap(np.linspace(0,0.9,5))
p1 = plt.bar(num_pc, t1, width, color=colors[0])
p2 = plt.bar(num_pc, t2, width, bottom=t1, color=colors[1])
p3 = plt.bar(num_pc, t3, width, bottom=t12, color=colors[2])
p4 = plt.bar(num_pc, t4, width, bottom=t123, color=colors[3])
p5 = plt.bar(num_pc, t5, width, bottom=t1234, color=colors[4])
p6 = plt.bar(num_pc, t6, width, bottom=t12345, color='lightgray')


plt.ylabel('Time (s)')
plt.xlabel('Number of processors')
plt.title('Time Performance for Parallelization')
plt.xticks(num_pc, ('1', '2', '4', '8'))
plt.yticks(np.arange(0, 601, 100))
plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0]), ('convertToCSR', 
   'solve_nonlinear', 'solve_linear', 'linearize', 'mpi.comm.bcast', 'others'))

plt.show()



#--------------------- time plot for optimization -----------------
#---------------------(factorized with number of iterations)-----------------
N = 4
t1 = (122.17, 123.95, 128.28, 150.85)
t2 = (95.81, 61.25, 50.96, 47.02)
t3 = (93.95, 83.10, 98.08, 106.81)
t4 = (90.17, 56.41, 35.71, 32.54)
t5 = (0.00, 279.14, 244.44, 326.64)
t6 = (153.24, 2.21, 4.14, 8.88)
t12 = np.array(t1)+np.array(t2)
t123 = np.array(t1)+np.array(t2)+np.array(t3)
t1234 = np.array(t1)+np.array(t2)+np.array(t3)+np.array(t4)
t12345 = np.array(t1)+np.array(t2)+np.array(t3)+np.array(t4)+np.array(t5)
num_pc = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence


from matplotlib import cm
#cmap = cm.get_cmap('PuBu')
cmap = cm.get_cmap('Blues')
colors = cmap(np.linspace(0,0.9,5))
p1 = plt.bar(num_pc, t1, width, color=colors[0])
p2 = plt.bar(num_pc, t2, width, bottom=t1, color=colors[1])
p3 = plt.bar(num_pc, t3, width, bottom=t12, color=colors[2])
p4 = plt.bar(num_pc, t4, width, bottom=t123, color=colors[3])
p5 = plt.bar(num_pc, t5, width, bottom=t1234, color=colors[4])
p6 = plt.bar(num_pc, t6, width, bottom=t12345, color='lightgray')

plt.ylabel('Time (s)')
plt.xlabel('Number of processors')
plt.title('Time per 1000 iteration for Parallelization')
plt.xticks(num_pc, ('1', '2', '4', '8'))
plt.yticks(np.arange(0, 1001, 100))
plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0]), ('convertToCSR', 
   'solve_nonlinear', 'solve_linear', 'linearize', 'mpi.comm.bcast', 'others'))

plt.show()



p1 = plt.bar(num_pc, t2, width, color=colors[0])

plt.ylabel('Time (s)')
plt.xlabel('Number of processors')
plt.title('Time for solve_nonlinear per 1000 iteration')
plt.xticks(num_pc, ('1', '2', '4', '8'))
plt.yticks(np.arange(0, 101, 20))
#plt.legend((p1[0]), ('solve_nonlinear'))

plt.show()



