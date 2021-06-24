import numpy as np
from matplotlib import pyplot as plt


N = 4
num_pc = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

from matplotlib import cm
cmap = cm.get_cmap('Blues')
colors = cmap(np.linspace(0.1,1,N))


def bar_plot(N,T):
    t1 = T[0][:]
    t2 = T[1][:]
    t3 = T[2][:]
    t4 = T[3][:]
    t5 = T[4][:]
    t6 = T[5][:]
    t12 = t1+t2
    t123 = t12+t3
    t1234 = t123+t4
    t12345 = t1234+t5

    p1 = plt.bar(num_pc, t1, width, color=colors[0])
    p2 = plt.bar(num_pc, t2, width, bottom=t1, color=colors[1])
    p3 = plt.bar(num_pc, t3, width, bottom=t12, color=colors[2])
    p4 = plt.bar(num_pc, t4, width, bottom=t123, color=colors[3])
    p5 = plt.bar(num_pc, t5, width, bottom=t1234, color='slategray')
    p6 = plt.bar(num_pc, t6, width, bottom=t12345, color='lightgray')

    plt.xticks(num_pc, ('1', '2', '4', '8'))
    plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0]), ('solve_nonlinear', 
    'solve_linear', 'linearize', 'mpi.comm.bcast', 'convertToCSR', 'others'))

#--------------------- time plot for optimization -----------------
#f1 = plt.figure(1)


#t1 = (717.00,489.67,273.67,431.00)  # time for solve_nonlinear
#t2 = (637.33,437.67,248.33,412.67)  # time for solve_linear
#t3 = (194.00,113.33,51.40,55.03)   # time for linearize
#t4 = (0,0,0,0)     # time for mpi.comm.bcast
#t5 = (0,0,0,0)    # time for convertToCSR
#t6 = (1.00,1.33,1.63,6.63)      # time for others

#bar_plot(N,np.array([t1,t2,t3,t4,t5,t6]))

#plt.ylabel('Time (s)')
#plt.xlabel('Number of processors')
#plt.title('Actual Time for Parallelization')


#--------------------- time plot for optimization -----------------
#---------------------(factorized with number of iterations)-----------------
f2 = plt.figure(2)


t1 = (2068.07,1431.50,947.06,855.85)
t2 = (1954.62,1358.66,906.67,828.30)
t3 = (600.84,350.39,189.80,133.21)
t4 = (0.,0.,0.,0.)  
t5 = (0.,0.,0.,0.)  
t6 = (136.97,103.54,105.49,156.23)
bar_plot(N,np.array([t1,t2,t3,t4,t5,t6]))

plt.ylabel('Time (s)')
plt.xlabel('Number of processors')
plt.title('Time per 100 iterations for Parallelization')



#---------------------- Speedup plot for optimization --------------------------------
f3 = plt.figure(3)


t1 = (2068.07,1431.50,947.06,855.85)
t2 = (1954.62,1358.66,906.67,828.30)
t3 = (600.84,350.39,189.80,133.21)
t4 = (0.,0.,0.,0.)  
t5 = (0.,0.,0.,0.)  
t6 = (136.97,103.54,105.49,156.23)
RealTime = np.array([t1,t2,t3,t4,t5,t6])
ScaleTime = np.empty(RealTime.shape)

T0 = np.sum(RealTime[:,0])
for j in np.arange(0,N):
    Tj = np.sum(RealTime[:,j])
    for i in np.arange(0,6):
        ScaleTime[i,j] = RealTime[i,0]/RealTime[i,j]*(RealTime[i,0]/T0)


bar_plot(N,ScaleTime)

plt.ylabel('Speedup')
plt.xlabel('Number of processors')
plt.title('Speedup for Parallelization')

#plt.show()


#---------------------- Time plot for each method --------------------------------
f4 = plt.figure(4)

# solve_nonlinear() 
p1 = plt.bar(num_pc-width/2, t1, width/2, color=colors[0])

# solve_linear() 
p2 = plt.bar(num_pc, t2, width/2, color=colors[1])

# linearize() 
p3 = plt.bar(num_pc+width/2, t3, width/2, color=colors[2])

plt.ylabel('Time (s)') 
plt.xlabel('Number of processors')
plt.title('Time for each method per 100 iteration')
plt.xticks(num_pc, ('1', '2', '4', '8'))
plt.legend((p1[0], p2[0], p3[0]), ('solve_nonlinear', 'solve_linear', 'linearize'))

plt.show()


