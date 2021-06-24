import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure()

num_el = np.array([1376, 5504, 22016, 88064, 352256])
N = len(num_el)
refine = np.arange(N)
w_exact = 0.0555556

w = np.zeros((6,N))
w[0,:] = [0.05509, 0.05547, 0.05558, 0.05561, 0.05561]
w[1,:] = [0.05359, 0.05502, 0.05543, 0.05553, 0.05556]
w[2,:] = [0.0492, 0.0535, 0.0550, 0.0554, 0.05552]
w[3,:] = [0.0222, 0.0292, 0.0375, 0.0465, 0.05247]
w[4,:] = [0.00123, 0.00383, 0.00977, 0.0183, 0.02668]
w[5,:] = [1.315e-5, 4.769e-5, 18.09e-5, 68.77e-5, 0.002467]


w_norm = w/w_exact

plt.grid(True, which="both")

plt.plot(refine, w_norm[0], 'bo-')
plt.plot(refine, w_norm[1], 'gx-')
plt.plot(refine, w_norm[2], 'rd-')
plt.plot(refine, w_norm[3], 'c*-')
plt.plot(refine, w_norm[4], 'mv-')
plt.plot(refine, w_norm[5], 'yh-')

plt.plot(np.ones(6), 'k--')
plt.xticks(np.arange(0, 6, step=1))
plt.yticks(np.arange(0, 1.6, step=0.2))
plt.title('Study of Locking Behavior For Triangular Elements')

plt.xlabel('Times of Mesh Refinement')
plt.ylabel('Normalized Tip Deflection')
plt.legend(['h/L = 0.4','h/L = 0.2','h/L = 1e-2','h/L = 1e-3','h/L = 1e-4','h/L = 1e-5', 'Exact solution'],loc='center right')
plt.show()
