import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure()

num_el = np.array([1376, 5504, 22016, 88064])
refine = np.arange(4)
w_exact = 0.027778

w = np.zeros((4,4))

w[0,:] = [0.0492, 0.0535, 0.0550, 0.0554]
w[1,:] = [0.0222, 0.0292, 0.0375, 0.0465]
w[2,:] = [0.00123, 0.00383, 0.00977, 0.0183]
w[3,:] = [1.315e-5, 4.769e-5, 18.09e-5, 68.77e-5]


w_norm = w/w_exact

plt.grid(True, which="both")

plt.plot(refine, w_norm[0], 'mo-')
plt.plot(refine, w_norm[1], 'bx-')
plt.plot(refine, w_norm[2], 'yD-')
plt.plot(refine, w_norm[3], 'g*-')

plt.plot(np.ones(5), 'k--')
plt.xticks(np.arange(0, 5, step=1))
plt.yticks(np.arange(0, 2, step=0.2))
plt.title('Study of Locking Behavior')

plt.xlabel('Times of Mesh Refinement')
plt.ylabel('Normalized Tip Deflection')
plt.legend(['h/L = 1e-2','h/L = 1e-3','h/L = 1e-4','h/L = 1e-5'],loc='center right')
plt.show()
