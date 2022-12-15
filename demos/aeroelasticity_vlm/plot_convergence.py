from matplotlib import pyplot as plt
import numpy as np

tip_disp_5 = np.loadtxt("test_5_1/tip_disp_5.out")
tip_disp_20 = np.loadtxt("test_20/tip_disp_20.out")
tip_disp_40 = np.loadtxt("test_40/tip_disp_40.out")
tip_disp_100 = np.loadtxt("test_100/tip_disp_100.out")
tip_disp_300 = np.loadtxt("test_300/tip_disp_300.out")

T = 0.5
t_5 = np.linspace(0., T, 5+1)[1:]
t_20 = np.linspace(0., T, 20+1)[1:]
t_40 = np.linspace(0., T, 40+1)[1:]
t_100 = np.linspace(0., T, 100+1)[1:]
t_300 = np.linspace(0., T, 300+1)[1:]

fig, ax = plt.subplots()
ax.plot(t_5, tip_disp_5, 'r-o', label="Nsteps=5")
ax.plot(t_20, tip_disp_20, 'b-*', label="Nsteps=20")
ax.plot(t_40, tip_disp_40, 'y-+', label="Nsteps=40")
ax.plot(t_100, tip_disp_100, 'g-', label="Nsteps=100")
ax.plot(t_300, tip_disp_300, 'm-', label="Nsteps=300")
ax.set_xlabel("time (s)")
ax.set_ylabel("tip displacement (m)")
ax.legend(loc="best")
plt.show()
fig.savefig("convergence_time_steps.pdf", dpi=150)
