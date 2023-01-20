from matplotlib import pyplot as plt
import numpy as np

tip_disp_25 = np.loadtxt("test_25_1/tip_disp_25.out")
# tip_disp_20 = np.loadtxt("test_20/tip_disp_20.out")
# tip_disp_40 = np.loadtxt("test_40/tip_disp_40.out")
# tip_disp_100 = np.loadtxt("test_100/tip_disp_100.out")
# tip_disp_300 = np.loadtxt("test_300/tip_disp_300.out")
tip_disp_50 = np.loadtxt("test_50_1/tip_disp_50.out")
tip_disp_100 = np.loadtxt("test_100_1/tip_disp_100.out")
tip_disp_200 = np.loadtxt("test_200_1/tip_disp_200.out")
T = 0.5
t_25 = np.linspace(0., T, 25+1)[1:]
# t_20 = np.linspace(0., T, 20+1)[1:]
# t_40 = np.linspace(0., T, 40+1)[1:]
# t_100 = np.linspace(0., T, 100+1)[1:]
# t_300 = np.linspace(0., T, 300+1)[1:]
t_50 = np.linspace(0., T, 50+1)[1:]
t_100 = np.linspace(0., T, 100+1)[1:]
t_200 = np.linspace(0., T, 200+1)[1:]

fig, ax = plt.subplots()
ax.plot(t_25, tip_disp_25, color='steelblue', label="Nsteps=25")
ax.plot(t_50, tip_disp_50, 'm-',label="Nsteps=50")
ax.plot(t_100, tip_disp_100, color='darkorange', label="Nsteps=100")
# ax.plot(t_200, tip_disp_200, color='darkslategray', label="Nsteps=200")
ax.set_xlabel("time (s)")
ax.set_ylabel("tip displacement (m)")
ax.legend(loc="best")
plt.show()
fig.savefig("convergence_time_steps.png", dpi=150)


'''
  -> VLMModel -> A2SCoupling -> ShellModel -> S2ACoupling -->
 |                                                          |
  -------------------cyclic relationship---------------------
'''

"""
--------------------------------------------------
-------- eVTOL_wing_half_tri_107695_136686.xdmf ---------
--------------------------------------------------
Tip deflection: 0.036563181704068366
  Number of elements = 136686
  Number of vertices = 66974
  Number of total dofs =  1013097

--------------------------------------------------
"""
