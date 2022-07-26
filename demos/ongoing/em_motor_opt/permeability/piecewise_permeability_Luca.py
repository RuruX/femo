import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ------------ GENERAL FUNCTION FORMS ------------
def linfun(x, a, b):
    func    = a * x + b 
    return func

def expfun(x, a, b, c):
    func    = (a * np.exp((b * x) +c)) + 1
    return func

def cubicfun(x, a, b, c, d):
    func    = (a * x**3 + b * x**2 + c*x + d)
    return func

def fit_linear():
    pass

def fit_exponential():
    pass

# ---------------------------------------------------------------

file_name   = 'permeability/Magnetic alloy, silicon core iron C.tab'
data        = np.genfromtxt(file_name,skip_header = 1, delimiter = '\t')
H_data      = data[:,0]
B_data      = data[:,1]
mu_data     = B_data / H_data / (4e-7*np.pi)

# LINEAR & EXPONENTIAL FITTING -------------------------------- 
linear_fit_ind_start    = 1
linear_fit_ind_end      = 3
exp_fit_ind_start       = 4

B_lin, mu_lin    = B_data[linear_fit_ind_start:linear_fit_ind_end], mu_data[linear_fit_ind_start:linear_fit_ind_end]
B_exp, mu_exp    = B_data[exp_fit_ind_start:], mu_data[exp_fit_ind_start:]

popt_lin, pconv_lin     = curve_fit(linfun, B_lin, mu_lin)
popt_exp, pconv_exp     = curve_fit(expfun, B_exp, mu_exp)

B_cont  = np.linspace(0., 3, 2000)

mu_lin  = linfun(B_cont, popt_lin[0], popt_lin[1])
mu_exp  = expfun(B_cont, popt_exp[0], popt_exp[1], popt_exp[2])

# CUBIC FUNCTION DETERMINATION -------------------------------- 
# original: x1 =  1.0, x2 = 1.433
x1      = 0.8
x2      = 1.4

# f: function value
# d: function derivative
mu_lin_f    = linfun(x1, popt_lin[0], popt_lin[1])
mu_lin_d    = popt_lin[0]
mu_exp_f    = expfun(x2, popt_exp[0], popt_exp[1], popt_exp[2])
mu_exp_d    = (expfun(x2, popt_exp[0], popt_exp[1], popt_exp[2]) - 1) * popt_exp[1]

A = np.array([[3*x1**2, 2*x1, 1, 0],
              [3*x2**2, 2*x2, 1, 0],
              [x1**3, x1**2, x1, 1],
              [x2**3, x2**2, x2, 1]])
b = np.array([mu_lin_d, mu_exp_d, mu_lin_f, mu_exp_f])
X_2 = np.linalg.solve(A,b)

mu_cubic    = cubicfun(B_cont, X_2[0], X_2[1], X_2[2], X_2[3])

# ------------ LINEAR FUNCTIONS & COEFFICIENTS (y = mx + b) ------------
# linearA = 1496.0690548058094
# linearB = 2609.92718529671

linearA = popt_lin[0]
linearB = popt_lin[1]

def linearPortion(x):
    return linfun(x, linearA, linearB)

# ------------ CUBIC FUNCTIONS & COEFFICIENTS (y = ax^3  + bx^2  + cx + d) ------------
# cubicA = -20102.05272597 # old -19989.35447329
# cubicB = 53012.56254536  # old 52622.69081931
# cubicC = -44163.58419442 # old -43721.52669411
# cubicD = 15358.95134454 # old 15194.0665036

cubicA = X_2[0]
cubicB = X_2[1]
cubicC = X_2[2]
cubicD = X_2[3]

def cubicPortion(x):
    return cubicfun(x, cubicA, cubicB, cubicC, cubicD)

# ------------ EXPONENTIAL FUNCTIONS & COEFFICIENTS (y = ae^(bx + c)) ------------
# expA = 19.125062472757403 #Old number 20.280410182691398
# expB = -9.032027582050487 # Old Number-9.018264323503123
# expC = 17.47567394067801 # old Number17.397642622485392

expA = popt_exp[0]
expB = popt_exp[1]
expC = popt_exp[2]
def expDecayPortion(x):
    return expfun(x, expA, expB, expC)

def extractexpDecayCoeff():
    return popt_exp

def extractCubicBounds():
    return x1, x2

# ------------ DOMAIN BOUNDS FOR EACH PIECEWISE FUNCTION ------------
# linearPortion x <= 1.004
# cubicPortion 1.004 < x <= 1.433
# expDecayPortion  x > 1.433 


# ------------ PLOT SHOWING FIT IN RELATION TO RAW DATA ------------
if __name__ == '__main__':

    plt.figure(100)
    plt.plot(B_data, mu_data, 'k*', markersize=10, label='Data')
    plt.plot(B_cont, mu_lin,  'r', linewidth = 3, label='Linear')
    plt.plot(B_cont, mu_exp, 'g', linewidth = 3, label='Exp Decay')
    plt.plot(B_cont, mu_cubic, 'b', linewidth = 3, label='Cubic')
    plt.plot(x1, mu_lin_f, 'm*', markersize=10)
    plt.plot(x2, mu_exp_f, 'm*', markersize=10)
    plt.grid()
    plt.xlabel('B (T)')
    plt.ylabel('mu_r (Relative Permeability)')
    plt.xlim([0, 3])
    plt.ylim([0, 5000])
    plt.legend()

    plt.show()
    exit()



    # file_name   = 'Magnetic alloy, silicon core iron C.tab'
    # data        = np.genfromtxt(file_name,skip_header = 1, delimiter = '\t')
    # H_data      = data[:,0]
    # B_data      = data[:,1]

    # B_cont = np.linspace(0., 3, 2000) #X input (B values)
    # plt.plot(B_data, B_data/H_data / (4e-7 * np.pi), 'k*', markersize=10, label='Data')
    # plt.plot(B_cont, linearPortion(B_cont),  'r', linewidth = 3, label='Linear')
    # plt.plot(B_cont, cubicPortion(B_cont), 'b', linewidth = 3, label='Cubic')
    # plt.plot(B_cont, expDecayPortion(B_cont), 'g', linewidth = 3, label='Exp Decay')
    # plt.grid()
    # plt.xlabel('B (T)')
    # plt.ylabel('mu_r (Relative Permeability)')
    # plt.xlim([0, 3])
    # plt.ylim([0, 5000])
    # plt.legend()

    # plt.show()