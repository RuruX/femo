import numpy as np
import matplotlib.pyplot as plt

# ------------ GENERAL FUNCTION FORMS ------------
def linfun(x, a, b):
    func    = a * x + b 
    return func

def expfun2(x, a, b, c):
    func    = (a * np.exp((b * x) +c)) + 1
    return func

def cubicFun(x, a, b, c, d):
    func    = (a * x**3 + b * x**2 + c*x + d)
    return func

# ------------ LINEAR FUNCTIONS & COEFFICIENTS (y = mx + b) ------------
linearA = 1496.0690548058094
linearB = 2609.92718529671
def linearPortion(x):
    return linfun(x, linearA, linearB)

# ------------ CUBIC FUNCTIONS & COEFFICIENTS (y = ax^3  + bx^2  + cx + d) ------------
cubicA = -20102.05272597 # old -19989.35447329
cubicB = 53012.56254536  # old 52622.69081931
cubicC = -44163.58419442 # old -43721.52669411
cubicD = 15358.95134454 # old 15194.0665036
def cubicPortion(x):
    return cubicFun(x, cubicA, cubicB, cubicC, cubicD)

# ------------ EXPONENTIAL FUNCTIONS & COEFFICIENTS (y = ae^(bx + c)) ------------
expA = 19.125062472757403 #Old number 20.280410182691398
expB = -9.032027582050487 # Old Number-9.018264323503123
expC = 17.47567394067801 # old Number17.397642622485392
def expDecayPortion(x):
    return expfun2(x, expA, expB, expC)

# ------------ DOMAIN BOUNDS FOR EACH PIECEWISE FUNCTION ------------
# linearPortion x <= 1.004
# cubicPortion 1.004 < x <= 1.433
# expDecayPortion  x > 1.433 


# ------------ PLOT SHOWING FIT IN RELATION TO RAW DATA ------------
if __name__ == '__main__':

    file_name   = 'permeability.Magnetic alloy, silicon core iron C.tab'
    data        = np.genfromtxt(file_name,skip_header = 1, delimiter = '\t')
    H_data      = data[:,0]
    B_data      = data[:,1]

    B_cont = np.linspace(0., 3, 2000) #X input (B values)
    plt.plot(B_data, B_data/H_data / (4e-7 * np.pi), 'k*', markersize=10, label='Data')
    plt.plot(B_cont, linearPortion(B_cont),  'r', linewidth = 3, label='Linear')
    plt.plot(B_cont, cubicPortion(B_cont), 'b', linewidth = 3, label='Cubic')
    plt.plot(B_cont, expDecayPortion(B_cont), 'g', linewidth = 3, label='Exp Decay')
    plt.grid()
    plt.xlabel('B (T)')
    plt.ylabel('mu_r (Relative Permeability)')
    plt.xlim([0, 3])
    plt.ylim([0, 5000])
    plt.legend()

    plt.show()