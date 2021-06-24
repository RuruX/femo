import numpy as np
import scipy.sparse

from openmdao.api import ExplicitComponent


def get_bspline_mtx(num_cp, num_pt, order=4):
    order = min(order, num_cp)

    knots = np.zeros(num_cp + order)
    knots[order-1:num_cp+1] = np.linspace(0, 1, num_cp - order + 2)
    knots[num_cp+1:] = 1.0

    t_vec = np.linspace(0, 1, num_pt)

    basis = np.zeros(order)
    arange = np.arange(order)
    data = np.zeros((num_pt, order))
    rows = np.zeros((num_pt, order), int)
    cols = np.zeros((num_pt, order), int)

    for ipt in range(num_pt):
        t = t_vec[ipt]

        i0 = -1
        for ind in range(order, num_cp+1):
            if (knots[ind-1] <= t) and (t < knots[ind]):
                i0 = ind - order
        if t == knots[-1]:
            i0 = num_cp - order

        basis[:] = 0.
        basis[-1] = 1.

        for i in range(2, order+1):
            l = i - 1
            j1 = order - l
            j2 = order
            n = i0 + j1
            if knots[n+l] != knots[n]:
                basis[j1-1] = (knots[n+l] - t) / \
                              (knots[n+l] - knots[n]) * basis[j1]
            else:
                basis[j1-1] = 0.
            for j in range(j1+1, j2):
                n = i0 + j
                if knots[n+l-1] != knots[n-1]:
                    basis[j-1] = (t - knots[n-1]) / \
                                (knots[n+l-1] - knots[n-1]) * basis[j-1]
                else:
                    basis[j-1] = 0.
                if knots[n+l] != knots[n]:
                    basis[j-1] += (knots[n+l] - t) / \
                                  (knots[n+l] - knots[n]) * basis[j]
            n = i0 + j2
            if knots[n+l-1] != knots[n-1]:
                basis[j2-1] = (t - knots[n-1]) / \
                              (knots[n+l-1] - knots[n-1]) * basis[j2-1]
            else:
                basis[j2-1] = 0.

        data[ipt, :] = basis
        rows[ipt, :] = ipt
        cols[ipt, :] = i0 + arange

    data, rows, cols = data.flatten(), rows.flatten(), cols.flatten()

    return scipy.sparse.csr_matrix(
        (data, (rows, cols)), 
        shape=(num_pt, num_cp),
    )


class BsplineComp(ExplicitComponent):
    """
    General function to translate from control points to actual points
    using a b-spline representation.
    """

    def initialize(self):
        self.options.declare('num_pt', types=int)
        self.options.declare('num_cp', types=int)
        self.options.declare('jac')
        self.options.declare('in_name', types=str)
        self.options.declare('out_name', types=str)

    def setup(self):
        num_pt = self.options['num_pt']
        num_cp = self.options['num_cp']
        jac = self.options['jac']
        in_name = self.options['in_name']
        out_name = self.options['out_name']

        self.add_input(in_name, shape=num_cp)
        self.add_output(out_name, shape=num_pt)

        jac = self.options['jac'].tocoo()

        self.declare_partials(out_name, in_name, val=jac.data, rows=jac.row, cols=jac.col)

    def compute(self, inputs, outputs):
        num_pt = self.options['num_pt']
        num_cp = self.options['num_cp']
        jac = self.options['jac']
        in_name = self.options['in_name']
        out_name = self.options['out_name']

        outputs[out_name] = jac * inputs[in_name]
        
if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp
    num_cp = 5
    num_pt = 100
    mtx = get_bspline_mtx(num_cp,num_pt)

    prob = Problem()
#    cp = np.array([7.596166E-03, -4.011293E-02, 1.043192E-02,  -4.514498E-03, 1.235953E-03, 
#                    -6.648183E-04, 4.046970E-03, -9.331267E-03, 2.356887E-02, 2.064098E-02])
#    cp = np.array([0.01033369, -0.02012367, -0.00803041,  0.03023046, -0.00605389, -0.00911292])
#    cp = np.array([3.407115E-03, -4.441580E-02, 1.157083E-02, 1.863296E-02, 2.171117E-02])

#    cp = np.array([2.297956E-02,-5.985355E-02,1.936941E-02,-1.298938E-02,3.154014E-02,1.949934E-02])
    cp = np.array([0.00350663,-0.04317053,0.00778344,0.02093448,0.02144338])
    comp = IndepVarComp()
    comp.add_output('control_points', val=cp, shape=num_cp)
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    comp = BsplineComp(
                       num_pt=num_pt,
                       num_cp=num_cp,
                       jac=mtx,
                       in_name='control_points',
                       out_name='points',
                       )

    prob.model.add_subsystem('bspline_comp', comp, promotes=['*'])
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print(prob['control_points'])
    print(prob['points'])
    print('check_partials:')
    prob.check_partials(compact_print=True)

    from matplotlib import pyplot as plt
    t=np.linspace(0,1,num_pt)
    plt.plot(np.linspace(0,1,num_cp),cp,'bo')
    plt.plot(t,prob['points'],'ro')
    plt.show()
