import importlib
import sys
import types
import numpy as np
import pytest


class _ParameterStore(dict):
    def declare(self, name, default=None, types=None):
        if name not in self:
            self[name] = default


class _BaseModel:
    def __init__(self, **kwargs):
        self.parameters = _ParameterStore(kwargs)
        self._added = []

    def add(self, model, name=None):
        self._added.append((name, model))

    def declare_variable(self, name, shape=None, val=0.0):
        if isinstance(val, np.ndarray):
            return val
        if shape is None:
            return np.array(val)
        return np.full(shape, val)

    def register_output(self, name, value):
        return value

    def print_var(self, value):
        return None


class _BaseOperation:
    def __init__(self, **kwargs):
        self.parameters = _ParameterStore(kwargs)
        self._inputs = {}
        self._outputs = {}

    def add_input(self, name, shape=None):
        self._inputs[name] = shape

    def add_output(self, name, shape=None):
        self._outputs[name] = shape

    def declare_derivatives(self, of, wrt):
        return None


@pytest.fixture(scope="session", autouse=True)
def dependency_stubs():
    # csdl stubs
    csdl = types.ModuleType("csdl")
    csdl.Model = _BaseModel
    csdl.CustomExplicitOperation = _BaseOperation
    csdl.CustomImplicitOperation = _BaseOperation
    csdl.custom = lambda *args, op=None: np.zeros((1,))
    sys.modules["csdl"] = csdl

    # matplotlib stub
    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.figure = lambda *a, **k: None
    pyplot.plot = lambda *a, **k: None
    pyplot.show = lambda *a, **k: None
    matplotlib.pyplot = pyplot
    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.pyplot"] = pyplot

    # mpi4py stubs
    mpi4py = types.ModuleType("mpi4py")
    mpi4py.MPI = types.SimpleNamespace(
        COMM_WORLD=types.SimpleNamespace(allreduce=lambda x, op=None: x),
        SUM="SUM",
    )
    sys.modules["mpi4py"] = mpi4py

    # petsc4py stubs
    petsc4py = types.ModuleType("petsc4py")

    class _DummyKSP:
        def create(self, comm=None):
            return self

        def setOperators(self, *args, **kwargs):
            return None

        def setType(self, *args, **kwargs):
            return None

        def getPC(self):
            return types.SimpleNamespace(
                setType=lambda *a, **k: None,
                setFactorSolverType=lambda *a, **k: None,
                getASMSubKSP=lambda: [types.SimpleNamespace(
                    setType=lambda *a, **k: None,
                    getPC=lambda: types.SimpleNamespace(setType=lambda *a, **k: None),
                    setTolerances=lambda *a, **k: None,
                )],
            )

        def setFromOptions(self):
            return None

        def setUp(self):
            return None

        def solve(self, b, x):
            return None

        def setConvergenceHistory(self):
            return None

        def getConvergenceHistory(self):
            return []

        def setTolerances(self, *args, **kwargs):
            return None

    class _DummySNES:
        def create(self):
            return self

        def setTolerances(self, **kwargs):
            return None

        def getKSP(self):
            return _DummyKSP()

        def setFunction(self, *args, **kwargs):
            return None

        def setJacobian(self, *args, **kwargs):
            return None

        def setFromOptions(self):
            return None

        def solve(self, *args, **kwargs):
            return None

        def getConvergedReason(self):
            return 1

    petsc4py.PETSc = types.SimpleNamespace(
        InsertMode=types.SimpleNamespace(ADD_VALUES=1, INSERT=2),
        ScatterMode=types.SimpleNamespace(REVERSE=1, FORWARD=2),
        KSP=_DummyKSP,
        SNES=_DummySNES,
        Options=lambda: {},
        Mat=lambda comm=None: object(),
        Vec=lambda: types.SimpleNamespace(create=lambda: types.SimpleNamespace(setSizes=lambda *a: None, setUp=lambda: None, assemble=lambda: None, getArray=lambda: np.array([]))),
    )
    sys.modules["petsc4py"] = petsc4py

    # scipy stubs (only if unavailable)
    if "scipy" not in sys.modules:
        scipy = types.ModuleType("scipy")
        sparse = types.ModuleType("scipy.sparse")
        spatial = types.ModuleType("scipy.spatial")

        class _FakeKDTree:
            def __init__(self, coords):
                self.coords = np.asarray(coords)

            def query(self, nodes):
                nodes = np.asarray(nodes)
                dists = []
                idxs = []
                for node in nodes:
                    diff = self.coords - node
                    ds = np.sqrt(np.sum(diff * diff, axis=1))
                    i = int(np.argmin(ds))
                    dists.append(ds[i])
                    idxs.append(i)
                return np.array(dists), np.array(idxs)

        sparse.csr_matrix = lambda *args, **kwargs: types.SimpleNamespace(tocoo=lambda: object())
        spatial.KDTree = _FakeKDTree
        scipy.sparse = sparse
        scipy.spatial = spatial
        sys.modules["scipy"] = scipy
        sys.modules["scipy.sparse"] = sparse
        sys.modules["scipy.spatial"] = spatial

    # ufl stubs
    ufl = types.ModuleType("ufl")
    _f = lambda *a, **k: (a, k)
    for name in [
        "Identity", "dot", "derivative", "TestFunction", "TrialFunction", "inner", "ds", "dS", "dx",
        "grad", "inv", "as_vector", "sqrt", "conditional", "lt", "det", "Measure", "exp", "tr",
        "CellDiameter", "div", "SpatialCoordinate", "FacetNormal",
    ]:
        setattr(ufl, name, _f)
    sys.modules["ufl"] = ufl

    # dolfinx stubs
    dolfinx = types.ModuleType("dolfinx")
    dolfinx.log = types.SimpleNamespace(LogLevel=types.SimpleNamespace(INFO=1), set_log_level=lambda lvl: None)

    io = types.ModuleType("dolfinx.io")

    class _XDMFFile:
        def __init__(self, *args, **kwargs):
            self.path = args[1] if len(args) > 1 else ""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read_mesh(self, name="Grid"):
            return types.SimpleNamespace(topology=types.SimpleNamespace(dim=2, create_connectivity=lambda *a, **k: None))

        def read_meshtags(self, mesh, name="Grid"):
            return {"name": name, "path": self.path}

        def write_mesh(self, mesh):
            return None

        def write_function(self, func, step):
            return None

    io.XDMFFile = _XDMFFile

    mesh = types.ModuleType("dolfinx.mesh")
    mesh.create_unit_square = lambda *a, **k: object()
    mesh.create_rectangle = lambda *a, **k: object()
    mesh.create_interval = lambda *a, **k: object()
    mesh.locate_entities_boundary = lambda *a, **k: np.array([], dtype=np.int32)
    mesh.locate_entities = lambda *a, **k: np.array([], dtype=np.int32)
    mesh.meshtags = lambda *a, **k: object()

    cpp = types.ModuleType("dolfinx.cpp")
    cpp_mesh = types.ModuleType("dolfinx.cpp.mesh")
    cpp_mesh.CellType = types.SimpleNamespace(quadrilateral="quadrilateral")
    cpp_mesh.h = lambda mesh, tdim, cells: np.array([])
    cpp.mesh = cpp_mesh

    fem = types.ModuleType("dolfinx.fem")

    class _DummyVector:
        def __init__(self):
            self.array = np.array([], dtype=float)

        def getArray(self):
            return self.array

        def set(self, value):
            self.array[:] = value

        def assemble(self):
            return None

        def ghostUpdate(self, *args, **kwargs):
            return None

        def pointwiseDivide(self, b, a):
            return None

        def localForm(self):
            class _Ctx:
                def __enter__(self_non):
                    return types.SimpleNamespace(set=lambda v: None)

                def __exit__(self_non, exc_type, exc, tb):
                    return False
            return _Ctx()

        def copy(self, other):
            return None

    class _DummyFunction:
        def __init__(self, function_space=None):
            self.function_space = function_space or types.SimpleNamespace(num_sub_spaces=1)
            self.vector = _DummyVector()
            self.x = types.SimpleNamespace(array=np.array([], dtype=float))

        def rename(self, *args, **kwargs):
            return None

        def interpolate(self, *args, **kwargs):
            return None

        def split(self):
            return self, None

        def sub(self, idx):
            return types.SimpleNamespace(collapse=lambda: self)

    fem.form = lambda x: x
    fem.assemble_scalar = lambda x: 0.0
    fem.Function = _DummyFunction
    fem.FunctionSpace = lambda mesh, desc: types.SimpleNamespace(mesh=mesh)
    fem.VectorFunctionSpace = lambda *a, **k: types.SimpleNamespace(mesh=a[0] if a else None)
    fem.dirichletbc = lambda *a, **k: (a, k)
    fem.locate_dofs_geometrical = lambda *a, **k: np.array([], dtype=np.int32)
    fem.locate_dofs_topological = lambda *a, **k: np.array([], dtype=np.int32)
    fem.Constant = lambda mesh, v: v
    fem.set_bc = lambda *a, **k: None

    fem_petsc = types.ModuleType("dolfinx.fem.petsc")
    fem_petsc.assemble_vector = lambda *a, **k: types.SimpleNamespace(array=np.array([]), ghostUpdate=lambda *aa, **kk: None)
    fem_petsc.assemble_matrix = lambda *a, **k: types.SimpleNamespace(
        assemble=lambda: None,
        copy=lambda: types.SimpleNamespace(assemble=lambda: None, convert=lambda fmt: types.SimpleNamespace(getDenseArray=lambda: np.array([[]]))),
        transpose=lambda *aa, **kk: types.SimpleNamespace(),
        getValuesCSR=lambda: (np.array([0]), np.array([0]), np.array([0.0])),
        size=(0, 0),
        getSizes=lambda: (0, 0),
        getComm=lambda: None,
        multTranspose=lambda *aa, **kk: None,
    )
    fem_petsc.NonlinearProblem = lambda *a, **k: object()
    fem_petsc.apply_lifting = lambda *a, **k: None
    fem_petsc.set_bc = lambda *a, **k: None
    fem_petsc.create_matrix = lambda *a, **k: object()
    fem_petsc._assemble_matrix_mat = lambda *a, **k: object()

    nls = types.ModuleType("dolfinx.nls")
    nls_petsc = types.ModuleType("dolfinx.nls.petsc")
    nls_petsc.NewtonSolver = lambda *a, **k: types.SimpleNamespace(solve=lambda *aa, **kk: None)

    la = types.ModuleType("dolfinx.la")
    la.create_petsc_vector = lambda *a, **k: types.SimpleNamespace(
        localForm=lambda: types.SimpleNamespace(__enter__=lambda self: types.SimpleNamespace(set=lambda v: None), __exit__=lambda self, exc_type, exc, tb: False)
    )

    dolfinx.io = io
    dolfinx.mesh = mesh
    dolfinx.cpp = cpp
    dolfinx.fem = fem
    dolfinx.nls = nls
    dolfinx.la = la

    sys.modules["dolfinx"] = dolfinx
    sys.modules["dolfinx.io"] = io
    sys.modules["dolfinx.mesh"] = mesh
    sys.modules["dolfinx.cpp"] = cpp
    sys.modules["dolfinx.cpp.mesh"] = cpp_mesh
    sys.modules["dolfinx.fem"] = fem
    sys.modules["dolfinx.fem.petsc"] = fem_petsc
    sys.modules["dolfinx.nls"] = nls
    sys.modules["dolfinx.nls.petsc"] = nls_petsc
    sys.modules["dolfinx.la"] = la

    yield


@pytest.fixture
def fresh_import():
    def _fresh(module_name):
        if module_name in sys.modules:
            del sys.modules[module_name]
        return importlib.import_module(module_name)

    return _fresh
