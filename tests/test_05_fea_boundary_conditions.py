def test_add_strong_bc_appends_with_and_without_space(fresh_import):
    fea_mod = fresh_import("femo.fea.fea_dolfinx")
    fea = fea_mod.FEA(mesh=object())

    fea_mod.dirichletbc = lambda *args: args

    fea.add_strong_bc(ubc="u0", locate_BC_list=[1, 2])
    fea.add_strong_bc(ubc="u1", locate_BC_list=[3], function_space="V")

    assert fea.bc[0] == ("u0", 1)
    assert fea.bc[1] == ("u0", 2)
    assert fea.bc[2] == ("u1", 3, "V")
