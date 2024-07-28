import torch
import torchdyn.numerics.solvers.ode as solvers


class Euler(solvers.Euler):
    pass


class Midpoint(solvers.Midpoint):
    pass


class RungeKutta4(solvers.RungeKutta4):
    def step(self, **kwargs):
        self.sync_device_dtype(kwargs["x"], kwargs["t"])
        return super().step(**kwargs)


class ImplicitEuler(solvers.ImplicitEuler):
    def step(self, **kwargs):
        self.sync_device_dtype(kwargs["x"], kwargs["t"])
        return super().step(**kwargs)


class AsynchronousLeapfrog(solvers.AsynchronousLeapfrog):
    def step(self, **kwargs):
        self.sync_device_dtype(kwargs["x"], kwargs["t"])
        x = kwargs.pop("x")
        return super().step(xv=x, **kwargs)


SOLVER_DICT = {'euler': Euler, 'midpoint': Midpoint,
               'rk4': RungeKutta4, 'rk-4': RungeKutta4, 'RungeKutta4': RungeKutta4,
               # 'dopri5': DormandPrince45, 'DormandPrince45': DormandPrince45, 'DormandPrince5': DormandPrince45,
               # 'tsit5': Tsitouras45, 'Tsitouras45': Tsitouras45, 'Tsitouras5': Tsitouras45,
               'ieuler': ImplicitEuler, 'implicit_euler': ImplicitEuler,
               # 'alf': AsynchronousLeapfrog, 'AsynchronousLeapfrog': AsynchronousLeapfrog
}


def str_to_solver(solver_name, dtype=torch.float32):
    "Transforms string specifying desired solver into an instance of the Solver class."
    solver = SOLVER_DICT[solver_name]
    return solver(dtype)
