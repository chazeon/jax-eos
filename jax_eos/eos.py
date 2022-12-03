from jax import grad, jit, vmap
from scipy.optimize import root
from functools import partial, cached_property


class FiniteStrainEVEquationOfState:

    def __init__(self, volume, energy, order=3):
        self.v_ref = volume[0]
        x = self.eulerian_strain(energy, self.v_ref)
        self.p = jnp.polyfit(x, energy, deg=order)

    @staticmethod
    def eulerian_strain(volume, v_ref):
        return (v_ref / volume) ** (2/3) - 1

    @partial(jit, static_argnums=(0,))
    def free_energy(self, volume):
        x = self.eulerian_strain(volume, self.v_ref)
        energy = jnp.polyval(self.p, x)
        return energy

    @partial(jit, static_argnums=(0,))
    def _pressure(self, volume):
        return -grad(self.free_energy)(volume)

    def pressure(self, volume):
        return vmap(self._pressure)(volume)

    @partial(jit, static_argnums=(0,))
    def _bulk_modulus(self, volume):
        return -grad(self._pressure)(volume) * volume

    def bulk_modulus(self, volume):
        return vmap(self._bulk_modulus)(volume)

    @cached_property
    def v0(self):
        return root(lambda v: self.pressure(v), self.v_ref).x[0]

    @property
    def k0(self):
        return self._bulk_modulus(self.v0)

    @property
    def kp0(self):
        return grad(self._bulk_modulus)(self.v0) / grad(self._pressure)(self.v0)
