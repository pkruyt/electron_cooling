# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import xtrack as xt
import xobjects as xo
import xpart as xp

import ducktrack as dtk




for ctx in xo.context.get_test_contexts():
    print(f"Test {ctx.__class__}")

I=2.4
L = 1.5 # m cooler length
r_beam=25*1e-3


#ne = 8.27e13 #nocolo's value
ne=83435412163766.98
T_perp = 0.1 # <E> [eV] = kb*T
T_l =  0.001 # <E> [eV]
B = 0.060 # T for LEIR
Z=54
beta=0.305
gamma = 1.050
mass0=938.27208816*1e6
c=299792458.0
p0c=p = mass0*beta*gamma

dtk_particle = dtk.TestParticles(
        
        p0c=p0c,
        x=np.random.normal(0, 10*1e-3, 10000),
        px=np.random.normal(0, 1*1e-3, 10000),
        y=0,
        py=0,
        delta=0,
        zeta=0)




dtk_cooler = dtk.elements.ElectronCooler(I=I,L=L,r_beam=r_beam,ne=ne,
                                         T_perp=T_perp,T_l=T_l,
                                         B=B,Z=Z,B_ratio=1e-10)



dtk_cooler.track(dtk_particle)

Fx = dtk_cooler.force(dtk_particle)

import matplotlib.pyplot as plt

plt.scatter(dtk_particle.px,(Fx),label='$B_⟂$/$B_∥$')
plt.xlim([-0.002,0.002])
plt.title('cooling force vs px')
plt.ylabel('cooling force [eV/m]')
plt.xlabel('transverse momentum px [rad]')
plt.legend()

print(dtk_particle.x)
       


























test_source = r"""
/*gpufun*/
void test_function(TestElementData el,
                LocalParticle* part0,
                /*gpuglmem*/ double* b){

    double const a = TestElementData_get_a(el);

    //start_per_particle_block (part0->part)

        const int64_t ipart = part->ipart;
        double const val = b[ipart];

        LocalParticle_add_to_x(part, val + a);

    //end_per_particle_block
}

/*gpufun*/
void TestElement_track_local_particle(TestElementData el,
                LocalParticle* part0){

    double const a = TestElementData_get_a(el);

    //start_per_particle_block (part0->part)

        LocalParticle_set_x(part, a);

    //end_per_particle_block
}

"""


def test_per_particle_kernel():

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        class TestElement(xt.BeamElement):
            _xofields={
                'a': xo.Float64
            }

            _extra_c_sources=[test_source]

            _per_particle_kernels={
                'test_kernel': xo.Kernel(
                    c_name='test_function',
                    args=[
                        xo.Arg(xo.Float64, pointer=True, name='b')
                    ]),}

        el = TestElement(_context=context, a=10)

        # p = xp.Particles(p0c=1e9, x=[1,2,3], _context=context)
        # el.track(p)
        # p.move(_context=xo.ContextCpu())
        # assert np.all(p.x == [10,10,10])

        p = xp.Particles(p0c=1e9, x=[1,2,3], _context=context)
        b = p.x*0.5
        el.test_kernel(p, b=b)
        p.move(_context=xo.ContextCpu())
        assert np.all(p.x == np.array([11.5, 13, 14.5]))


