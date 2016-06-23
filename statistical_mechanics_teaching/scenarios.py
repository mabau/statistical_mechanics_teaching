__author__ = "Martin Bauer <martin.bauer@fau.de>"
__copyright__ = "Copyright 2016, Martin Bauer"
__license__ = "GPL"
__version__ = "3"

import numpy as np
import matplotlib.pyplot as plt

from .simulation import ParticleSimulation
from .mpl_display import *


plt.style.use('ggplot')


def single_species_setup(domain_size=[20, 20], vel=[0.5, 0.5], dt=0.02, particle_region=[[0.1, 0.9], [0.1, 0.9]],
                         show_velocity_histogram=False, show_speed_histogram=False, density=0.1, display_interval=10,
                         window=500, frames=300):
    """
    Runs a Lennard-Jones molecular dynamics simulation of one particle type using lammps.
    Optionally plots 2D velocity distribution and speed distribution
    :param domain_size: size of the domain as list [x_width, y_width]
    :param vel: list of [x_vel,y_vel] to set the same velocity for all atoms, or a single number to set thermal velocity
    :param dt: time step length
    :param particle_region: nested list defining the relative part of the domain where particles should be initialized
                            [[x_min, x_max], [y_min,y_max]]
    :param show_velocity_histogram: enable to plot a 2D histogram of velocity distribution
    :param show_speed_histogram: enable to plot a 1D histogram of speed distribution
    :param density: packing density of atoms (should be smaller than 1)
    :param display_interval: how many time steps to run before a
                             new histogram sample is taken and particles are displayed
    :param window: number of frames to average over for histograms
    :param frames: length of animation in frames
    :return: matplotlib animation
                - save this to a variable and call plt.show()
                - can also be used to create video or show in a jupyter notebook
    """
    sim = ParticleSimulation(np.array(domain_size), 1, lj_eps=0.4, boundaries="f f", verbose=False)
    sim.activate_reflecting_walls()
    sim.set_atom_type_parameters(atom_type=0, mass=1.0, radius=0.4)
    sim.add_atoms('atoms', 0, np.array(particle_region),  lattice_scale=density, region_is_relative=True)
    sim.set_delta_t(dt)
    sim.set_velocity('atoms', vel)

    if not show_velocity_histogram and not show_speed_histogram:
        grid = (1, 1)
    elif show_velocity_histogram and show_speed_histogram:
        grid = (2, 3)
    else:
        grid = (1, 2)

    sim_plot = MolecularDynamicsAnimation(sim, plt.figure(), grid, display_interval)

    if not show_velocity_histogram and not show_speed_histogram:
        sim_plot.add_particle_plot()
    elif show_velocity_histogram and not show_speed_histogram:
        sim_plot.add_particle_plot((0, 1, 0, 1))
        sim_plot.add_velocity_histogram((0, 1, 1, 2), window=window, show_entropy=False)
    elif not show_velocity_histogram and show_speed_histogram:
        sim_plot.add_particle_plot((0, 1, 0, 1))
        sim_plot.add_speed_histogram((0, 1, 1, 2), window=window, )
    else:
        sim_plot.add_particle_plot((0, 2, 0, 2))
        sim_plot.add_velocity_histogram((0, 1, 2, 3), window=window, show_entropy=False)
        sim_plot.add_speed_histogram((1, 2, 2, 3), window=window, )

    return sim_plot.get_animation_object(interval=display_interval, frames=frames)


def two_species_setup(domain_size=[20, 20], vel1=[-2, 1.0], vel2=[0.0, 0.0],
                      mass1=1.0, mass2=8.0, dt=0.02, display_interval=20, frames=300):
    """
    Runs a Lennard-Jones molecular dynamics simulation of one particle type using lammps.
    Optionally plots 2D velocity distribution and speed distribution
    :param domain_size: size of the domain as list [x_width, y_width]
    :param vel1: initial velocity of particle type 1
    :param vel2: initial velocity of particle type 1
    :param mass1: mass for particle type 1
    :param mass2: mass for particle type 2
    :param dt: time step length
    :param display_interval: how many time steps to run before a
                             new histogram sample is taken and particles are displayed
    :param frames: length of animation in frames
    :return: matplotlib animation
    """
    sim = ParticleSimulation(np.array(domain_size), 2, lj_eps=0.4, boundaries="f f", verbose=False)
    sim.activate_reflecting_walls()
    sim.set_atom_type_parameters(atom_type=0, mass=mass1, radius=0.4)
    sim.set_atom_type_parameters(atom_type=1, mass=mass2, radius=0.4)
    sim.add_atoms('atoms0', 0, np.array([[0.1, 0.5], [0.10, 0.9]]),  lattice_scale=0.4, region_is_relative=True)
    sim.add_atoms('atoms1', 1, np.array([[0.5, 0.9], [0.10, 0.9]]),  lattice_scale=0.4, region_is_relative=True)
    sim.set_velocity('atoms0', vel1)
    sim.set_velocity('atoms1', vel2)
    sim.set_delta_t(dt)

    sim_plot = MolecularDynamicsAnimation(sim, plt.figure(), (2, 3), display_interval)
    sim_plot.add_particle_plot((0, 2, 0, 2))
    sim_plot.add_velocity_histogram((0, 1, 2, 3))
    sim_plot.add_speed_histogram((1, 2, 2, 3), one_speed_hist_for_each_type=True)

    return sim_plot.get_animation_object(interval=display_interval, frames=frames)


def moving_wall_setup(domain_size=[30, 50], wallvelocity_callback=lambda t: -0.02, keep_temperature_fixed=True,
                      density=0.3, dt=0.01, vel=1, display_interval=100, equilibration_steps=500, frames=300):
    """
    Simulation with a single moving wall to experiment with volume changes
    :param domain_size: size of the domain as list [x_width, y_width]
    :param wallvelocity_callback: a function that gets number of frame and should return velocity of wall (as scalar)
    :param keep_temperature_fixed: if True, particle velocities are rescaled to keep temperature fixed
    :param density: packing density of atoms (should be smaller than 1)
    :param dt: time step length
    :param vel: initial velocity of particles
    :param display_interval: how many time steps to run before a
                             new histogram sample is taken and particles are displayed
    :param equilibration_steps: number of timesteps to run before sampling and display starts
    :param frames: length of animation in frames
    :return: matplotlib animation
    """
    sim = ParticleSimulation(np.array(domain_size), 2, lj_eps=0.4, boundaries="f f", verbose=False)
    sim.activate_reflecting_walls()
    sim.set_atom_type_parameters(atom_type=0, mass=1.0, radius=0.4)

    sim.add_atoms('fluid', 0, np.array([[0.1, 0.9], [0.1,  0.6]]),   region_is_relative=True, lattice_scale=density)
    sim.add_atoms('wall',  1, np.array([[0.0, 1.0], [0.65, 0.68]]),  region_is_relative=True, lattice_scale=0.6)

    force_calculation_params = {'interval': display_interval, 'start': equilibration_steps, 'window': 100}
    force_on_wall_id = sim.set_force('constantWallForce', 'wall', [0.0, 0.0], force_calculation_params)

    sim.run_lammps("fix f_move_wall g_wall move linear NULL %f NULL" % (-0.02))

    if keep_temperature_fixed:
        sim.fix_temperature(vel, 'fluid')

    sim.set_delta_t(dt)
    sim.set_velocity_thermal('fluid', vel)

    def get_force():
        return -sim.force_on_group(force_on_wall_id)[1]

    def set_wall_velocity_cb(i):
        cmd = """
        unfix f_move_wall
        fix f_move_wall g_wall move linear NULL %f NULL
        """
        sim.run_lammps(cmd % (wallvelocity_callback(i),))

    sim_plot = MolecularDynamicsAnimation(sim, plt.figure(), (2, 4), display_interval)
    sim_plot.add_frame_callback(set_wall_velocity_cb)
    sim_plot.add_particle_plot((0, 2, 0, 2))
    sim_plot.add_xy_histogram((0, 1, 2, 3), particle_type=0)
    sim_plot.add_velocity_histogram((1, 2, 2, 3), particle_type=0)
    sim_plot.add_speed_histogram((0, 1, 3, 4), one_speed_hist_for_each_type=(0,))
    scalar_plot_ax = sim_plot.add_scalar_over_time_plot((1, 2, 3, 4), get_force, num_samples_kept=1000)
    scalar_plot_ax.set_title("Force on wall")
    scalar_plot_ax.set_xlabel("t")
    scalar_plot_ax.set_ylabel("$-F_y$")

    sim.run(equilibration_steps)

    return sim_plot.get_animation_object(interval=display_interval, frames=frames)


def viscosity_setup(domain_size=[50, 100], temperature=6.0, wall_velocity=1.0, dt=0.008,
                    display_interval=1000, equilibration_steps=50000, frames=300):
    """
    Moving wall (Couette) scenario with display of velocity profile and force on wall to `measure` viscosity
    :param domain_size: size of the domain as list [x_width, y_width]
    :param temperature: initial temperature of the simulation, while simulation is running this temperature is
                        enforced, since energy is added by moving wall
    :param wall_velocity: velocity of moving wall
    :param dt: time step length
    :param display_interval: how many time steps to run before a
                             new histogram samples iare taken and particles are displayed
    :param equilibration_steps: number of timesteps to run before sampling and display starts
    :param frames: length of animation in frames
    :return: matplotlib animation
    """
    sim = ParticleSimulation(np.array(domain_size), 3, lj_eps=1, verbose=False, boundaries="f p")

    sim.set_atom_type_parameters(atom_type=0, mass=1.0, radius=0.4)
    sim.set_atom_type_parameters(atom_type=1, mass=1.0, radius=0.4)
    sim.set_atom_type_parameters(atom_type=2, mass=1.0, radius=0.4)

    sim.add_atoms('fluid',      0, np.array([[0.1, 0.9], [0.05, 0.95]]), region_is_relative=True, lattice_scale=0.2)
    sim.add_atoms('left_wall',  1, np.array([[0.02, 0.05], [0.00, 1]]),  region_is_relative=True, lattice_scale=0.6)
    sim.add_atoms('right_wall', 2, np.array([[0.95, 0.98], [0.00, 1]]),  region_is_relative=True, lattice_scale=0.6)

    force_calculation_params = {'interval': display_interval, 'start': equilibration_steps, 'window': 10000000}
    force_on_wall_id = sim.set_force('constantUpperWallForce', 'right_wall', [0.0, 0.0], force_calculation_params)
    sim.set_force('constantLowerWallForce', 'left_wall',  [0.0, 0.0])
    sim.set_velocity('right_wall', [0, wall_velocity])

    sim.set_velocity_thermal('fluid', temperature)
    sim.set_delta_t(dt)
    sim.fix_temperature(temperature, 'fluid', interval=100)

    def get_force():
        return -sim.force_on_group(force_on_wall_id)[1]

    sim_plot = MolecularDynamicsAnimation(sim, plt.figure(), (2, 3), display_interval)
    sim_plot.add_particle_plot((0, 2, 0, 2))
    sim_plot.add_spatial_velocity_component_histogram((0, 1, 2, 3), groups=['fluid'], nr_of_bins=30)
    scalar_plot_ax = sim_plot.add_scalar_over_time_plot((1, 2, 2, 3), get_force)
    scalar_plot_ax.set_title("Force on moving wall")
    scalar_plot_ax.set_xlabel("t")
    scalar_plot_ax.set_ylabel("$-F_y$")

    sim.run(equilibration_steps)

    return sim_plot.get_animation_object(interval=display_interval, frames=frames)


def local_maxwellian(domain_size=[50, 50], temperature=1, wall_velocity=0.3, dt=0.01,
                     display_interval=1000, equilibration_steps=50000, frames=300):
    """
    Moving wall (Couette) scenario with display of velocity profile and force on wall to `measure` viscosity
    :param domain_size: size of the domain as list [x_width, y_width]
    :param temperature: initial temperature of the simulation, while simulation is running this temperature is
                        enforced, since energy is added by moving wall
    :param wall_velocity: velocity of moving wall
    :param dt: time step length
    :param display_interval: how many time steps to run before a
                             new histogram samples iare taken and particles are displayed
    :param equilibration_steps: number of timesteps to run before sampling and display starts
    :param frames: length of animation in frames
    :return: matplotlib animation
    """
    sim = ParticleSimulation(np.array(domain_size), 3, lj_eps=0.4, verbose=False, boundaries="f p")

    sim.set_atom_type_parameters(atom_type=0, mass=1.0, radius=0.4)
    sim.set_atom_type_parameters(atom_type=1, mass=1.0, radius=0.4)
    sim.set_atom_type_parameters(atom_type=2, mass=1.0, radius=0.4)

    sim.add_atoms('fluid', 0, np.array([[0.1, 0.9], [0.05, 0.95]]), region_is_relative=True, lattice_scale=0.4)
    sim.add_atoms('left_wall', 1, np.array([[0.02, 0.05], [0.00, 1]]), region_is_relative=True, lattice_scale=0.6)
    sim.add_atoms('right_wall', 2, np.array([[0.95, 0.98], [0.00, 1]]), region_is_relative=True, lattice_scale=0.6)

    sim.set_force('constantUpperWallForce', 'right_wall', [0.0, 0.0])
    sim.set_force('constantLowerWallForce', 'left_wall', [0.0, 0.0])
    sim.set_velocity('right_wall', [0, wall_velocity])

    sim.set_velocity_thermal('fluid', temperature)
    sim.set_delta_t(dt)
    sim.fix_temperature(temperature, 'fluid', interval=100)

    sim_plot = MolecularDynamicsAnimation(sim, plt.figure(), (3, 2), display_interval)
    sim_plot.add_particle_plot((0, 2, 0, 2))
    left_sub_domain = ((0, domain_size[0] * 0.4), (0, domain_size[1]))
    right_sub_domain = ((domain_size[0] * 0.6, domain_size[0]), (0, domain_size[1]))

    sim_plot.add_velocity_histogram((2, 3, 0, 1), sub_domain=left_sub_domain, particle_type=0, show_entropy=False)\
            .set_title("Vel Distribution (left)")
    sim_plot.add_velocity_histogram((2, 3, 1, 2), sub_domain=right_sub_domain, particle_type=0, show_entropy=False)\
            .set_title("Vel Distribution (right)")

    sim.run(equilibration_steps)

    return sim_plot.get_animation_object(interval=display_interval, frames=frames)

if __name__ == "__main__":
    #my_animation = single_species_setup(display_interval=100, vel=[0.5, 0.5],
    #                                    show_velocity_histogram=True, show_speed_histogram=True)
    #my_animation = two_species_setup(display_interval=10)
    #my_animation = viscosity_setup(wall_velocity=2.0, equilibration_steps=10)
    #my_animation = moving_wall_setup(keep_temperature_fixed=False)
    my_animation = local_maxwellian(wall_velocity=2.0, equilibration_steps=10)
    plt.show()
