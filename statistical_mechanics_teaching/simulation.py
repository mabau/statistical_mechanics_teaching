__author__ = "Martin Bauer <martin.bauer@fau.de>"
__copyright__ = "Copyright 2016, Martin Bauer"
__license__ = "GPL"
__version__ = "3"

from lammps import lammps
import numpy as np


class LammpsConstants:
    LMP_GLOBAL_DATA = 0
    LMB_SCALAR = 0
    LMP_VECTOR = 1
    LMP_ARRAY = 2


class ParticleSimulation:

    def __init__(self, domain_size, nr_of_atom_types=1, lj_eps=0.001, boundaries="p p", verbose=False):
        self._domain_size = domain_size
        self._nr_of_atom_types = nr_of_atom_types
        self._verbose=verbose
        self._lammps_id_counter = 0
        cmdargs = ["-pk", "omp", "4", "-sf", "omp"]
        if not verbose:
            cmdargs += ['-screen', 'none']
        self._lammps = lammps(cmdargs=cmdargs)
        self._lj_eps = lj_eps

        commands = """
        dimension 2
        boundary {boundaries} p
        atom_style atomic
        neighbor 0.6 bin      # <skin> <style>
        neigh_modify delay 2

        region r_whole_domain block 0 {domain_size[0]} 0 {domain_size[1]} -0.1 0.1
        create_box {nr_of_atom_types} r_whole_domain

        pair_style lj/cut 1.2246
        timestep 0.1
        fix f_time_integration all nve
        fix f_enforce_2d all enforce2d

        #compute myke all ke
        #compute mype all pe
        #variable myte equal "c_myke + c_mype"
        thermo 10000
        #thermo_style custom c_myke c_mype v_myte

        #thermo_modify lost ignore flush yes
        """
        self._run_lammps(commands.format(domain_size=domain_size, nr_of_atom_types=nr_of_atom_types,
                                         boundaries=boundaries))

        self._atom_type_info = [None] * nr_of_atom_types
        for i in range(nr_of_atom_types):
            self._atom_type_info[i] = {'mass': 1.0, 'radius': 1.0}

    def activate_reflecting_walls(self, x=True, y=True):
        if x:
            self.run_lammps("fix xwalls all wall/reflect xlo EDGE xhi EDGE")
        if y:
            self.run_lammps("fix ywalls all wall/reflect ylo EDGE yhi EDGE")

    def get_next_unique_lamps_id(self):
        self._lammps_id_counter += 1
        return "generated_id_%d" % (self._lammps_id_counter,)

    def set_atom_type_parameters(self, atom_type, mass, radius):
        assert atom_type < len(self._atom_type_info)
        self._atom_type_info[atom_type]['mass'] = mass
        self._atom_type_info[atom_type]['radius'] = radius

        coeff_cmd = "pair_coeff {i} {j} {eps} {sigma} {r_cut}\n"
        mass_cmd = "mass {i} {mass}\n"

        commands = ""
        for i, entry1 in enumerate(self._atom_type_info):
            commands += mass_cmd.format(i=i+1, mass=entry1['mass'])
            for j, entry2 in enumerate(self._atom_type_info):
                if i > j:
                    continue
                sigma = 2.5 * 0.5 * (entry1['radius'] + entry2['radius'])
                commands += coeff_cmd.format(i=i+1, j=j+1, sigma=sigma, eps=self._lj_eps, r_cut=1.8*sigma)

        self._run_lammps(commands)

    def get_atom_radius(self, atom_type):
        assert atom_type < self.nr_of_atom_types
        return self._atom_type_info[atom_type]['radius']

    def get_atom_mass(self, atom_type):
        assert atom_type < self.nr_of_atom_types
        return self._atom_type_info[atom_type]['mass']

    @property
    def atom_type_groups(self):
        return ["atom_type_%d" % (i+1,) for i in range(self._nr_of_atom_types)]

    def add_atoms(self, name, atom_type, region, lattice_type='hex', lattice_scale=0.7, region_is_relative=False):
        command_tmpl = """
        region r_{name} block {region[0][0]} {region[0][1]} {region[1][0]} {region[1][1]} -0.1 0.1
        lattice {lattice_type} {lattice_scale}
        create_atoms {atom_type} region r_{name}
        group g_{name} region r_{name}
        group g_atom_type_{atom_type} type {atom_type}
        lattice none 1.0 # reset scaling
        """
        atom_type += 1
        if region_is_relative:
            region[0, :] *= self._domain_size[0]
            region[1, :] *= self._domain_size[1]
        d = locals()
        cmd = command_tmpl.format(**locals())
        self._run_lammps(cmd)

    def set_velocity(self, name, vel):
        if hasattr(vel, '__len__'):
            cmd = """velocity g_{name} set {vel[0]} {vel[1]} 0""".format(name=name, vel=vel)
            self._run_lammps(cmd)
        else:
            self.set_velocity_thermal(name, vel)

    def set_velocity_thermal(self, name, temp, seed=4242):
        cmd = """velocity g_{name} create {temp} {seed}""".format(name=name, temp=temp, seed=seed)
        self._run_lammps(cmd)

    def fix_temperature(self, temp, group_id, interval=100):
        cmd = "fix {fix_id} g_{group_id} temp/rescale {interval} {temp} {temp} 0.05 1.0"
        cmd = cmd.format(fix_id=self.get_next_unique_lamps_id(), group_id=group_id, temp=temp, interval=interval)
        self._run_lammps(cmd)

    def set_force(self, fix_name, group, force, force_on_wall_computation=None):
        """Fixes the force on a group of atoms and optionally computes the force acting on the group of atoms
            Example for computing
            \param force_on_wall_computation:
                    None if no force should be computed or a dict like the following
                     {'interval' : 1000,   # how often the force should be calculated
                      'start': 5000,       # timestep to begin calculation
                      'window': 10000000}) # do not include data from timestep longer than window ago
                    use member function force_on_group( returnedValueOfSet_force ) to get the force
        """
        cmd = "fix {fix_name} g_{group} setforce {force[0]} {force[1]} 0\n".format(**(locals()))
        avg_fix_name = ""
        if force_on_wall_computation:
            avg_fix_name = "timeavg_" + fix_name
            avg_cmd = "fix {avg_fix_name} g_{group} ave/time 1 {p[interval]} {p[interval]} f_{fix_name} " + \
                      " mode vector start {p[start]} ave window {p[window]}"
            cmd += avg_cmd.format(fix_name=fix_name, avg_fix_name=avg_fix_name, group=group, p=force_on_wall_computation)
        self._run_lammps(cmd)
        return avg_fix_name

    def force_on_group(self, fix_id):
        C = LammpsConstants
        f = [ self.lammps.extract_fix(fix_id, C.LMP_GLOBAL_DATA, C.LMP_VECTOR, i, 0) for i in range(3)]
        return f

    def ave_force(self, fix_name, group, force):
        cmd = "fix {fix_name} g_{group} aveforce {force[0]} {force[1]} 0".format(**(locals()))
        self._run_lammps(cmd)

    def set_delta_t(self, delta_t):
        self._run_lammps("timestep %f" % (delta_t,))

    def run(self, time_steps):
        cmd = """
        unfix f_time_integration
        unfix f_enforce_2d
        fix f_time_integration all nve
        fix f_enforce_2d all enforce2d
        run {time_steps}
        """.format(time_steps=time_steps)
        self._run_lammps(cmd)

    @property
    def positions(self):
        res = np.array(self._lammps.gather_atoms("x", 1, 3))
        res = res.reshape((self._lammps.get_natoms(), 3))
        return res

    @property
    def nr_of_atom_types(self):
        return self._nr_of_atom_types

    @property
    def types(self):
        res = np.array(self._lammps.gather_atoms("type", 0, 1))
        return res-1

    @property
    def velocities(self):
        res = np.array(self._lammps.gather_atoms("v", 1, 3))
        res = res.reshape((self._lammps.get_natoms(), 3))
        return res

    @property
    def domain_size(self):
        return self._domain_size

    def _run_lammps(self, command):
        for line in command.split("\n"):
            if self._verbose:
                print("> " + line.strip())
            self._lammps.command(line)

    def run_lammps(self, command):
        return self._run_lammps(command)

    @property
    def lammps(self):
        return self._lammps

