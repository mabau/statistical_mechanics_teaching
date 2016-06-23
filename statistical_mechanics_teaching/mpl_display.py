__author__ = "Martin Bauer <martin.bauer@fau.de>"
__copyright__ = "Copyright 2016, Martin Bauer"
__license__ = "GPL"
__version__ = "3"

from matplotlib.patches import Circle
import matplotlib.animation as animation
import matplotlib
import numpy as np
import numpy.ma as ma

from . import simulation


class ParticleDisplay:

    def __init__(self, axes, sim, type_to_radius, type_to_color):
        assert len(type_to_radius) == len(type_to_color)

        particle_types_arr = sim.types
        self._typeToRadius = type_to_radius
        self._typeToColor = type_to_color
        self._particleCircles = []
        self._sim = sim
        for particleType in particle_types_arr:
            c = Circle((0, 0, 0), type_to_radius[particleType], )
            c.set_color(self._typeToColor[particleType])
            self._particleCircles.append(c)
            axes.add_patch(c)

        axes.set_xlim([0, sim.domain_size[0]])
        axes.set_ylim([0, sim.domain_size[1]])
        axes.set_aspect('equal')

    def update(self):
        positions = self._sim.positions
        for pos, circle in zip(positions, self._particleCircles):
            circle.center = tuple(pos)
        return self._particleCircles

    def __call__(self):
        return self.update()


class Histogram2D:

    def __init__(self, axes, sim, nr_of_bins=[30, 30], hist_range=None, sub_domain=None,
                 particle_type=None, show_entropy=True, window=50, variable='velocities'):
        assert len(nr_of_bins) == 2
        self._sim = sim
        self._nr_of_bins = nr_of_bins
        self._hist_range = hist_range
        self._sub_domain = sub_domain
        self._particle_type = particle_type
        self._variable = variable

        self._histogram_array = None
        self._x_edges = None
        self._y_edges = None
        self._entropy_text = None
        self._window = window
        self._dataHistory = [] # stores result from maximal `window` last steps

        self.clear()
        self.update_data()

        x, y = np.meshgrid(self._x_edges, self._y_edges)

        self._colorMeshPlot = axes.pcolormesh(x, y, np.swapaxes(self._histogram_array, 0, 1))
        axes.set_aspect('equal')
        axes.yaxis.tick_right()
        if self._hist_range:
            axes.set_xlim(self._hist_range[0])
            axes.set_ylim(self._hist_range[1])
        axes.set_xlabel('$v_x$')
        axes.set_ylabel('$v_y$')

        if show_entropy:
            self._entropy_text = axes.text(0.5, 0.1, "Entropy: 123.12", fontsize=14,
                                           horizontalalignment='center', verticalalignment='center',
                                           transform=axes.transAxes)
            self._entropy_text.set_bbox(dict(facecolor='white', alpha=0.9, edgecolor='gray'))

    def clear(self):
        """Clears the histogram history - deletes result obtained by previous update() calls"""
        self._histogram_array = np.zeros(self._nr_of_bins)

    def update_data(self):
        var = getattr(self._sim, self._variable)[:,0:2]

        mask = None
        if self._sub_domain:
            pos = self._sim.positions
            mask_x = np.logical_or(pos[:, 0] <= self._sub_domain[0][0],
                                   pos[:, 0] >= self._sub_domain[0][1])
            mask_y = np.logical_or(pos[:, 1] <= self._sub_domain[1][0],
                                   pos[:, 1] >= self._sub_domain[1][1])
            mask = np.logical_or(mask_x, mask_y)
        if self._particle_type is not None:
            if mask is None:
                mask = (self._sim.types != self._particle_type)
            else:
                mask = np.logical_or(mask, (self._sim.types != self._particle_type))

        if mask is not None:
            tiledmask = np.transpose(np.tile(mask, (2, 1)))
            var = ma.masked_array(var, tiledmask)
            var = var.compressed()
            var = var.reshape([len(var)//2, 2])

        hist, self._x_edges, self._y_edges = np.histogram2d(var[:, 0], var[:, 1],
                                                            bins=self._nr_of_bins, range=self._hist_range)
        if self._window is not None:
            self._dataHistory.append(hist)
            if len(self._dataHistory) > self._window:
                del self._dataHistory[0]
            self._histogram_array = sum(self._dataHistory)
        else:
            self._histogram_array += hist

    def update(self):
        self.update_data()

        hist = np.swapaxes(self._histogram_array, 0, 1).flatten()

        if self._entropy_text:
            import scipy.stats
            entropy = scipy.stats.entropy(hist)
            self._entropy_text.set_text("Entropy %.3f" % (entropy,))

        self._colorMeshPlot.norm.autoscale(hist)
        self._colorMeshPlot.set_array(hist)
        return [self._colorMeshPlot]

    def __call__(self):
        return self.update()


def histogram_surface_plot(axes_3d, histogram, **kwargs):
    x, y = np.meshgrid(histogram.x_edges[:-1], histogram.y_edges[:-1])
    return axes_3d.plot_surface(x, y, histogram.data, **kwargs)


class LammpsHistogram:
    def __init__(self, axes, sim, lammps_setup_commands, nr_of_values, fix_names, colors):
        C = simulation.LammpsConstants
        sim.run_lammps(lammps_setup_commands)
        sim.run(1)  # at least on time step has to be run, for fix to be available
        self._sim = sim
        self._nr_of_values = nr_of_values
        self._fix_names = fix_names
        self._count = 0
        self._axes = axes
        self._rectCollections = []
        self._data_arrays = []
        axes.yaxis.tick_right()

        cc = matplotlib.colors.ColorConverter()
        for fix_name, color in zip(fix_names, colors):
            coordinate_column = 0
            x_values = [self._sim.lammps.extract_fix(fix_name, C.LMP_GLOBAL_DATA, C.LMP_ARRAY, i, coordinate_column)
                        for i in range(self._nr_of_values)]
            width = (max(x_values)-min(x_values)) / nr_of_values

            color = cc.to_rgba(color, alpha=0.8)  # make color slightly transparent
            self._data_arrays.append(np.zeros(nr_of_values))
            rects = axes.bar(x_values, self._data_arrays[-1], width, color=color)
            self._rectCollections.append(rects)

    def __call__(self):
        return self.update()

    def update(self):
        C = simulation.LammpsConstants
        self._count += 1
        x_max = 0
        y_max = 0
        for fix_name, rectCollection, data_array in zip(self._fix_names, self._rectCollections, self._data_arrays):
            data_column = 3
            coordinate_column = 0

            x_values = [self._sim.lammps.extract_fix(fix_name, C.LMP_GLOBAL_DATA, C.LMP_ARRAY, i, coordinate_column)
                        for i in range(self._nr_of_values)]

            new_data = [self._sim.lammps.extract_fix(fix_name, C.LMP_GLOBAL_DATA, C.LMP_ARRAY, i, data_column)
                        for i in range(self._nr_of_values)]

            data_array += new_data
            for rect, h in zip(rectCollection, data_array / self._count):
                rect.set_height(h)

            x_max = max(x_max, max(x_values))
            y_max = max(y_max, max(data_array / self._count))*1.1

        def relative_error(a, b):
            return abs((b - a) / b)

        if relative_error(self._axes.get_xlim()[1], x_max) > 0.1:
            self._axes.set_xlim(0, x_max)
        if relative_error(self._axes.get_ylim()[1], y_max) > 0.1:
            self._axes.set_ylim(0, y_max)

        # Make figure square
        ax = self._axes
        cur_ration = abs((ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]))
        ax.set(aspect=cur_ration, adjustable='box-forced')

        return self._rectCollections


def plot_particles(axes, sim, type_to_radius, type_to_color):
    return ParticleDisplay(axes, sim,type_to_radius, type_to_color)


def plot_velocity_histogram_2d(axes, sim, nr_of_bins=[30, 30], particle_type=None, hist_range=None, sub_domain=None,
                               window=1000, show_entropy=True):
    return Histogram2D(axes, sim, nr_of_bins, hist_range, sub_domain, particle_type=particle_type,
                       window=window, show_entropy=show_entropy)


def plot_xy_histogram(axes, sim, nr_of_bins=[30, 30], particle_type=None, window=1000, show_entropy=True):
    xs, ys = sim.domain_size
    return Histogram2D(axes, sim, nr_of_bins, [[0, xs], [0, ys]], variable='positions',
                       particle_type=particle_type, window=window, show_entropy=show_entropy)


def plot_spatial_velocity_distribution(axes, sim, coordinate, component='vx', groups=['all'], colors=['#3498db'],
                                       interval=1000, nr_of_bins=15, window=10):
    if coordinate == 'x':
        bin_size = sim.domain_size[0] / nr_of_bins
    elif coordinate == 'y':
        bin_size = sim.domain_size[1] / nr_of_bins
    else:
        raise ValueError("Valid coordinates are 'x' and 'y'")

    commands = ""
    fix_ids = []
    for group in groups:
        if group != 'all':
            group = "g_" + group

        fix_id = sim.get_next_unique_lamps_id()
        chunk_id = sim.get_next_unique_lamps_id()
        fix_ids.append(fix_id)
        cmd = """
        compute {chunk_id} {group} chunk/atom bin/1d {coordinate} 0.0 {bin_size}
        fix     {fix_id} {group} ave/chunk 1 {interval} {interval} {chunk_id} {component} ave window {window}
        """.format(chunk_id=chunk_id, fix_id=fix_id, coordinate=coordinate, bin_size=bin_size,
                   group=group, interval=interval, component=component, window=window)
        commands += cmd

    return LammpsHistogram(axes, sim, commands, nr_of_bins, fix_ids, colors=colors)


def plot_speed_distribution(axes, sim, groups=['all'], interval=1000, colors=['blue'], nr_of_bins=30,
                            low=0, high=4, window=10):

    commands = ""
    fix_ids = []
    for group in groups:
        if group != 'all':
            group = "g_" + group

        fix_id = sim.get_next_unique_lamps_id()
        chunk_id = sim.get_next_unique_lamps_id()
        fix_ids.append(fix_id)
        cmd = """
        variable speed atom sqrt(vx*vx+vy*vy+vz*vz)
        fix     {fix_id}  {group} ave/histo 1 {interval} {interval} {low} {high} {nr_of_bins} v_speed mode vector ave window {window}
        """.format(chunk_id=chunk_id, fix_id=fix_id, nr_of_bins=nr_of_bins, group=group, interval=interval,
                   low=low, high=high, window=window)
        commands += cmd

    axes.set_xlabel('$||v||$')
    return LammpsHistogram(axes, sim, commands, nr_of_bins, fix_ids, colors=colors)


def plot_scalar_over_time(axes, scalar_update_function, window=100):

    temporal_data = []
    axes.set_xlim(0, window)
    line, = axes.plot(np.arange(len(temporal_data)), temporal_data, ls='None', marker='o', ms=5)

    def update():
        temporal_data.append(abs(scalar_update_function()))
        if len(temporal_data) > window:
            del temporal_data[0]
        line.set_data(np.arange(len(temporal_data)), temporal_data)
        axes.set_ylim(0, 1.5 * max(temporal_data))

        return [line]

    return update


# --------------------------------  Helper Functions  ------------------------------------------------------------------

def get_default_particle_colors(num_types):
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#95a5a6', '#34495e']
    cyclic_colors = colors * (num_types // len(colors) + 1)
    return cyclic_colors[:num_types]


# --------------------------------  Animated Plot collections ----------------------------------------------------------

class MolecularDynamicsAnimation:

    def __init__(self, sim, fig, grid=[3, 3], display_interval=100, type_to_color=None):
        """
        Create new animation showing results of molecular dynamics simulation

        :param sim: simulation instance
        :param fig: matplotlib figure
        :param grid: size of the plot grid, i.e. how many spaces where plots can be positioned to
        :param display_interval: how many time steps to run before a new animation frame is drawn
        :param type_to_color: array of matplotlib colors ith same length as there are particle types in the simulation,
                              mapping each particle type to a color
        """
        self._sim = sim
        self._fig = fig
        self._display_interval = display_interval
        self._animators = []
        self._grid_spec = matplotlib.gridspec.GridSpec(*grid)
        self._type_to_color = type_to_color
        self._frame_callbacks = []

        if not self._type_to_color:
            self._type_to_color = get_default_particle_colors(sim.nr_of_atom_types)

    def add_particle_plot(self, grid_pos=(0, 1, 0, 1), type_to_radius=None):
        """Adds a plot showing particles as circles
        :param grid_pos: (x_begin, x_end, y_begin, y_end) position to place the plot in the plot grid
        :param type_to_radius: array of radii with same length as there are particle types in the simulation,
                               mapping each particle type to a radius it should be displayed with
        :returns matplotlib axes object that can be used to set title, axes labels, etc.
        """
        if not type_to_radius:
            type_to_radius = [self._sim.get_atom_radius(i) for i in range(self._sim.nr_of_atom_types)]

        grid_spec = self._grid_spec[grid_pos[0]:grid_pos[1], grid_pos[2]:grid_pos[3]]
        ax = self._fig.add_subplot(grid_spec, adjustable='box', aspect=1)
        ax.set_title("Lennard Jones MD Simulation")
        self._animators.append(plot_particles(ax, self._sim, type_to_radius, self._type_to_color))
        return ax

    def add_velocity_histogram(self, grid_pos=(0, 1, 0, 1), hist_range=((-2, 2), (-2, 2)),
                               particle_type=None, sub_domain=None, show_entropy=True, window=1000):
        """Adds a plot showing a 2D histogram of velocity distribution as a colormap
        :param grid_pos: (x_begin, x_end, y_begin, y_end) position to place the plot in the plot grid
        :param hist_range: (x_begin, x_end, y_begin, y_end) range for histogram, velocities outside are discarded
        :param particle_type: include only particles of this type in histogram (if None all are included)
        :param sub_domain: only used particles in the given box for histogram.
                            ((x_min,x_max),(y_min,y_max)) or None for complete domain
        :param show_entropy: add textfield to plot displaying current entropy
        :param window: number of temporal samples to keep for histogram, or None to keep all
        :returns matplotlib axes object that can be used to set title, axes labels, etc.
        """
        grid_spec = self._grid_spec[grid_pos[0]:grid_pos[1], grid_pos[2]:grid_pos[3]]
        ax = self._fig.add_subplot(grid_spec)
        self._animators.append(plot_velocity_histogram_2d(ax, self._sim, hist_range=hist_range,
                                                          particle_type=particle_type, show_entropy=show_entropy,
                                                          sub_domain=sub_domain, window=window))
        ax.set_title("Velocity Distribution")
        ax.set_xlabel("$v_x$")
        ax.set_ylabel("$v_y$")
        return ax

    def add_xy_histogram(self, grid_pos=(0, 1, 0, 1), particle_type=None, show_entropy=True, window=1000):
        """Adds a plot showing a 2D histogram particle position distribution
        :param grid_pos: (x_begin, x_end, y_begin, y_end) position to place the plot in the plot grid
        :param particle_type: include only particles of this type in histogram (if None all are included)
        :param show_entropy: add textfield to plot displaying current entropy
        :param window: number of temporal samples to keep for histogram, or None to keep all
        :returns matplotlib axes object that can be used to set title, axes labels, etc.
        """
        grid_spec = self._grid_spec[grid_pos[0]:grid_pos[1], grid_pos[2]:grid_pos[3]]
        ax = self._fig.add_subplot(grid_spec)
        self._animators.append(plot_xy_histogram(ax, self._sim, particle_type=particle_type,
                                                 show_entropy=show_entropy, window=window))
        ax.set_title("Spatial Distribution")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        return ax

    def add_speed_histogram(self, grid_pos=(0, 1, 0, 1), one_speed_hist_for_each_type=False, window=1000):
        """Adds a plot showing a 1D histogram of the speed distribution
        :param grid_pos: (x_begin, x_end, y_begin, y_end) position to place the plot in the plot grid
        :param one_speed_hist_for_each_type: if enabled separate speed histograms are plotted for each particle type
        :param window: number of temporal samples to keep for histogram, or None to keep all
        :returns matplotlib axes object that can be used to set title, axes labels, etc.
        """
        grid_spec = self._grid_spec[grid_pos[0]:grid_pos[1], grid_pos[2]:grid_pos[3]]
        ax = self._fig.add_subplot(grid_spec)

        if hasattr(one_speed_hist_for_each_type, '__len__'):
            groups = [self._sim.atom_type_groups[i] for i in one_speed_hist_for_each_type]
            colors = [self._type_to_color[i] for i in one_speed_hist_for_each_type]
        else:
            if one_speed_hist_for_each_type:
                groups = self._sim.atom_type_groups
                colors = self._type_to_color
            else:
                groups = ['all']
                colors = [self._type_to_color[0]]

        self._animators.append(plot_speed_distribution(ax, self._sim, interval=self._display_interval,
                                                       groups=groups, colors=colors, window=window))
        ax.set_title("Speed Distribution")
        ax.set_xlabel("$||v||$")
        return ax

    def add_spatial_velocity_component_histogram(self, grid_pos=(0, 1, 0, 1),
                                                 coordinate='x', component='vy', **kwargs):
        """
        Adds a plot of velocity along a certain coordinate axis (spatial velocity profile)
        :param grid_pos: (x_begin, x_end, y_begin, y_end) position to place the plot in the plot grid
        :param coordinate: 'x' or 'y', velocity profile is shown along that axis
        :param component: 'vx' or 'vy' which velocity component to show
        :param kwargs: further arguments are passed to plot_spatial_velocity_distribution
        :return: matplotlib axes object that can be used to set title, axes labels, etc.
        """
        grid_spec = self._grid_spec[grid_pos[0]:grid_pos[1], grid_pos[2]:grid_pos[3]]
        ax = self._fig.add_subplot(grid_spec)
        self._animators.append(plot_spatial_velocity_distribution(ax, self._sim, coordinate=coordinate,
                                                                  component=component, **kwargs))
        ax.set_title("Spatial velocity distribution")
        ax.set_xlabel("$%s$" % (coordinate,))
        ax.set_ylabel("$%s$" % (component,))
        return ax

    def add_scalar_over_time_plot(self, grid_pos, scalar_update_function, num_samples_kept=60):
        """
        Adds a plot showing the time evolution of some scalar quantity.
        The scalar quantity is obtained calling the provided update function.
        :param grid_pos: (x_begin, x_end, y_begin, y_end) position to place the plot in the plot grid
        :param scalar_update_function: function return a single (positive!) value which is called every frame
        :param num_samples_kept: how many values to show in the plot, older values are discarded
        :return: matplotlib axes object that can be used to set title, axes labels, etc.
        """
        grid_spec = self._grid_spec[grid_pos[0]:grid_pos[1], grid_pos[2]:grid_pos[3]]
        ax = self._fig.add_subplot(grid_spec)
        self._animators.append(plot_scalar_over_time(ax, scalar_update_function, num_samples_kept))
        return ax

    def get_animation_object(self, **kwargs):
        """Returns matplotlib FuncAnimation object - use this e.g. to store animation as video"""
        self.__update()
        return animation.FuncAnimation(self._fig, self.__update, **kwargs)

    def __update(self, i=0):
        for cb in self._frame_callbacks:
            cb(i)
        self._sim.run(self._display_interval)
        return [a() for a in self._animators]

    def add_frame_callback(self, callback):
        """Add a function that is called before a frame is rendered.
         :callback: function taking one integer (the frame number) as argument
        """
        self._frame_callbacks.append(callback)
