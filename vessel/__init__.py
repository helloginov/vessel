import warnings
from copy import deepcopy

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.interpolate import interpn
try:
    from freeqdsk import geqdsk 
except ImportError as e:
    raise ImportError(
        "The 'freeqdsk-0.5.0' library is required but not installed. "
        "Please install it from https://github.com/freegs-plasma/FreeQDSK?tab=readme-ov-file"
    ) from e
# docs: https://freeqdsk.readthedocs.io/en/stable/geqdsk.html


class Trace:
    """
    Represents a diagnostic trace in a tokamak vessel poloidal cut.

    The `Trace` class models a cross-section of the tokamak chamber as a straight line 
    between two points (antenna and view). It provides methods for interpolating 
    magnetic field and poloidal flux (psi) profiles and calculating angles to magnetic surfaces
    based on conditions.

    Attributes:
        r (np.ndarray): Radial coordinates of the trace.
        z (np.ndarray): Vertical coordinates of the trace.
        dist (np.ndarray): Distance from the antenna along the trace.
        psi (np.ndarray): Poloidal magnetic flux along the trace.
        b_tor (np.ndarray): Toroidal magnetic field along the trace.
        b_pol (np.ndarray): Poloidal magnetic field along the trace.
        reflection_angle (np.ndarray): Reflection angle along the trace.
        b_mode (float): Magnetic field value at the magnetic axis.
    """

    def __init__(
            self, pos: tuple, view: tuple,
            r_grid: np.ndarray, z_grid: np.ndarray, 
            psi_profile: np.ndarray, b_toroid_profile: np.ndarray, b_poloid_profile: np.ndarray, 
            maxis_b_value: float, resolution: int, need_b=True, need_angle=True, need_dist=True, crop_tail=True
        ):
        """
        Creates a cross-section of the chamber as a straight line between two points: antenna and view.
        The cross-section is described by the attributes r, z, dist, psi, b_tor, b_pol, reflection_angle.

        Args:
            pos (tuple): Initial point of the trace in (R, Z) coordinates.
            view (tuple): Any (R, Z) point in the cut, defining the view of the trace.
            r_grid (np.ndarray): Radial coordinates of shape (n,).
            z_grid (np.ndarray): Height coordinates of shape (m,).
            psi_profile (np.ndarray): Poloidal magnetic flux cut distribution, grid of shape (m, n).
            b_toroid_profile, b_toroid_profile (np.ndarray): Toroidal and poloidal magnetic field cut distribution, grids of shape (m, n).
            maxis_b_value (float): Magnetic field value at the magnetic axis ("center"). If it does not correspond to the value in the profile, it will be used for normalization.
            resolution (int): Number of points to interpolate the trace on.
            need_b (bool): If True, interpolates the magnetic field components along the trace. Defaults to True.
            need_angle (bool): If True, calculates the reflection angle along the trace. Defaults to True.
            need_dist (bool): If True, calculates the distance from the antenna along the trace. Defaults to True.
            crop_tail (bool): If True, the trace will be cropped at the minimum poloidal flux. Defaults to True.
        """

        R_2Dgrid, Z_2Dgrid = np.meshgrid(r_grid, z_grid, indexing='ij')

        # задание луча в координатах (R, Z)                 с запасом в двое длиннее
        self._r = np.linspace(pos[0], view[0], resolution)
        self._z = np.linspace(pos[1], view[1], resolution)

        # проверка на выход за границы сетки: если луч выходит по какой-то координате за допустимые пределы, то строим новый луч до границы.
        inlying_part = np.argwhere(((self._r > R_2Dgrid[0, 0]) & (self._r < R_2Dgrid[-1, 0]) & (self._z > Z_2Dgrid[0, 0]) & (self._z < Z_2Dgrid[0, -1])))

        if inlying_part.size != 0:
            end_idx = np.array(inlying_part[-1]).item() # inlying_part[-1] может оказаться скаляром, а может массивом с 1 элементом

            if end_idx != len(self._r) - 1:
                self._r = np.linspace(pos[0], self._r[end_idx], resolution)
                self._z = np.linspace(pos[1], self._z[end_idx], resolution)

        # интерполяция двумерного распределения полоидального потока на направление зондирования
        self._psi = self.interpolate_on_trace(r_grid, z_grid, psi_profile)

        # долой часть луча, идущую после минимума полоидального потока
        if crop_tail:
            self.crop_tail()

        # если появились артефакты в виде отрицательных значений, то зануляем их
        self._psi[self._psi < 0] = 0.0

        # задание луча в кординатах расстояния от антенны
        if need_dist:
            self._dist = np.sqrt((self._r - pos[0])**2 + (self._z - pos[1])**2)

        # интерполяция двумерных распределений тороидального и полоидального полей на направление зондирования
        if need_b:
            self._b_tor = self.interpolate_on_trace(r_grid, z_grid, b_toroid_profile)
            self._b_pol = self.interpolate_on_trace(r_grid, z_grid, b_poloid_profile)
            self._b_tor_mode = maxis_b_value
        
        # наклон зондирующего луча к нормалям магнитных поверхностей на протяжении всей длины.
        if need_angle:
            self._reflection_angle = self._trace_angle(r_grid, z_grid, psi_profile)


    def cut(self, condition: np.array = None, inplace=False):
        """
        Cuts the trace based on a given condition.

        Args:
            condition (np.ndarray, optional): A boolean array indicating which points to keep. 
                Defaults to keeping points where `self.psi <= 1`.
            inplace (bool): If True, modifies the current trace in place. 
                If False, creates and returns a new trace object. Defaults to False.

        Returns:
            Trace: The modified trace (if `inplace=True`) or a new trace object (if `inplace=False`).
        """

        if condition is None:
            condition = (self.psi <= 1)

        trace = self if inplace else deepcopy(self)
        
        trace._r = self._r[condition]
        trace._z = self._z[condition]
        trace._psi = self._psi[condition]
        try:
            trace._dist = self._dist[condition]
        except AttributeError:
            pass
        try:
            trace._b_pol = self._b_pol[condition]
            trace._b_tor = self._b_tor[condition]
        except AttributeError:
            pass
        try:
            trace._reflection_angle = self._reflection_angle[condition]
        except AttributeError:
            pass

        return trace
    

    def crop_tail(self, end_idx=None):
        """
        Crops the trace by removing points after a specified index or the minimum poloidal flux.

        Args:
            end_idx (int, optional): The index up to which the trace should be kept. 
                If None, the trace is cropped at the first occurrence of the minimum poloidal flux 
                where `self._psi > 0.0`. Defaults to None.

        Returns:
            None: Modifies the trace in place by calling the `cut` method with the appropriate condition.
        """
        if end_idx is None:
            end_idx = self._psi[self._psi > 0.0].argmin()
        self.cut(condition=(np.arange(len(self._psi)) <= end_idx), inplace=True)

    
    def interpolate_on_trace(self, r_grid, z_grid, profile2d, method="linear"):
        """
        Interpolates a 2D profile along the trace.

        Args:
            r_grid (np.ndarray): Radial grid.
            z_grid (np.ndarray): Vertical grid.
            profile2d (np.ndarray): 2D profile to interpolate.
            method (str): Interpolation method ('linear', 'nearest', 'splinef2d').

        Returns:
            np.ndarray: Interpolated values along the trace.
        """
        return interpn(
            points=(r_grid, z_grid),
            values=profile2d,
            xi=(self._r, self._z),
            method=method
        )


    def _trace_angle(self, r_grid: np.ndarray, z_grid: np.ndarray, psi_profile: np.ndarray):
        """
        Calculates the angle between the trace and the magnetic surfaces.

        Args:
            r_grid (np.ndarray): Radial 1D grid.
            z_grid (np.ndarray): Vertical 1D grid.
            psi_profile (np.ndarray): Poloidal magnetic flux 2D profile.

        Returns:
            np.ndarray: Array of angles (in radians) along the trace.
        """
        grad_psi_r, grad_psi_z = np.gradient(psi_profile, r_grid, z_grid, edge_order=2)
        grad_psi_along_trace = np.stack([
            self.interpolate_on_trace(r_grid, z_grid, grad_psi_r),
            self.interpolate_on_trace(r_grid, z_grid, grad_psi_z)
        ], axis=0)

        k = np.array([self._r[-1] - self._r[0], self._z[-1] - self._z[0]])
        norm_k = np.linalg.norm(k)
        norm_grad = np.linalg.norm(grad_psi_along_trace, axis=0)

        dot_product = np.einsum('i,ij->j', -k, grad_psi_along_trace)
        angles = np.arccos(dot_product / (norm_k * norm_grad))
        return angles
    
    @property
    def r(self):        return self._r
    @property
    def z(self):        return self._z
    @property
    def dist(self):     return self._dist
    @property
    def psi(self):      return self._psi
    @property
    def b_full(self):   return np.sqrt(self._b_tor**2 + self._b_pol**2)
    @property
    def b_tor(self):    return self._b_tor
    @property
    def b_pol(self):    return self._b_pol
    @property
    def reflection_angle(self): return self._reflection_angle
    @property
    def b_mode(self):   return self._b_tor_mode


class Antenna:
    """
    Represents an antenna in a tokamak vessel.

    The `Antenna` class models the position, view direction, and associated trace 
    of an antenna. It provides methods for coordinate transformations and accessing 
    antenna properties.

    Attributes:
        name (str): The name of the antenna.
        pos (tuple): The position of the antenna in (R, Z) coordinates.
        view (tuple): The view direction of the antenna in (R, Z) coordinates.
        trace (Trace): The trace object associated with the antenna.
        rad (float): Distance of the antenna from the magnetic axis.
        rotated_by (float): Angle by which the antenna is rotated relative to the horizon.
    """

    def __init__(self, name: str, pos: tuple, view: tuple, trace:Trace, maxis: tuple):
        """
        Initializes an Antenna object.

        Args:
            name (str): The name of the antenna.
            pos (tuple): The position of the antenna in (R, Z) coordinates.
            view (tuple): The view direction of the antenna in (R, Z) coordinates.
            trace (Trace): The trace object associated with the antenna.
            maxis (tuple): The position of the magnetic axis in (R, Z) coordinates.

        Raises:
            ValueError: If `pos`, `view`, or `maxis` are not 2D points.
        """

        if len(pos) != 2 or len(view) != 2 or len(maxis) != 2:
            raise ValueError("Arguments 'pos', 'view', and 'maxis' must be 2D points (tuples or arrays of length 2).")

        self.name = name
        self._pos = pos
        self._view = view
        self._trace = trace

        self._maxis = maxis

        ex = self._pos - self._maxis
        self._rad = np.linalg.norm(ex)

        ex /= self._rad
        ey = np.array([-ex[1], ex[0]])

        self._trans_mx = np.array([ex, ey])
        self._invtrans_mx = np.linalg.inv(self._trans_mx)
        
        self._rotated_by = np.arccos(ex[0]) * np.sign(ex[1])
        
    
    def xy2rz(self, xy):
        """
        Converts Cartesian (x, y) coordinates to cylindrical (R, Z) coordinates.

        Args:
            xy (array-like): A 2D point in Cartesian coordinates.

        Returns:
            np.ndarray: The corresponding point in cylindrical coordinates.
        """

        return np.asarray(xy) @ self._trans_mx + self._maxis


    def rz2xy(self, rz):
        """
        Converts cylindrical (R, Z) coordinates to Cartesian (x, y) coordinates.

        Args:
            rz (array-like): A 2D point in cylindrical coordinates.

        Returns:
            np.ndarray: The corresponding point in Cartesian coordinates.
        """

        return (np.asarray(rz) - self._maxis) @ self._invtrans_mx
    

    @property
    def trace(self):
        """
        Returns the trace object associated with the antenna.

        Returns:
            Trace: The trace object.
        """

        return self._trace
    
    @property
    def rad(self):
        """
        Returns the distance of the antenna from the magnetic axis.

        Returns:
            float: The distance.
        """

        return self._rad
    
    @property
    def rotated_by(self): 
        """
        Returns the angle by which the antenna is rotated relative to the horizon.

        Returns:
            float: The rotation angle in radians.
        """

        return self._rotated_by
    
    
    def get_pos(self, coords='rz'):
        """
        Returns the position of the antenna in the specified coordinate system.

        Args:
            coords (str): Coordinate system: 'xy' for Cartesian, 'rz' for cylindrical. Defaults to 'rz'.

        Returns:
            np.ndarray: Position in the specified coordinate system.
        """

        return self.adjust_coords(self._pos, coords)
    

    def get_view(self, coords='rz'):
        """
        Returns the view direction of the antenna in the specified coordinate system.

        Args:
            coords (str): Coordinate system: 'xy' for Cartesian, 'rz' for cylindrical. Defaults to 'rz'.

        Returns:
            np.ndarray: View direction in the specified coordinate system.
        """

        return self.adjust_coords(self._view, coords)
    
    
    def adjust_coords(self, point, coords='rz'):
        """
        Adjusts the coordinates of a point between 'rz' and 'xy' systems.

        Args:
            point (array-like): The point to adjust.
            coords (str): The coordinate system to adjust to ('rz' or 'xy').

        Returns:
            np.ndarray: The adjusted point in the specified coordinate system.

        Raises:
            ValueError: If `coords` is not 'rz' or 'xy'.
        """
        
        if coords == 'rz':
            return np.asarray(point)
        elif coords == 'xy':
            return self.rz2xy(point)
        else:
            raise ValueError(f"Invalid coordinate system '{coords}'. Valid options are 'rz' or 'xy'.")
    

class Vessel:
    """
    Represents the geometry, magnetic field profiles, and plasma configuration of a tokamak vessel.

    The `Vessel` class provides a comprehensive representation of a tokamak's equilibrium 
    for reflectometry purposes. It allows for the addition of antennas and traces, 
    visualization of parameters within the vessel, and parsing gEQDSK files for initialization.

    Attributes:
        r (np.ndarray): Radial grid of shape (n, ).
        z (np.ndarray): Vertical grid of shape (m, ).
        psi_profile (np.ndarray): Normalized poloidal flux profile of shape (m, n).
        b_toroid_profile (np.ndarray): Toroidal magnetic field profile of shape (m, n).
        b_poloid_profile (np.ndarray): Poloidal magnetic field profile of shape (m, n).
        maxis (tuple): Coordinates of the magnetic axis (R, Z).
        vessel_shape (np.ndarray): Boundary of the vessel, array of shape (2, V).
        separatrix (np.ndarray): Plasma separatrix, array of shape (2, S).
    """

    def __init__(
            self, r_grid: np.ndarray, z_grid: np.ndarray, psi_profile: np.ndarray, 
            maxis: tuple, maxis_mfield_value: float, 
            b_toroid_profile: np.ndarray, b_poloid_profile: np.ndarray,
            vessel_shape: np.ndarray = None, separatrix: np.ndarray = None
        ):
        """
        Initializes the Vessel object with the given parameters.
        The vessel is represented by a grid of radial and vertical coordinates, along with profiles of poloidal flux and magnetic fields.

        Args:
            r_grid, z_grid (np.ndarray): 1D arrays of radial and vertical coordinates.
            psi_profile (np.ndarray): 2D array of the normalized poloidal flux.
            maxis (tuple): Coordinates of the magnetic axis (R, Z).
            maxis_mfield_value (float): Magnetic field value at the magnetic axis.
            b_toroid_profile, b_poloid_profile (np.ndarray): 2D array of the toroidal and poloidal magnetic field.
            vessel_shape (np.ndarray): Array of shape (2, V) representing the vessel boundary. Defaults to None.
            separatrix (np.ndarray): Array of shape (2, S) representing the plasma separatrix. Defaults to None.

        Raises:
            ValueError: If the input profiles do not have the same shape.
            RuntimeWarning: If the grid is square and the coordinate axes are ambiguous.
        """

        if not psi_profile.shape == b_toroid_profile.shape == b_poloid_profile.shape:
            raise ValueError("profiles have not the same shapes")
        elif psi_profile.shape[0] == psi_profile.shape[1]:
            warnings.warn("grid is square. Be sure if axis 0 relates R-coordinate, axis 1 - Z-coodrinate.", RuntimeWarning)

        self._r = r_grid
        self._z = z_grid
        self._separatrix = separatrix
        self._vessel_shape = vessel_shape
        self._maxis = maxis

        vessel_mask = ~self._create_vessel_mask()

        self._psi_profile = ma.array(psi_profile, mask=vessel_mask, )
        self._b_poloid_profile = ma.array(b_poloid_profile, mask=vessel_mask, )
        self._b_toroid_profile = ma.array(b_toroid_profile, mask=vessel_mask, )
        self._b_mode = maxis_mfield_value
    
        self._antennae = dict()

        self._warn_if_bad_maxis(r_grid, z_grid)


    def add_antenna(self, pos=None, view=None, resolution=None, store_as: str = None, **trace_kwargs):
        """
        Adds an antenna and its associated trace to the vessel.

        Args:
            pos (tuple, optional): Position of the antenna in (R, Z) coordinates. 
                Defaults to the outer side of the vessel boundary in front of the magnetic axis.
            view (tuple, optional): View direction of the antenna in (R, Z) coordinates. Defaults to the magnetic axis.
            resolution (int, optional): Number of points to interpolate the trace on. Defaults to the radial grid size.
            store_as (str, optional): Name to store the antenna as. If None, the method returns the trace and forgets about it. Defaults to None.
            **trace_kwargs: Additional arguments for the `Trace` object.

        Returns:
            Trace: The trace object associated with the antenna.
        """

        if store_as in self._antennae:
            raise ValueError(f"An antenna with the name '{store_as}' already exists. Please choose a unique name.")

        # по дефолту луч строится в экваториальной плоскости со стороны слабого поля
        pos        = pos        if pos        is not None else (self._vessel_shape[0].max(), self._maxis[1])
        view       = view       if view       is not None else self._maxis
        resolution = resolution if resolution is not None else len(self._r)

        self._check_view_is_ok(pos, view)

        trace = Trace(
            pos, view,
            self._r, self._z, self._psi_profile.data, 
            self._b_toroid_profile.data, self._b_poloid_profile.data, self._b_mode,
            resolution, **trace_kwargs
        )
        if store_as is not None:
            self._antennae[store_as] = Antenna(
                store_as,
                np.array(pos), np.array(view),
                trace, self._maxis
            )
        return trace
    

    @classmethod
    def from_geqdsk(cls, filepath):
        """
        Initializes a Vessel object from a GEQDSK file.

        Args:
            filepath (str): Path to the GEQDSK file.

        Returns:
            Vessel: A Vessel object initialized from the GEQDSK file.
        """

        with open(filepath, "r") as f:
            eqdsk = geqdsk.read(f)

        # Coordinate grids
        R = np.linspace(eqdsk.rleft, eqdsk.rleft + eqdsk.rdim, eqdsk.nx)
        Z = np.linspace(eqdsk.zmid - eqdsk.zdim / 2, eqdsk.zmid + eqdsk.zdim / 2, eqdsk.ny)

        # Gradients of the poloidal flux
        psi_grad = np.gradient(eqdsk.psi, R, Z)

        # Poloidal and toroidal components of the magnetic field
        b_pol = (np.linalg.norm(psi_grad, axis=0).T / R).T
        b_tor = np.vstack([eqdsk.bcentr * eqdsk.rcentr / R for _ in range(eqdsk.ny)]).T

        # Profile of the normalized poloidal flux
        psi_norm = 1 - (eqdsk.psi - eqdsk.sibdry) / (eqdsk.simagx - eqdsk.sibdry)

        return cls(
            r_grid=R, 
            z_grid=Z, 
            psi_profile=psi_norm,
            maxis=np.array([eqdsk.rmaxis, eqdsk.zmaxis]),
            maxis_mfield_value=eqdsk.bcentr,
            b_toroid_profile=b_tor,
            b_poloid_profile=b_pol,
            vessel_shape=np.vstack((eqdsk.rlim, eqdsk.zlim)) / 100, # / 100 only for T-15MD.eqdsk
            separatrix=np.vstack((eqdsk.rbdry, eqdsk.zbdry))
            )            
       
    
    def list_antennae(self, names_only=True):
        """
        Lists all antennas added to the vessel.

        Args:
            names_only (bool): If True, returns only the names of the antennas. Defaults to True.

        Returns:
            list: List of antenna names or objects.
        """

        return self._antennae.keys() if names_only else self._antennae
    
    
    def get_trace(self, name):
        """
        Retrieves an trace object by the antenna name.

        Args:
            name (str): Name of the antenna.

        Returns:
            Antenna: The antenna object.
        """

        return self._antennae[name].trace
    

    def get_antenna(self, name):
        """
        Retrieves an antenna object by name.

        Args:
            name (str): Name of the antenna.

        Returns:
            Antenna: The antenna object.
        """

        return self._antennae[name]


    def get_psi2d(self): 
        """
        Returns the 2D poloidal flux profile.

        Returns:
            np.ndarray: The normalized poloidal flux profile.
        """

        return self._psi_profile


    def get_b(self, which='full'):
        """
        Returns the magnetic field profile.

        Args:
            which (str): Specifies the field component ('full', 'tor', 'pol'). Defaults to 'full'.

        Returns:
            np.ndarray: The requested magnetic field profile.
        """

        if which == 'tor':
            return self._b_toroid_profile
        elif which == 'pol':
            return self._b_poloid_profile
        elif which == 'full':
            return np.sqrt(self._b_toroid_profile**2 + self._b_poloid_profile**2)
        else:
            raise ValueError("argument 'which' can be one of the following: 'full'(by default), 'tor', 'pol'")
    

    def get_maxis(self): 
        """
        Returns the coordinates of the magnetic axis.

        Returns:
            tuple: Coordinates of the magnetic axis (R, Z).
        """

        return self._maxis

    def get_coords(self): 
        """
        Returns the radial and vertical coordinate grids.

        Returns:
            tuple: Radial and vertical grids.
        """

        return (self._r, self._z)

    def get_separatrix(self):   
        """
        Returns the plasma separatrix.

        Returns:
            np.ndarray: Coordinates of the separatrix.
        """

        return self._separatrix

    def get_vessel_shape(self):   
        """
        Returns the vessel boundary.

        Returns:
            np.ndarray: Coordinates of the vessel boundary.
        """

        return self._vessel_shape


    def visualize_param_in_vessel(self, param_grid, param_name=None, draw_traces=False, fig_ax=None, **contourf_kwargs):
        """
        Visualizes a parameter grid within the vessel.

        Args:
            param_grid (np.ndarray): The parameter grid to visualize.
            param_name (str, optional): Name of the parameter. Defaults to None.
            draw_traces (bool or list, optional): If True, draws all traces. If a list, draws specified traces. Defaults to False.
            fig_ax (tuple, optional): Tuple of (figure, axis) for the plot. Defaults to None.
            **contourf_kwargs: Additional arguments for the contour plot.

        Returns:
            tuple: The figure and axis of the plot.
        """

        fig, ax = plt.subplots(figsize=(6, 7), tight_layout=True) if fig_ax is None else fig_ax
            
        contourf = ax.contourf(*np.meshgrid(self._r, self._z, indexing='ij'), param_grid, **contourf_kwargs)
        cbar = plt.colorbar(contourf)

        if self._vessel_shape is not None:
            ax.plot(self._separatrix[0], self._separatrix[1], color="m", label="Сепаратриса", linewidth=3)
        if self._separatrix is not None:
            ax.plot(self._vessel_shape[0], self._vessel_shape[1], color="k", label="Вакуумная камера", linewidth=3)
            
        ax.scatter(*self._maxis, marker="x", s=100, color="m")

        ax.axis('scaled')
        ax.set_xlabel("$R$, м")
        ax.set_ylabel("$Z$, м")
        if param_name is not None:
            ax.set_title(param_name)

        traces2draw = self.list_antennae() if draw_traces is True else draw_traces
        if type(traces2draw) is not bool:
            for name in traces2draw:
                trace = self.get_trace(name) 
                ax.plot(trace.r, trace.z, '--w', linewidth=2)

        return fig, ax
    

    def _create_vessel_mask(self):
        """
        Creates a mask for the vessel boundary.

        Returns:
            np.ndarray: Boolean mask indicating points inside the vessel.
        """

        r, z = np.meshgrid(self._r, self._z, indexing='ij')
        r, z = r.flatten(), z.flatten()
        grid = np.vstack((r, z)).T

        path = Path(self._vessel_shape.T)
        mask = path.contains_points(grid)

        return mask.reshape((len(self._r), len(self._z)))
    

    def _warn_if_bad_maxis(self, r_grid, z_grid):
        """
        Warns if the given magnetic axis does not match the axis computed with the given psi_profile.
        """

        grid_maxis_location = (self._psi_profile.argmin() // self._psi_profile.shape[1], self._psi_profile.argmin() % self._psi_profile.shape[1])
        dist_btwn_set_and_grid_maxis = ((r_grid[grid_maxis_location[0]] - self._maxis[0])**2 + (z_grid[grid_maxis_location[1]] - self._maxis[1])**2)**0.5
        diagonal_coordinate_step = ((r_grid[1] - r_grid[0])**2 + (z_grid[1] - z_grid[0])**2)**0.5

        if  dist_btwn_set_and_grid_maxis > diagonal_coordinate_step / 2:
            warnings.warn("given magnetic axis doesn't match the axis computed with given psi_profile.",
                          category=RuntimeWarning)
            

    def _check_view_is_ok(self, antenna, view):
        """
        Checks if the beam is directed towards the magnetic axis.

        Args:
            antenna (tuple): Position of the antenna.
            view (tuple): View direction of the antenna.

        Raises:
            Warning: If the beam is directed away from the magnetic axis.
        """
        view = np.array(view) - (a := np.array(antenna))
        maxis_line = np.array(self._maxis) - a
        if view @ maxis_line <= 0:
            warnings.warn(f'The beam is directed away from the magnetic axis:\nantenna: {antenna}, view: {view}\nmaxis: {self.get_maxis()}')
