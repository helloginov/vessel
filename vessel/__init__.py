import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.interpolate import interpn
from copy import deepcopy
import warnings

from freeqdsk import geqdsk 
# sourse: https://freeqdsk.readthedocs.io/en/stable/geqdsk.html 0.5.0

class Trace:

    def __init__(self, R_grid:np.ndarray, Z_grid:np.ndarray, psi_profile:np.ndarray, 
                 B_toroid_profile:np.ndarray, B_poloid_profile:np.ndarray, maxis_B_value:float,
                 begin:tuple, end:tuple, resolution:int, need_B=True, need_angle=True, need_dist=True, crop_tail=True):
        """Создаёт сечение камеры прямой линией по имеющимся двум точкам antenna и direction.
        Сечение описывается атрибутами _r, _z, _dist, _psi, _B_tor, _B_pol, _reflection_angle

        Args:
            R_grid (np.ndarray): radial coordinates of shape (n)
            Z_grid (np.ndarray): height coordinates of shape (m)
            psi_profile (np.ndarray): poloidal magnetic flux cut distribution, grid of shape (m, n)
            B_toroid_profile (np.ndarray): toroidal magnetic field cut distribution, grid of shape (m, n)
            B_poloid_profile (np.ndarray): poloidal magnetic field cut distribution, grid of shape (m, n)
            maxis_B_value (float): the value of magnetic field on magnetic axes ("center") of B_toroid_profile
            begin (tuple): initial point of the trace in (R, Z) coordinates
            end (tuple): any (R, Z) point in the cut, defining the direction of trace
            resolution (int): number of point to interpolate trace on
            crop_tail: if True, the trace will be cropped by the poloidal flux minimum
        """
        self._begin = begin
        self._end = end

        R_2Dgrid, Z_2Dgrid = np.meshgrid(R_grid, Z_grid, indexing='ij')

        # задание луча в координатах (R, Z)                 с запасом в двое длиннее
        self._r = np.linspace(begin[0], end[0], resolution)
        self._z = np.linspace(begin[1], end[1], resolution)

        # проверка на выход за границы сетки: если луч выходит по какой-то координате за допустимые пределы, то строим новый луч до границы.
        inlying_part = np.argwhere(((self._r > R_2Dgrid[0, 0]) & (self._r < R_2Dgrid[-1, 0]) & (self._z > Z_2Dgrid[0, 0]) & (self._z < Z_2Dgrid[0, -1])))

        if inlying_part.size != 0:
            end_idx = np.array(inlying_part[-1]).item() # inlying_part[-1] может оказаться скаляром, а может массивом с 1 элементом

            if end_idx != len(self._r) - 1:
                self._r = np.linspace(begin[0], self._r[end_idx], resolution)
                self._z = np.linspace(begin[1], self._z[end_idx], resolution)

        # интерполяция двумерного распределения полоидального потока на направление зондирования
        self._psi = self._interpolate_on_trace(R_grid, Z_grid, psi_profile)

        # долой часть луча, идущую после минимума полоидального потока
        if crop_tail:
            self.crop_tail()

        # если появились артефакты в виде отрицательных значений, то зануляем их
        self._psi[self._psi < 0] = 0.0

        # задание луча в кординатах расстояния от антенны
        if need_dist:
            self._dist = np.sqrt((self._r - begin[0])**2 + (self._z - begin[1])**2)

        # интерполяция двумерных распределений тороидального и полоидального полей на направление зондирования
        if need_B:
            self._B_tor = self._interpolate_on_trace(R_grid, Z_grid, B_toroid_profile)
            self._B_pol = self._interpolate_on_trace(R_grid, Z_grid, B_poloid_profile)
            self._B_tor_mode = maxis_B_value
        
        # наклон зондирующего луча к нормалям магнитных поверхностей на протяжении всей длины.
        if need_angle:
            self._reflection_angle = self._trace_angle(R_grid, Z_grid, psi_profile) * 180 / np.pi   # [degree]


    def cut(self, condition=None, inplace=False):
        
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
            trace._B_pol = self._B_pol[condition]
            trace._B_tor = self._B_tor[condition]
        except AttributeError:
            pass
        try:
            trace._reflection_angle = self._reflection_angle[condition]
        except AttributeError:
            pass

        return trace
    

    def crop_tail(self, end_idx=None):
        if end_idx is None:
            end_idx = self._psi[self._psi > 0.0].argmin()
        self.cut(condition=(np.arange(len(self._psi)) <= end_idx), inplace=True)

    
    def _interpolate_on_trace(self, R_grid, Z_grid, profile2d):
        return interpn(
            points=(R_grid, Z_grid), 
            values=profile2d, 
            xi=(self._r, self._z), 
            method="splinef2d"
            )


    def _trace_angle(self, R_grid, Z_grid, psi_profile):
        """
        Считает 2-профиль градиентов Пси^(1/2), интеполирует его на направление зондирования.
        Далее, в каждой точке луча считается угол градиента с направляющим вектором луча (в радианах) и возвращает этот массив

        tol - относительный параметр малости, отвечающий за присвоение градиенту значения 0
        """        
        grad_psi_profile = np.gradient(psi_profile, R_grid, Z_grid, edge_order=2)

        # в каждой точке хранится пара (R_grad, Z_grad) -- градиент Пси в данной точке
        grad_psi_along_trace = np.stack(
            [self._interpolate_on_trace(R_grid, Z_grid, grad_psi_profile[axis]) for axis in (0, 1)],
            axis=0
        )
        # направляющий вектор луча (не нормированный).
        k = np.array([self._r[-1] - self._r[0], self._z[-1] - self._z[0]])

        # длины векторов
        norm_k = np.linalg.norm(k)
        norm_grad = np.linalg.norm(grad_psi_along_trace, axis=0)
        
        angles = np.arccos((-k @ grad_psi_along_trace) / norm_k / norm_grad) * np.sign(np.cross(grad_psi_along_trace, k, axisa=0, axisb=0, axisc=0))

        return angles
    
    @property
    def R(self):        return self._r
    @property
    def Z(self):        return self._z
    @property
    def dist(self):     return self._dist
    @property
    def psi(self):      return self._psi
    @property
    def B_full(self):   return np.sqrt(self._B_tor**2 + self._B_pol**2)
    @property
    def B_tor(self):    return self._B_tor
    @property
    def B_pol(self):    return self._B_pol
    @property
    def reflection_angle(self): return self._reflection_angle
    @property
    def B_mode(self):   return self._B_tor_mode
    @property
    def edges(self): 
        return {
        'begin': self._begin,
        'end': self._end
    }


class Vessel:

    def __init__(self, R_grid, Z_grid, psi_profile, maxis, maxis_mfield_value, toroidal_mfield_profile, poloidal_mfield_profile, vessel_shape=None, separatrix=None):
        """_summary_

        Args:
            R_grid, Z_grid (numpy.ndarray): one-dimensional coordinate grids
            psi_profile (numpy.ndarray): the profile of the normalized poloidal flow (2d-grid)
            maxis (tuple): position of the magnetic axis (R_maxis, Z_maxis)
            maxis_mfield_value (float): magnetic field on the magnetic axis
            toroidal_mfield_profile, poloidal_mfield_profile (numpy.ndarray): profiles of the toroidal and poloidal components of the magnetic field.
            vessel_shape (numpy.ndarray): camera contour, array of shape (V, 2). Defaults to None.
            separatrix (numpy.ndarray): the boundary of the plasma cord, array of shape (S, 2). Defaults to None.

        Raises:
            ValueError: profiles have not the same shapes
        """
        if not psi_profile.shape == toroidal_mfield_profile.shape == poloidal_mfield_profile.shape:
            raise ValueError("profiles have not the same shapes")
        elif psi_profile.shape[0] == psi_profile.shape[1]:
            warnings.warn("grid is square. Be sure if axis 0 relates R-coordinate, axis 1 - Z-coodrinate.", RuntimeWarning)

        self._r = R_grid
        self._z = Z_grid
        self._separatrix = separatrix
        self._vessel_shape = vessel_shape
        self._maxis = maxis

        vessel_mask = ~self._create_vessel_mask()

        self._psi_profile = ma.array(psi_profile, mask=vessel_mask, )
        self._B_poloid_profile = ma.array(poloidal_mfield_profile, mask=vessel_mask, )
        self._B_toroid_profile = ma.array(toroidal_mfield_profile, mask=vessel_mask, )
        self._B_mode = maxis_mfield_value
    
        self._traces = dict()

        self._warn_if_bad_maxis(R_grid, Z_grid)


    def add_antenna(self, launch=None, direction=None, resolution=None, store_as=None, **trace_kwargs):

        # по дефолту луч строится в экваториальной плоскости со стороны слабого поля
        launch =     launch     if launch     is not None else (self._vessel_shape[:, 0].max(), self._maxis[1])
        direction =  direction  if direction  is not None else self._maxis
        resolution = resolution if resolution is not None else len(self._r)

        self._check_direction_is_ok(launch, direction)

        trace = Trace(
            self._r, self._z, self._psi_profile.data, 
            self._B_toroid_profile.data, self._B_poloid_profile.data, self._B_mode,
            launch, direction, resolution, **trace_kwargs
        )
        if store_as is not None:
            self._traces[store_as] = trace
        return trace
    
    @classmethod
    def from_geqdsk(clf, filepath):
        with open(filepath, "r") as f:
            eqdsk = geqdsk.read(f)

        # координатные сетки
        R = np.linspace(eqdsk.rleft, eqdsk.rleft + eqdsk.rdim, eqdsk.nx)
        Z = np.linspace(eqdsk.zmid - eqdsk.zdim / 2, eqdsk.zmid + eqdsk.zdim / 2, eqdsk.ny)

        # градиенты полоидального потока
        psi_grad = np.gradient(eqdsk.psi, R, Z)

        # полоидальная и тороидальная компоненты магнитного поля
        B_pol = (np.linalg.norm(psi_grad, axis=0).T / R).T
        B_tor = np.vstack([eqdsk.bcentr * eqdsk.rcentr / R for _ in range(eqdsk.ny)]).T

        # профиль нормализованного полоидального потока
        psi_norm = 1 - (eqdsk.psi - eqdsk.sibdry) / (eqdsk.simagx - eqdsk.sibdry)

        return clf(
            R_grid=R, 
            Z_grid=Z, 
            psi_profile=psi_norm,
            maxis=np.array([eqdsk.rmaxis, eqdsk.zmaxis]),
            maxis_mfield_value=eqdsk.bcentr,
            toroidal_mfield_profile=B_tor,
            poloidal_mfield_profile=B_pol,
            vessel_shape=np.vstack((eqdsk.rlim, eqdsk.zlim)).T / 100, 
            separatrix=np.vstack((eqdsk.rbdry, eqdsk.zbdry)).T
            )            
       
    
    def list_antennae(self, with_traces=False):
        return self._traces if with_traces else self._traces.keys()
    
    def get_trace(self, name): return self._traces[name]
    
    def get_antenna(self, name):
        edges = self._traces[name].edges
        return {
            'antenna': edges['begin'],
            'direction': edges['end']
        }

    def get_psi(self): return self._psi_profile

    def get_B(self, which='full'):
        if which == 'tor':
            return self._B_toroid_profile
        elif which == 'pol':
            return self._B_poloid_profile
        elif which == 'full':
            return np.sqrt(self._B_toroid_profile**2 + self._B_poloid_profile**2)
        else:
            raise ValueError("argument 'which' can be one of the following: 'full'(by default), 'tor', 'pol'")
    
    def get_maxis(self): return self._maxis

    def get_coord_grids(self): return (self._r, self._z)

    def set_resol(self, name, new_resol): 
        
        if name in self._traces:
            raise NameError("No antenna with such name")
        antenna = self._traces.pop(name).antenna

        return self.add_antenna(name, antenna['antenna'], antenna['direction'], new_resol)


    def visualize_param_in_vessel(self, param_grid, param_name=None, draw_traces=False, fig_ax=None, **contourf_kwargs):
        # plt.ion()
        fig, ax = plt.subplots(figsize=(5, 7), tight_layout=True) if fig_ax is None else fig_ax
            
        contourf = ax.contourf(*np.meshgrid(self._r, self._z, indexing='ij'), param_grid, **contourf_kwargs)
        plt.colorbar(contourf, spacing='proportional')

        if self._vessel_shape is not None:
            ax.plot(self._separatrix[:, 0], self._separatrix[:, 1], color="m", label="сепаратриса", linewidth=4)
        if self._separatrix is not None:
            ax.plot(self._vessel_shape[:, 0],    self._vessel_shape[:, 1], color="k", label="вакуумная камера", linewidth=4)
            
        ax.scatter(*self._maxis, marker="x", s=120, color="m")

        ax.axis('scaled')
        ax.set_xlabel("$R$, м")
        ax.set_ylabel("$Z$, м")
        if param_name is not None:
            ax.set_title(param_name)

        traces2draw = self._traces.keys() if draw_traces is True else draw_traces
        if type(traces2draw) is not bool:
            for name in traces2draw:
                trace = self.get_trace(name) 
                ax.plot(trace.R, trace.Z, '--w', linewidth=2, label=f"Луч {name}")

        return fig, ax
    

    def _create_vessel_mask(self):

        r, z = np.meshgrid(self._r, self._z, indexing='ij')
        r, z = r.flatten(), z.flatten()
        grid = np.vstack((r, z)).T

        path = Path(self._vessel_shape)
        mask = path.contains_points(grid)

        return mask.reshape((len(self._r), len(self._z)))
    

    def _warn_if_bad_maxis(self, R_grid, Z_grid):
        """
        Кидает варнинг, если заданная магнитная ось сильно не совпадает с минимумом полоидального потока.
        """
        grid_maxis_location = (self._psi_profile.argmin() // self._psi_profile.shape[1], self._psi_profile.argmin() % self._psi_profile.shape[1])
        dist_btwn_set_and_grid_maxis = ((R_grid[grid_maxis_location[0]] - self._maxis[0])**2 + (Z_grid[grid_maxis_location[1]] - self._maxis[1])**2)**0.5
        diagonal_coordinate_step = ((R_grid[1] - R_grid[0])**2 + (Z_grid[1] - Z_grid[0])**2)**0.5

        if  dist_btwn_set_and_grid_maxis > diagonal_coordinate_step / 2:
            warnings.warn("given magnetic axis doesn't match the axis computed with given psi_profile.",
                          category=RuntimeWarning)
            

    def _check_direction_is_ok(self, antenna, direction):
        view = np.array(direction) - (a := np.array(antenna))
        maxis_line = np.array(self._maxis) - a
        if view @ maxis_line <= 0:
            warnings.warn(f'The beam is directed away from the magnetic axis:\nantenna: {antenna}, direction: {direction}\nmaxis: {self.get_maxis()}')
        

if __name__ == '__main__':
    
    warnings.filterwarnings('ignore')
    device = Vessel.from_geqdsk("C:/Users/login/python-projects/tref/eqdsk-t15.txt")
    device.add_antenna(store_as=0)
    device.visualize_param_in_vessel(device._psi_profile, draw_traces=True)
    plt.show()