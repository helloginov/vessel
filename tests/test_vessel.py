import unittest
import numpy as np
from vessel import Vessel, Trace

class TestVessel(unittest.TestCase):
    
    def setUp(self):
        # Mock data for Vessel initialization
        self.r_grid = np.linspace(.9, 2.2, 130)
        self.z_grid = np.linspace(-1.2, 1.1, 230)
        self.sp_err = (self.r_grid[1] - self.r_grid[0])
        self.maxis = (1.5, 0.0)

        rr, zz = np.meshgrid(self.r_grid, self.z_grid, indexing="ij")
        self.psi_profile = (rr - self.maxis[0])**2 + ((zz - self.maxis[1]) / 2)**2
        psi_grad = np.gradient(self.psi_profile, self.r_grid, self.z_grid)

        self.maxis_mfield_value = 2.0
        self.b_toroid_profile = np.vstack(
            [
                self.maxis_mfield_value * self.maxis[0] / self.r_grid for _ in range(len(self.z_grid))
            ]
        ).T
        self.b_poloid_profile = (np.linalg.norm(psi_grad, axis=0).T / self.r_grid).T
        self.vessel_shape = np.array([[1.0, 2.0, 2.0, 1.0, 1.0], [-1.0, -1.0, 1.0, 1.0, -1.0]])
        self.separatrix = None

        self.vessel = Vessel(
            r_grid=self.r_grid,
            z_grid=self.z_grid,
            psi_profile=self.psi_profile,
            maxis=self.maxis,
            maxis_mfield_value=self.maxis_mfield_value,
            b_toroid_profile=self.b_toroid_profile,
            b_poloid_profile=self.b_poloid_profile,
            vessel_shape=self.vessel_shape,
            separatrix=self.separatrix,
        )

    def test_vessel_initialization(self):
        self.assertEqual(
            self.vessel.get_maxis(), 
            self.maxis, 
            msg=f"Maxis should be {self.maxis} but got {self.vessel.get_maxis()}"
            )
        self.assertEqual(
            self.vessel.get_coords(), 
            (self.r_grid, self.z_grid), 
            msg=f"Coordinates should be ({self.r_grid}, {self.z_grid}) but got {self.vessel.get_coords()}"
        )
        self.assertTrue(np.all(self.vessel.get_psi2d() == self.psi_profile))
        self.assertTrue(np.all(self.vessel.get_b('tor') == self.b_toroid_profile))
        self.assertTrue(np.all(self.vessel.get_b('pol') == self.b_poloid_profile))


    def test_mask(self):
        psi2d = self.vessel.get_psi2d()
        self.assertTrue(psi2d.mask.sum() != psi2d.size, msg=f"The mask is:\n {psi2d.mask}")


    def test_add_antenna(self):
        antenna_name = "test_antenna"
        self.vessel.add_antenna(store_as=antenna_name)
        self.assertIn(
            antenna_name, 
            self.vessel.list_antennae(), 
            msg=f"Antenna {antenna_name} should be in the list of antennae but got {self.vessel.list_antennae()}"
        )

    def test_get_trace(self):
        antenna_name = "test_antenna"
        self.vessel.add_antenna(store_as=antenna_name)
        trace = self.vessel.get_trace(antenna_name)
        self.assertIsInstance(trace, Trace)

    def test_check_view_is_ok(self):
        with self.assertWarns(Warning):
            self.vessel._check_view_is_ok((1.5, 0.0), (2, 1))

    def test_visualize_param_in_vessel(self):
        try:
            self._init_test_antenna()
            fig, _ = self.vessel.visualize_param_in_vessel(
                self.vessel.get_psi2d(), 
                param_name="Test Psi Profile", 
                draw_traces=True
            )
            fig.savefig("tests/test_psi_profile.png")
        except Exception as e:
            self.fail(f"Visualization failed with exception: {e}")

    def _init_test_antenna(self, pos=(2.0, 1.0), view=(1.5, 0.0)):
        antenna_name = "test_antenna"
        self.vessel.add_antenna(store_as=antenna_name, pos=pos, view=view)
        return self.vessel.get_antenna(antenna_name)

    def test_antenna_rotated_by(self):
        antenna = self._init_test_antenna()
        self.assertAlmostEqual(
            np.degrees(antenna.rotated_by), 
            np.degrees(np.arctan(2)), 
            delta=np.degrees(self.sp_err / np.sqrt(0.5**2 + 1.0**2))
        )

    def test_antenna_rz2xy(self):
        test_point = (1.5, 1.0)
        antenna = self._init_test_antenna()
        x, y = antenna.rz2xy(test_point)
        self.assertAlmostEqual(x, np.cos(np.arctan(0.5)), delta=self.sp_err)
        self.assertAlmostEqual(y, np.sin(np.arctan(0.5)), delta=self.sp_err)

    def test_antenna_xy2rz(self):
        antenna = self._init_test_antenna()
        r, z = antenna.xy2rz((np.cos(np.arctan(0.5)), np.sin(np.arctan(0.5))))
        self.assertAlmostEqual(r, 1.5, delta=self.sp_err)
        self.assertAlmostEqual(z, 1.0, delta=self.sp_err)

    def test_antenna_rad(self):
        antenna = self._init_test_antenna()
        self.assertAlmostEqual(antenna.rad, np.sqrt(0.5**2 + 1.0**2), delta=self.sp_err)

    def test_trace_crop_tail(self):
        antenna = self._init_test_antenna(view=(1, -1))
        trace = antenna.trace
        self.assertAlmostEqual(trace.r[-1], 1.5, delta=self.sp_err)
        self.assertAlmostEqual(trace.z[-1], 0.0, delta=self.sp_err)
        self.assertTrue(np.all(trace.psi > 0.0))

    def test_trace_cut(self):
        antenna = self._init_test_antenna(pos=(2.0, .0))
        trace = antenna.trace
        new_trace = trace.cut(trace.b_tor < 1.5)
        self.assertTrue(np.all(new_trace.b_tor < 1.5), msg=f"Trace b_tor should be less than 1.5 but got {new_trace.b_tor}")


if __name__ == "__main__":
    unittest.main()