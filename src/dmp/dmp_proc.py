import numpy as np
from typing import Optional, Union, Tuple
from dmp.msg import DMPTraj, DMPPoint
from dmp.srv import LearnDMPFromDemo, LearnDMPFromDemoResponse
from dmp.srv import SetActiveDMP
from dmp.srv import GetDMPPlan
from custom_ros_tools.ros_comm import get_srv_handler

class DMPProcessor:

    def __init__(self):
        self.learn_dmp_from_demo = get_srv_handler('learn_dmp_from_demo', LearnDMPFromDemo, persistent=True)
        self.set_active_dmp = get_srv_handler('set_active_dmp', SetActiveDMP, persistent=True)
        self.get_dmp_plan = get_srv_handler('get_dmp_plan', GetDMPPlan, persistent=True)

    @staticmethod
    def fmt_gains(gains: Union[np.ndarray, float], n: int) -> np.ndarray:
        if isinstance(gains, float):
            gains = gains*np.ones(n)
        return gains.tolist()

    def learn_dmp(
            self,
            t: np.ndarray,
            pos_traj: np.ndarray,
            k_gains: Union[np.ndarray, float],
            d_gains: Union[np.ndarray, float],
            num_bases: int) -> LearnDMPFromDemoResponse:

        # Ensure pos_traj is n-by-N (N is the number of points in demo, n is the ndof)
        N = t.shape[0]
        if pos_traj.shape[0] == N:
            pos_traj = pos_traj.T
        n = pos_traj.shape[0]

        # Handle gains
        k_gains = self.fmt_gains(k_gains, n)
        d_gains = self.fmt_gains(d_gains, n)

        # Create demo trajectory as DMPTraj
        demo = DMPTraj(
            times=t.tolist(),
            points=[DMPPoint(positions=pos_traj[:,i].tolist()) for i in range(N)],
        )

        # Make request from dmp server
        dmp = self.learn_dmp_from_demo(demo, k_gains, d_gains, num_bases)

        return dmp

    def generate_plan(
            self,
            dmp: LearnDMPFromDemoResponse,
            pos0: np.ndarray,
            vel0: np.ndarray,
            t0: float,
            goal: np.ndarray,
            goal_thresh: np.ndarray,
            seg_length: float,
            tau: float,
            dt: float,
            int: integrate_iter) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:

        # Set active dmp
        resp = self.set_active_dmp(dmp.dmp_list)
        if not resp.success:
            rospy.logerr('failed to set active dmp')
            return

        # Plan trajectory
        resp = self.get_dmp_plan(
            pos0.tolist(),
            vel0.tolist(),
            t0,
            goal.tolist(),
            goal_thresh.tolist(),
            seg_length,
            tau,
            dt,
            integrate_iter,
        )

        # Extract data
        success = resp.at_goal == 1

        t = np.array(resp.plan.times)
        pos_traj = np.array([pt.positions for pt in resp.plan.points]).T
        vel_traj = np.array([pt.velocities for pt in resp.plan.points]).T

        return success, t, pos_traj, vel_traj
