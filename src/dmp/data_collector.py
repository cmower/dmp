import rospy
from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Optional, Tuple
from custom_ros_tools.tf import TfInterface
from scipy.interpolate import interp1d

class DataCollector(ABC):

    @abstractmethod
    def is_empty(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get(self) -> Tuple[np.ndarray]:
        pass

class DMPDataCollector(DataCollector):

    def __init__(self, zero_time: Optional[bool] = True, interpolate: Optional[Union[int, None]] = None):
        self.reset()
        self.zero_time = zero_time
        self.interpolate = interpolate

    def is_empty(self):
        return len(self.t) == 0

    def reset(self):
        self.t = []
        self.p = []

    def log(self, t: float, p: np.ndarray):
        self.t.append(t)
        self.p.append(p.tolist())

    def get(self):
        t = np.array(self.t)
        if self.zero_time:
            t -= t[0]
        pos_traj = np.array(self.p).T

        if isinstance(self.interpolate, int):
            pos_traj_fun = interp1d(t, pos_traj)
            t_out = np.linspace(t[0], t[-1], self.interpolate)
            pos_traj_out = pos_traj_fun(t_out)
        else:
            t_out = t
            pos_traj_out = pos_traj

        return t_out, pos_traj_out

class TFPositionDMPDataCollector(DataCollector):

    def __init__(
            self,
            parent_frame_id: str,
            child_frame_id: str,
            hz: Optional[int] = 100,
            zero_time: Optional[bool] = True,
            collect_x: Optional[bool] = True,
            collect_y: Optional[bool] = True,
            collect_z: Optional[bool] = True,
            interpolate: Optional[Union[int, None]] = None):

        self.timer = None
        self.collect_idx = np.array([collect_x, collect_y, collect_z], dtype=bool)
        self.duration = rospy.Duration(1.0/float(hz))
        self.parent_frame_id = parent_frame_id
        self.child_frame_id = child_frame_id
        self.tf = TfInterface()
        self.data_collector = DMPDataCollector(zero_time=zero_time, interpolate=interpolate)

    def is_empty(self):
        return self.data_collector.is_empty()

    def reset(self):
        self.data_collector.reset()

    def start(self):
        self.timer = rospy.Timer(self.duration, self.main_loop)

    def stop(self):
        self.timer.shutdown()

    def main_loop(self, event):
        tf = self.tf.get_tf_msg(self.parent_frame_id, self.child_frame_id)
        if tf is None: return
        t = rospy.Time.now().to_sec()
        p = self.tf.msg_to_pos(tf)
        self.data_collector.log(t, p[self.collect_idx])

    def get(self):
        return self.data_collector.get()
