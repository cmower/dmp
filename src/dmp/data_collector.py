import rospy
import numpy as np
from typing import Union
from custom_ros_tools.tf import TfInterface

class DMPDataCollector:

    def __init__(self):
        self.t = []
        self.p = []

    def log(self, t: float, p: np.ndarray):
        self.t.append(t)
        self.p.append(p.tolist())

    def get(self):
        t = np.array(self.t)
        pos_traj = np.array(self.p).T
        return t, pos_traj

class TFPositionDMPDataCollector:

    def __init__(self, parent_frame_id, child_frame_id, hz=100):
        self.timer = None
        self.duration = rospy.Duration(1.0/float(hz))
        self.parent_frame_id = parent_frame_id
        self.child_frame_id = child_frame_id
        self.tf = TfInterface()
        self.data_collector = DMPDataCollector()

    def start(self):
        self.timer = rospy.Timer(self.duration, self.main_loop)

    def stop(self):
        self.timer.shutdown()

    def main_loop(self, event):
        tf = self.tf.get_tf_msg(self.parent_frame_id, self.child_frame_id)
        if tf is None: return
        t = rospy.Time.now().to_sec()
        p = self.tf.msg_to_pos(tf)
        self.data_collector.log(t, p)

    def get(self):
        return self.data_collector.get()
