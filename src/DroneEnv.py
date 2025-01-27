""" 
Currently this file serves as a test script to experiment with
pybullet / gymnasium and get the perfect environment for our 
use case

Use Case:
We want a roboter which controls a drone from an external device
(mobile phone) which receives the environment via video streaming
and the controlls to the drone

We have to defide this use case in small steps and the first step
is to create an environment with two checkpoints (a, b) and a drone.
The drone has to fly from a to b, wait there for 10 seconds and return
back to a. We will use DRL.

@author Lukas Graf
"""
import time
import random

import pybullet as p
import gymnasium as gym
import pybullet_data

class DroneEnv:
    # Maybe for the drone we have to create a custom Model (URDF)
    # http://wiki.ros.org/urdf/Tutorials/Create%20your%20own%20urdf%20file
    def __init__(self):
        # or p.DIRECT for non-graphical version
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10) # Gravity along x, y, z

        r2d2_id = p.loadURDF("r2d2.urdf")
        num_joints = p.getNumJoints(r2d2_id)
        print(f"Num Joints {num_joints}")
        for idx in range(num_joints):
            print(f"Joint {idx} Info: {p.getJointInfo(r2d2_id, idx)}")

        # For testing
        time.sleep(4)
        p.disconnect()



    def create_checkpoints(self) -> None:
        # Creates a block, can be use for checkpoint a and b
        self.col_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents = [1, 1, 1])
        self.cube_mass = 0

        positions: list = []
        orientation: list = [] # Quaternion
        for i in range(2):
            obj_id = p.createMultiBody(
                        baseMass = self.cube_mass,
                        baseCollisionShapeIndex = self.col_shape_id,
                        basePosition = [
                            random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)
                            ])
            
            pos, ori = p.getBasePositionAndOrientation(obj_id)
            positions.append(pos)
            orientation.append(ori)
        
        # Later use dictionary for information displaying
        print(f"Positions of Cubes: {positions}")
        print(f"Orientation of Cubes: {orientation}")
            

if __name__ == "__main__":
    env = DroneEnv()
    # if -1 -> not connected
    print(env.physicsClient)

