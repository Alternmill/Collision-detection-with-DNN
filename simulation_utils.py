import time

import pybullet as p


# Initialize and load simulation and robot
def initialize_simulation():
    print("Initializing simulation...")
    p.connect(p.GUI)


def load_scene_and_robot():
    print("Loading scene and robot...")
    scene_id = p.loadURDF("urdf/moderate_scene.urdf", basePosition=[0, 0, 0], useFixedBase=True)
    robot_id = p.loadURDF("urdf/robot.urdf", basePosition=[0, 0, 1])
    return scene_id, robot_id


# Calculate the closest distance to obstacles
def calculate_closest_distance(scene_id, robot_id, x, y, theta_radians):
    quat = p.getQuaternionFromEuler([0, 0, theta_radians])
    p.resetBasePositionAndOrientation(robot_id, [x, y, 0.5], quat)
    contacts = p.getContactPoints(bodyA=robot_id, bodyB=scene_id)
    collision_detected = any(contact[6][2] > 0.001 for contact in contacts)
    closest_points = p.getClosestPoints(bodyA=robot_id, bodyB=scene_id, distance=1000)
    min_distance = min(closest_points, key=lambda x: x[8])[8] if closest_points else 0
    return 0 if collision_detected or min_distance < 0 else min_distance


def calculate_overall_bounding_box():
    """
    Calculate the overall bounding box encompassing all objects in the PyBullet simulation.

    Returns:
    A tuple of two tuples, each containing two floats:
    - ((min_x, max_x), (min_y, max_y)): The minimum and maximum x and y coordinates of the collective bounding box.
    """
    num_objects = p.getNumBodies()
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    for body_id in range(num_objects):
        num_links = p.getNumJoints(body_id)
        min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
        max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')

        # Include the base link in the calculation
        base_aabb = p.getAABB(body_id, -1)
        min_x, min_y, min_z = base_aabb[0]
        max_x, max_y, max_z = base_aabb[1]

        # Iterate over all links
        for link_index in range(num_links):
            link_aabb = p.getAABB(body_id, link_index)
            # Update the overall bounding box
            min_x = min(min_x, link_aabb[0][0])
            min_y = min(min_y, link_aabb[0][1])
            min_z = min(min_z, link_aabb[0][2])
            max_x = max(max_x, link_aabb[1][0])
            max_y = max(max_y, link_aabb[1][1])
            max_z = max(max_z, link_aabb[1][2])

        return ((min_x, max_x), (min_y, max_y))