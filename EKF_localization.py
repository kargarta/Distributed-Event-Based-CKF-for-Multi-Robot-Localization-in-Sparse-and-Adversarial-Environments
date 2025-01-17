import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.controllers import *
import numpy as np
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import logging
from math import *
from scipy.interpolate import CubicSpline
from math import pi  # or import numpy as np and use np.pi
import pandas as pd


# Configure Logging
logging.basicConfig(level=logging.DEBUG,  # Set the log level
                    format='%(asctime)s - %(levelname)s - %(message)s')  # Set the log format

# Configuration Constants
CONFIG = {
    # Total number of robots in the system.
    "TOTAL_ROBOTS": 10,
    
    # Standard deviation of noise added to control inputs.
    "CONTROL_NOISE_STD": 0.001,
    
    # Covariance matrix representing the process noise in the system.
    "PROCESS_NOISE_COVARIANCE": np.eye(2) * 0.0001,  # 2x2 identity matrix scaled by 0.01.
    
    # Threshold for triggering events in the control strategy.
    "EVENT_TRIGGER_THRESHOLD": 0.001,
    
    # Small constant to avoid division by zero in calculations.
    "EPSILON": 1e-6,
    
    # Standard deviation of noise in range measurements.
    "RANGE_NOISE_STD": 100,
    
    # Standard deviation of noise in bearing measurements.
    "BEARING_NOISE_STD": 100,
    
    # Operational area limits for the x-coordinate.
    "x_limits": (-2, 2),
    
    # Operational area limits for the y-coordinate.
    "y_limits": (-2, 2),
    
    # Dimensions of the target area for the robots.
    "TARGET_AREA": (2, 2),
    
    # Number of steps in the simulation or control loop.
    "num_steps": 170,

    # Time interval for each iteration of the control loop.
    "dt": 0.01,
    
    # Initialize the integral term for the PID controller (2D control).
    "integral_leader": np.zeros(2),
    "integral_follower": np.zeros(2),
    
    # Initialize the previous error for the PID controller (2D control).
    "previous_error_leader": np.zeros(2),
    "previous_error_follower": np.zeros(2),

    # Limit for control input to prevent excessive values.
    "CONTROL_INPUT_LIMIT": 0.8,

    # Minimum safe distance between robots to avoid collisions.
    "SAFE_DISTANCE": 0.01,  # Minimum distance robots should maintain to avoid collisions.

    # Gain for the control barrier function (CBF) to ensure collision avoidance.
    "cbf_gain": 0.1,

    # Index of the leader robot (typically the first robot).
    "LEADER_INDEX": 0,

    # Index of the follower robots (all other robots in the system).
    "FOLLOWER_INDICES": [i for i in range(1, 50)],  # Automatically set based on TOTAL_ROBOTS.

    # Desired trajectory for the leader to follow.
    # This can be dynamically updated during the simulation if necessary.
    "desired_trajectory": np.array([5.0, 5.0]),  # Example initial target position for the leader.
    
    # Placeholder for storing follower indices for easier referencing in follower control.
    "FOLLOWER_INDEX": 1,  # Default for follower; update dynamically during the simulation if needed.

    # Distance threshold for reaching a waypoint.
    "close_enough": 0.01,

    # Regular Sensing range of robots
    "regular_sensing_range": 6,
    
    # Shadow Sensing range of robots
    "shadow_sensing_range": 7, 

    # distance between the wheel
    "b": 0.1,
    
    #Step size 
    "delta_steps": 0.01,

    # Assume the initial position of the leader is known
    "initial_leader_position": np.array([0.1, 0.1])  # Replace with actual initial position if different
}



# Global Variables for Attack Detection
consecutive_large_innovations = 0
attack_threshold = 10.0  # Threshold for detecting an attack
benign_threshold = 15.0   # Threshold for benign detection
steps_for_attack = 1     # Steps needed to confirm an attack
state = "normal"         # Initial state

# Initialize Arrays for Innovations and Adaptive Thresholds
previous_innovations = np.zeros((2, CONFIG["TOTAL_ROBOTS"], CONFIG["num_steps"]))  # Store innovations for each robot across all iterations
adaptive_thresholds = np.full((CONFIG["TOTAL_ROBOTS"], CONFIG["num_steps"]), CONFIG["EVENT_TRIGGER_THRESHOLD"])  # Adaptive thresholds for each robot across all iterations

# Define Range Limits
x_range = CONFIG["x_limits"]
y_range = CONFIG["y_limits"]

# Generate Nearby Positions for Demonstration
nearby_positions = 0.1*np.ones((2, CONFIG["TOTAL_ROBOTS"]))  

# Measurement Noise Covariance
num_nearby = nearby_positions.shape[1]
MEASUREMENT_NOISE_COVARIANCE = np.diag(
    [CONFIG["RANGE_NOISE_STD"]**2]  + 
    [CONFIG["BEARING_NOISE_STD"]**2] 
)

# Define initial positions for all robots
initial_positions = np.array([
    [0.1, -0.85, -0.65, -0.45, -0.85, -0.65, -0.45, -0.85, -0.65, -0.45],
    [0.1, -0.1,  -0.1,  -0.1,   0.1,   0.1,   0.1,   0.3,   0.3,   0.3],
    [0.0,  0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0]
])  # Shape: (3, TOTAL_ROBOTS)

# Define nearby positions deterministically
nearby_positions = initial_positions[:2, :]

# Define positions for multiple time steps deterministically
num_steps = CONFIG["num_steps"]
positions = np.zeros((2, CONFIG["TOTAL_ROBOTS"], CONFIG["num_steps"]))

# Populate positions with a systematic offset over time
positions[:, :, 0] = initial_positions[:2, :]


# Initialize Robotarium Environment
robotarium_env = robotarium.Robotarium(number_of_robots=CONFIG["TOTAL_ROBOTS"], show_figure=True)


# Initialize Predicted State Estimates and Covariance Matrices
x_hat_pred = np.zeros((3, CONFIG["TOTAL_ROBOTS"], CONFIG["num_steps"]))  # State vector: [x, y, theta]

# Initialize error covariance matrices for each robot and each time step
# P will have the shape (TOTAL_ROBOTS, 3, 3, num_steps)
P_self_pred = np.zeros((3, 3, CONFIG["TOTAL_ROBOTS"],  CONFIG["num_steps"]))  # For time-dependent covariance matrices
P_cross_pred = np.zeros((3, 3, CONFIG["TOTAL_ROBOTS"],  CONFIG["num_steps"]))  # For time-dependent covariance matrices


# Fill initial covariance matrices with identity matrices for the first time step
for i in range(CONFIG["TOTAL_ROBOTS"]):
    P_self_pred[:, :, i, 0] = 0.00001*np.eye(3)  # Initial covariance for the first time step
    P_cross_pred[:, :, i, 0] = 0.00001*np.eye(3)  # Initial covariance for the first time step


# Initialize State Estimates and Covariance Matrices
x_hat = np.zeros((3, CONFIG["TOTAL_ROBOTS"], CONFIG["num_steps"]))  # State vector: [x, y, theta]
# Initialize error covariance matrices for each robot and each time step
# P will have the shape (TOTAL_ROBOTS, 3, 3, num_steps)
P_self = np.zeros((3, 3, CONFIG["TOTAL_ROBOTS"],  CONFIG["num_steps"]))  # For time-dependent covariance matrices
P_cross = np.zeros((3, 3, CONFIG["TOTAL_ROBOTS"],  CONFIG["num_steps"]))  # For time-dependent covariance matrices

# Fill initial covariance matrices with identity matrices for the first time step
for i in range(CONFIG["TOTAL_ROBOTS"]):
    P_self[:, :, i, 0] = 0.00001*np.eye(3)  # Initial covariance for the first time step
    P_cross[:, :, i, 0] = 0.00001*np.eye(3)  # Initial covariance for the first time step




# Control inputs (random initial values)
control_inputs = np.random.rand(2, CONFIG["TOTAL_ROBOTS"]) * 0.1  # 2D control inputs (v, omega)

# Initialize leader state
State = 0 



def generate_spline_waypoints(initial_position, num_waypoints=5):
    # Define control points for the spline
    control_points_x = np.linspace(initial_position[0], 1, num_waypoints)
    
    # Generate control points for y using a smooth function (e.g., sine or cosine)
    # This creates a natural-looking path
    control_points_y = 0.2 * np.sin(1 * np.pi * control_points_x)  # Sine wave for smoothness

    # Create a cubic spline using the control points
    cs = CubicSpline(control_points_x, control_points_y)

    # Generate the waypoints from the spline
    x_waypoints = np.linspace(control_points_x[0], control_points_x[-1], num_waypoints * 5)  # More points for smoothness
    y_waypoints = cs(x_waypoints)

    # Ensure the waypoints stay within operational limits
    y_waypoints = np.clip(y_waypoints, -1, 1)

    return np.array([x_waypoints, y_waypoints])


def pure_pursuit_control(current_position, waypoints, look_ahead_distance=0.1):
    """
    Pure Pursuit Controller for tracking waypoints.
    
    Parameters:
    - current_position: (2,) array of the leader's current x, y position
    - waypoints: (2, N) array of x, y coordinates of the path
    - look_ahead_distance: distance to look ahead on the path

    Returns:
    - control_input: velocity and heading angle to reach the target
    """
    x, y = current_position
    target = None

    # Find the look-ahead point
    for waypoint in waypoints.T:
        distance = np.linalg.norm(waypoint - current_position)
        if distance >= look_ahead_distance:
            target = waypoint
            break

    if target is None:
        target = waypoints[:, -1]  # If no point meets the criterion, use the final point

    # Calculate control input
    angle_to_target = np.arctan2(target[1] - y, target[0] - x)
    distance_to_target = np.linalg.norm(target - current_position)
    
    # Basic control policy: adjust speed based on distance and angle
    velocity = 0.5 * distance_to_target  # Proportional speed control
    steering_angle = angle_to_target

    control_input = np.array([velocity, steering_angle])
    return control_input


def stanley_control(current_position, current_heading, waypoints, k=0.5):
    """
    Stanley Controller for path tracking.
    
    Parameters:
    - current_position: (2,) array of the leader's current x, y position
    - current_heading: current orientation of the leader robot (radians)
    - waypoints: (2, N) array of x, y coordinates of the path
    - k: gain for cross-track error correction

    Returns:
    - control_input: velocity and heading angle adjustment
    """
    x, y = current_position
    closest_point, min_dist = None, float('inf')
    closest_idx = 0

    # Find the closest point on the path
    for i, waypoint in enumerate(waypoints.T):
        distance = np.linalg.norm(waypoint - current_position)
        if distance < min_dist:
            min_dist = distance
            closest_point = waypoint
            closest_idx = i

    # Calculate the cross-track error
    cross_track_error = min_dist

    # Calculate the heading error
    path_direction = np.arctan2(
        waypoints[1, closest_idx + 1] - waypoints[1, closest_idx],
        waypoints[0, closest_idx + 1] - waypoints[0, closest_idx]
    )
    heading_error = path_direction - current_heading

    # Stanley control formula
    steering_angle = heading_error + np.arctan(k * cross_track_error)

    # Proportional speed control
    velocity = 3 * np.exp(-abs(steering_angle))  # Reduces speed if steering angle is large

    control_input = np.array([velocity, steering_angle])
    return control_input




def generate_initial_positions(num_robots, x_range, y_range, initial_leader_position):
    """
    Generate initial positions for the robots in a grid pattern around the leader's initial position.

    Parameters:
        num_robots (int): Total number of robots (including the leader).
        x_range (tuple): Operational range for the x-coordinate (min, max).
        y_range (tuple): Operational range for the y-coordinate (min, max).
        initial_leader_position (np.ndarray): Initial position of the leader robot (x, y).
        grid_size (int): The number of rows and columns for the grid layout for followers.

    Returns:
        np.ndarray: An array of shape (3, num_robots) containing the initial positions and orientations.
    """

    grid_size = 3
    
    # Create an array to hold the initial positions of all robots
    initial_positions = np.zeros((3, num_robots))  # 3 rows for [x, y, theta]

    # Set the initial position and orientation of the leader robot (index 0)
    initial_positions[0, 0] = initial_leader_position[0]  # Leader's x position
    initial_positions[1, 0] = initial_leader_position[1]  # Leader's y position
    initial_positions[2, 0] = 0  # Leader's orientation set to 90 degrees (pi/2)

    # Calculate spacing for grid placement
    spacing_x = 0.2*(x_range[1] - x_range[0]) / (grid_size + 1)  # Space between each robot in x-direction
    spacing_y = 0.2*(y_range[1] - y_range[0]) / (grid_size + 1)  # Space between each robot in y-direction

    # Position the followers in a grid-like pattern
    follower_index = 1
    for row in range(grid_size):
        for col in range(grid_size):
            if follower_index < num_robots:
                # Calculate the follower's position based on grid layout
                initial_positions[0, follower_index] = initial_leader_position[0] + (col - grid_size // 2) * spacing_x - 0.75
                initial_positions[1, follower_index] = initial_leader_position[1] + (row - grid_size // 2) * spacing_y 
                initial_positions[2, follower_index] = 0  # Random orientation
                follower_index += 1

    return initial_positions


# EKF Functions

def time_update(x_hat, P_self, Q, control_input, f, F_jacobian, P_cross):
    """
    EKF time update with enhanced numerical stability.
    """
    logging.debug("Performing EKF time update.")

    # Predict state using nonlinear transition function
    x_hat_pred = f(x_hat, control_input).flatten()

    print(x_hat_pred, "x_hat_pred")

    # Calculate Jacobians of f with respect to state and control input
    F_x, F_u = F_jacobian(x_hat, control_input)

    # Predict state covariance
    P_self_pred = F_x @ P_self @ F_x.T + F_u @ Q @ F_u.T

    # Ensure numerical stability
    P_self_pred = (P_self_pred + P_self_pred.T) / 2 

    # Predict cross-covariance
    P_cross_pred = F_x @ P_cross @ F_x.T


    logging.info("Time update completed.")
    return x_hat_pred, P_self_pred, P_cross_pred



def measurement_update(x_hat_pred, P_self_pred, measurement, R, predicted_nearby_positions, P_pred_nearby, nearby_robots, shadow_robots, shadow_positions, index, 
                       previous_innovation, adaptive_threshold, shadow_measurement, shadow_R, H_jacobian, P_cross_pred):

    """
    Performs the EKF measurement update (correction) with event-triggered communication and attack detection.
    """
    logging.debug(f"Performing EKF measurement update for UGV {index}.")

    # Predicted measurement using nonlinear measurement function
    z_hat_pred = measurement_function(x_hat_pred, predicted_nearby_positions, index)

    # Calculate Jacobian of h at the predicted state estimate
    H_r, H_l = H_jacobian(x_hat_pred, predicted_nearby_positions)

    print(H_r, "H_r")


    # Innovation (residual)
    innovation_regular = measurement - z_hat_pred

    logging.debug(f"Measurement residual (innovation): {innovation_regular}")

    # Calculate P_zz by summing over all neighboring robots
    P_zz = np.zeros((2, 2))
    for i in range(len(H_r)):
        P_zz += H_r[i] @ P_self_pred @ H_r[i].T + H_l[i] @ P_cross_pred @ H_l[i].T
    
    # Add the measurement noise covariance
    P_zz += R

    # Check for singularity of P_zz
    if np.linalg.det(P_zz) < 1e-10:
        logging.warning("P_zz is singular. Regularizing.")
        P_zz += CONFIG["EPSILON"] * np.eye(P_zz.shape[0])


    # Tuning Kalman Gain - try scaling by a small factor to avoid over-correction
    P_regular = np.zeros((3, 2))
    for i in range(len(H_r)):
        P_regular += (P_self_pred @ H_r[i].T + P_cross_pred @ H_l[i].T)

    # Compute the Kalman gain
    K_regular = P_regular @ np.linalg.inv(P_zz)
    logging.debug(f"Kalman gain K_regular: {K_regular.shape}")

    # Update the state estimate
    update = np.zeros((3,))
    for i in range(len(H_r)):
        index = i * 2
        if index + 2 <= len(innovation_regular):  # Avoid out of bounds
            update += K_regular @ innovation_regular[index:index + 2]
    
    # Updated state estimate
    x_hat_updated = x_hat_pred + update
    logging.debug(f"Updated state estimate: {x_hat_updated.shape}")

    print(x_hat_updated, "x_hat_updated")

    # Update covariance estimate
    P_self_updated = P_self_pred - K_regular @ P_zz @ K_regular.T
    P_self_updated = (P_self_updated + P_self_updated.T) / 2 + CONFIG["EPSILON"] * np.eye(P_self_updated.shape[0])

    # If shadow measurements are available, incorporate them
    #if shadow_measurement.size > 0:
        # Shadow measurement update (similar to regular measurements)
        #shadow_z_hat_pred = measurement_function(x_hat_pred, shadow_positions, index)
        #Hr_shadow, Hl_shadow = H_jacobian(x_hat_pred, shadow_positions)

        #shadow_innovation = shadow_measurement - shadow_z_hat_pred
        #P_zz_shadow = Hr_shadow[0] @ P_pred_nearby @ Hr_shadow[0].T + shadow_R

        # Regularize P_zz_shadow if ill-conditioned
        #if np.linalg.cond(P_zz_shadow) > 1e10:
            #P_zz_shadow += CONFIG["EPSILON"] * np.eye(P_zz_shadow.shape[0])

        #K_shadow = 0.5 * P_pred_nearby @ Hr_shadow[0].T @ np.linalg.inv(P_zz_shadow)

        # Incorporate shadow measurement update
        #x_hat_updated += K_shadow @ shadow_innovation
        #P_self_updated -= K_shadow @ P_zz_shadow @ K_shadow.T

    # Ensure the updated covariance is positive definite and symmetric
    P_self_updated = (P_self_updated + P_self_updated.T) / 2 

    # Event-triggered communication and attack detection
    event_trigger, updated_threshold = event_triggered(
            innovation_regular, previous_innovation, adaptive_threshold
        )

    logging.info(f"Event triggered for UGV {index}. Communicating state to neighbors.")
    return x_hat_updated, P_self_updated, innovation_regular, updated_threshold




def F_jacobian(previousPose, control_input):
    """
    Computes the Jacobians G_mut and G_ut.
    G_mut: Partial derivative of the current pose w.r.t previous pose (3x3 matrix).
    G_ut:  Partial derivative of the current pose w.r.t control input (3x2 matrix).
    
    Args:
        deltaSteps (float): Linear displacement.
        previousPose (numpy array): Previous pose [x, y, theta] (1x3 matrix).
        deltaTheta (float): Angular displacement.
    
    Returns:
        list: [G_mut, G_ut], where G_mut is 3x3 and G_ut is 3x2.
    """
    previousTheta = previousPose[2]  # Extract previous orientation (theta)
    b = CONFIG["b"]  # Wheelbase or distance between wheels
    deltaTheta = (control_input[1] - control_input[0]) / (2 * CONFIG["b"])
    
    deltaSteps = (control_input[1] + control_input[0]) / 2


    # Initialize G_mut (Jacobian w.r.t. previous pose)
    F_x = np.array([[1, 0, -deltaSteps * np.sin(previousTheta + deltaTheta)],
                      [0, 1,  deltaSteps * np.cos(previousTheta + deltaTheta)],
                      [0, 0, 1]])

    # Initialize G_ut (Jacobian w.r.t. control input)
    DS = deltaSteps
    DT = deltaTheta + previousTheta  # Total angle change

    G_ut = np.array([[ DS * np.sin(DT) + b * np.cos(DT), -DS * np.sin(DT) + b * np.cos(DT)],
                     [-DS * np.cos(DT) + b * np.sin(DT),  DS * np.cos(DT) + b * np.sin(DT)],
                     [-2, 2]])

    # Normalize G_ut by (2 * b)
    F_u = (1 / (2 * b)) * G_ut

    # Return the Jacobians
    return [F_x, F_u]





def H_jacobian(current_pose, nearby_pose):
    """
    Computes the Jacobians H_r and H_l for the measurement model with numerical stability checks.
    """
    rx, ry = current_pose[0], current_pose[1]
    H_r_list = []
    H_l_list = []

    for nearby in nearby_pose:
        lx, ly = nearby[0], nearby[1]
        q = np.sqrt((lx - rx) ** 2 + (ly - ry) ** 2 + CONFIG["EPSILON"])

        # Calculate Jacobians
        H_r = (1 / q) * np.array([
            [-(lx - rx), -(ly - ry), 0],
            [(ly - ry) / q, -(lx - rx) / q, -q]
        ])
        H_l = (1 / q) * np.array([
            [lx - rx, ly - ry, 0],
            [-(ly - ry) / q, (lx - rx) / q, 0]
        ])

        H_r_list.append(H_r)
        H_l_list.append(H_l)


    return np.array(H_r_list), np.array(H_l_list)





def state_transition(previousPose, ut):
    """
    Predicts the next state of a 2-wheeled robot.
    """
    previousX, previousY, previousTheta = previousPose
    DL, DR = ut

    # New state estimates
    currentX = previousX + ((DR + DL) / 2) * np.cos(previousTheta + ((DR - DL) / (2 * CONFIG["b"])))
    currentY = previousY + ((DR + DL) / 2) * np.sin(previousTheta + ((DR - DL) / (2 * CONFIG["b"])))
    currentTheta = previousTheta + (DR - DL) / CONFIG["b"]

    # Normalize orientation
    currentTheta = normalizeAngle(currentTheta)

    return np.array([currentX, currentY, currentTheta])


def normalizeAngle(originalAngle):
    """
    Normalizes angle to the range [-pi, pi].

    Args:
        originalAngle (float): Input angle in radians.

    Returns:
        float: Angle normalized to the range [-pi, pi].
    """
    # Normalize to [0, 2*pi)
    normalizedAngle = originalAngle % (2 * np.pi)

    # Shift to [-pi, pi] if necessary
    if normalizedAngle.all() > np.pi:
        normalizedAngle -= 2 * np.pi

    return normalizedAngle

#def state_transition(x, u):
    """Example state transition function."""
    #x_new = np.copy(x)
    #x_new[0] += u[0] * np.cos(x[2])  # x-position update
    #x_new[1] += u[0] * np.sin(x[2])  # y-position update
    #x_new[2] += u[1]  # Theta update
    #return x_new

def control_barrier_function(robot_position, other_positions, min_distance):
    """
    Computes the control barrier function value between one robot and all other robots.
    
    Parameters:
        robot_position (np.ndarray): Position of the robot (shape: (2,)).
        other_positions (np.ndarray): Positions of other robots (shape: (2, n), where n is the number of other robots).
        min_distance (float): Minimum safe distance to maintain from other robots.
        
    Returns:
        np.ndarray: Squared distance differences from the minimum safe distance.
    """
    # Compute squared distances between the robot and each position in other_positions
    distance_squared = np.sum((robot_position[:, np.newaxis] - other_positions)**2, axis=0)
    return distance_squared - min_distance**2


def apply_cbf_control(original_control_input, robot_position, nearby_positions, min_distance, config):
    """
    Modifies the control input using Control Barrier Function to avoid collisions.
    
    Parameters:
        original_control_input (np.ndarray): The original control input.
        robot_position (np.ndarray): Position of the robot (shape: (2,)).
        nearby_positions (np.ndarray): Positions of nearby robots (shape: (2, N)).
        min_distance (float): Minimum allowable distance to avoid collision.
        config (dict): Configuration dictionary containing control parameters.
        
    Returns:
        np.ndarray: Modified control input, clipped to stay within specified limits.
    """
    h_ij = control_barrier_function(robot_position, nearby_positions, min_distance)

    # If the barrier function is violated (too close), adjust the control input
    if (h_ij < 0).any():
        # Compute the direction for repulsive force
        direction = (robot_position[:, np.newaxis] - nearby_positions)  # Shape (2, N)

        # Normalize the direction vector to avoid division by zero
        norm = np.linalg.norm(direction, axis=0) + 1e-6  # Shape (N,)
        normalized_direction = direction / norm  # Shape (2, N)

        # Calculate the repulsive force based on the normalized direction
        repulsive_force = config["cbf_gain"] * normalized_direction / norm  # Shape (2, N)

        # Sum the repulsive forces and apply to the original control input
        original_control_input += np.sum(repulsive_force, axis=1)  # Shape (2,)

    # Ensure the control input stays within limits
    return np.clip(original_control_input, -config["CONTROL_INPUT_LIMIT"], config["CONTROL_INPUT_LIMIT"])


def leader_control_policy(current_position, current_heading, positions, min_distance, config, waypoints):
    """
    Control policy for the leader robot with Pure Pursuit and CBF for collision avoidance.
    
    Parameters:
        current_position (np.ndarray): Current position of the leader robot.
        target_position (np.ndarray): Target position for the leader robot.
        positions (np.ndarray): Positions of all robots (shape: (2, N)).
        min_distance (float): Minimum safe distance.
        config (dict): Configuration parameters.
        waypoints (np.ndarray): Waypoints for Pure Pursuit controller.
        
    Returns:
        np.ndarray: Control input for the leader robot.
    """
    # Pure Pursuit control to track the waypoints
    #control_input = pure_pursuit_control(current_position, target_position)
    control_input = stanley_control(current_position, current_heading, waypoints)

    # Check distances to all other robots and apply CBF if necessary
    for i in range(1, config["TOTAL_ROBOTS"]):  # Avoid checking itself
        control_input = apply_cbf_control(control_input, current_position, positions[:, i], min_distance, config)
    
    return control_input



def follower_control_policy(current_position, current_heading, desired_trajectory, positions, min_distance, config):
    """
    Control policy for follower robots using Stanley Controller with Control Barrier Function to avoid collisions.
    
    Parameters:
        current_position (np.ndarray): Current position of the follower robot (x, y).
        current_heading (float): Current heading angle of the follower robot (in radians).
        desired_trajectory (np.ndarray): Desired trajectory (e.g., leader's position).
        positions (np.ndarray): Positions of all robots (shape: (2, N)).
        min_distance (float): Minimum safe distance.
        config (dict): Configuration parameters.
        
    Returns:
        np.ndarray: Control input for the follower robot [steering_angle, velocity].
    """
    # Extract the follower's desired position and heading (goal position)
    target_position = desired_trajectory[:2]  # Leader's x, y position
    target_heading = np.arctan2(target_position[1] - current_position[1],
                                target_position[0] - current_position[0])  # Heading towards the leader

    # Compute the cross-track error (distance from current position to the desired trajectory)
    cross_track_error = np.linalg.norm(current_position - target_position)
    
    # Stanley control law for steering angle
    k = 50 # Gain for the Stanley controller
    heading_error = target_heading - current_heading  # Difference in heading
    steering_angle = heading_error + np.arctan2(k * cross_track_error, 0.4)

    # Ensure steering angle is within the range [-pi, pi]
    steering_angle = np.arctan2(np.sin(steering_angle), np.cos(steering_angle))

    # Define the control input as [steering_angle, velocity]
    control_input = np.array([steering_angle, 0.4])

    # Check distances to all other robots and apply CBF if necessary
    for i in range(config["TOTAL_ROBOTS"]):
        if i != config["FOLLOWER_INDEX"]:  # Ensure we're not checking the current follower
            control_input = apply_cbf_control(control_input, current_position, positions[:, i], min_distance, config)
    
    return control_input

def pid_control(current_position, target_position, config, robot_type):
    """
    General PID control logic used for both leader and follower robots.
    
    Parameters:
        current_position (np.ndarray): Current position of the robot.
        target_position (np.ndarray): Target position for the robot.
        config (dict): Configuration parameters.
        robot_type (str): Type of robot ("leader" or "follower").
        
    Returns:
        np.ndarray: Control input calculated from the PID controller.
    """
    # Retrieve configuration parameters for the specific robot type
    integral = config[f"integral_{robot_type}"]
    previous_error = config[f"previous_error_{robot_type}"]

    # Calculate the error
    error = target_position - current_position

    # Proportional term
    proportional = config[f"kp_{robot_type}"] * error

    # Integral term
    integral += config[f"ki_{robot_type}"] * error * config["dt"]

    # Derivative term
    derivative = config[f"kd_{robot_type}"] * (error - previous_error) / config["dt"]

    # Calculate control input
    control_input = proportional + integral + derivative

    # Update previous error in config
    config[f"previous_error_{robot_type}"] = error

    # Limit control input to a specified range
    return np.clip(control_input, -config["CONTROL_INPUT_LIMIT"], config["CONTROL_INPUT_LIMIT"])


def apply_process_noise(control_input, config):
    """
    Applies process noise to the control input.
    
    Parameters:
        control_input (np.ndarray): Control input before noise application.
        config (dict): Configuration parameters for noise application.
        
    Returns:
        np.ndarray: Noisy control input.
    """
    noise = np.random.normal(0, config["CONTROL_NOISE_STD"], size=control_input.shape)
    control_input_noisy = control_input + noise
    return control_input_noisy

def range_measurement(robot_position, neighbor_positions):
    """
    Calculate the range measurement between the UGV's current position and nearby positions.
    """

    # Check if there are any nearby positions
    if nearby_positions.shape[0] == 0:
        # If there are no nearby positions, return an empty array or handle accordingly
        return np.array([])  # Returning an empty array or you can handle this case as needed

    ranges_to_nearby = np.linalg.norm(neighbor_positions[:, :2] - robot_position[:2], axis=1)   # Shape: (N,)
    ranges = np.concatenate([ranges_to_nearby])  # Combine distances
    return ranges


def bearing_measurement(robot_position, neighbor_positions):
    """
    Calculates the bearing measurements to nearby UGVs and the initial position.
    """

    # Check if there are any nearby positions
    if neighbor_positions.shape[0] == 0:
        # If there are no nearby positions, return an empty array or handle accordingly
        return np.array([])  # Returning an empty array or handle this case as needed

    vectors_to_nearby = neighbor_positions[:, :2] - robot_position[:2]  # Shape: (N, 2)
    angles_to_nearby = np.arctan2(vectors_to_nearby[:, 1], vectors_to_nearby[:, 0])
    bearings = angles_to_nearby - robot_position[2]

    bearings = np.arctan2(np.sin(bearings), np.cos(bearings))
    bearings=normalizeAngle(bearings)
    return bearings





def calculate_distances_and_errors(positions):
    """
    Calculates distances and errors between robots based on their positions.

    Parameters:
        positions: Array of positions for all robots.

    Returns:
        distances: Dictionary of distances between robots.
        errors: Dictionary of error measurements for each pair of robots.
    """
    distances = {}
    errors = {}
    num_robots = positions.shape[1]

    for i in range(num_robots):
        for j in range(i + 1, num_robots):
            dist = np.linalg.norm(positions[:, i] - positions[:, j])
            distances[(i, j)] = dist
            distances[(j, i)] = dist  # Symmetric
            errors[(i, j)] = np.random.normal(0, 0.1)  # Add Gaussian noise for errors
            errors[(j, i)] = errors[(i, j)]  # Symmetric

    return distances, errors

def compute_rho(positions, threshold=1.0):
    """
    Computes a measure of closeness (rho) based on robot positions.

    Parameters:
        positions: Array of positions for all robots.
        threshold: Distance threshold for defining closeness.

    Returns:
        rho: Float value indicating the closeness measure.
    """
    rho = 0.0
    num_robots = positions.shape[1]

    for i in range(num_robots):
        for j in range(num_robots):
            if i != j:
                dist = np.linalg.norm(positions[:, i] - positions[:, j])
                if dist > 0 and dist < threshold:  # Avoid division by zero
                    rho += 1.0 / dist  # Closer robots contribute more to rho

    return rho

def shadow_range_measurement(positions, distances, errors, rho):
    """
    Calculate the shadow range measurement considering noise and distance constraints.
    
    :param positions: A dictionary of node positions {'v_i': (x, y), ...}
    :param distances: A dictionary of known distances {'d_ij': distance, ...}
    :param errors: A dictionary of errors {'eta_ij': error, ...}
    :param rho: Minimum distance threshold for shadow edges
    :return: A list of shadow range measurements and their uncertainties
    """
    shadow_ranges = []
    
    for (i, j) in distances.keys():
        d_ih = distances.get((i, 'h'), 0)  # Distance between node i and anchor h
        d_jh = distances.get((j, 'h'), 0)  # Distance between node j and anchor h
        d_ij = distances.get((i, j), 0)    # Distance between nodes i and j
        
        eta_ih = errors.get((i, 'h'), 0)   # Measurement error for i-h distance
        eta_jh = errors.get((j, 'h'), 0)   # Measurement error for j-h distance

        # Check conditions for shadow edges
        if d_ij <= rho or d_ij > 2 * rho:
            continue  # Not a valid shadow edge
        
        # Calculate angles using the cosine rule
        if d_jh > 0 and d_ih > 0:  # Ensure valid distances
            alpha_khj = np.arccos((d_jh**2 + d_ih**2 - d_ij**2) / (2 * d_jh * d_ih))
        
            # Calculate estimated shadow edge length using the law of cosines
            d_ij_estimated = np.sqrt(d_ih**2 + d_jh**2 - 2 * d_ih * d_jh * np.cos(alpha_khj))

            # Adjust noise for shadow range estimation
            eta_ij = (
                2 * d_jh * d_ih * np.cos(alpha_khj) +
                eta_jh + eta_ih -
                2 * np.sqrt((d_jh**2 + eta_jh) * (d_ih**2 + eta_ih)) * np.cos(alpha_khj)
            )
            
            shadow_ranges.append((d_ij_estimated, eta_ij))  # Append as a tuple
    
    return shadow_ranges






def measurement_function(robot_position, neighbor_positions, index):
    """
    Constructs a measurement vector from the ground truth.
    """
    
    ranges = range_measurement(robot_position, neighbor_positions)
    bearings = bearing_measurement(robot_position, neighbor_positions)
    
    measurement = np.concatenate([ranges, bearings])
    return measurement


def event_triggered(innovation, prev_innovation, threshold=None, decay_factor=0.0):
    """
    Checks if the event-triggered condition is met with an adaptive threshold.
    """
    # Handle the threshold input
    if isinstance(threshold, np.ndarray):
        threshold = threshold if np.any(threshold) else CONFIG["EVENT_TRIGGER_THRESHOLD"]
    else:
        threshold = threshold or CONFIG["EVENT_TRIGGER_THRESHOLD"]

    # Calculate the norms
    innovation_norm = np.linalg.norm(innovation)

    # Use np.maximum to handle array types for adaptive_threshold
    adaptive_threshold = np.maximum(threshold, decay_factor * np.linalg.norm(prev_innovation))

    logging.debug(f"Innovation norm: {innovation_norm}, Adaptive threshold: {adaptive_threshold}")

    # Event triggering logic: handle multiple thresholds
    if isinstance(adaptive_threshold, np.ndarray):
        return innovation_norm > adaptive_threshold, adaptive_threshold
    else:
        return innovation_norm > adaptive_threshold, adaptive_threshold





def attack_detected(innovation, threshold=0.0):
    # Ensure this variable is initialized before use
    global consecutive_large_innovations
    if 'consecutive_large_innovations' not in globals():
        consecutive_large_innovations = 0
    
    # Check if the innovation is large
    if np.linalg.norm(innovation) > threshold:
        consecutive_large_innovations += 1
    else:
        consecutive_large_innovations = 0
    
    # Define the condition for attack detection
    return consecutive_large_innovations > steps_for_attack



def create_laplacian_matrix(total_robots, ground_truth):
    """Create the Laplacian matrix based on ground_truth and sensing range."""
    
    # Initialize the adjacency matrix
    A = np.zeros((total_robots, total_robots))
    shadow = True

    # Fill the adjacency matrix based on the sensing range and shadow
    for i in range(total_robots):
        for j in range(total_robots):
            if i != j:  # Exclude self
                distance = np.linalg.norm(ground_truth[:, i] - ground_truth[:, j])
                
                # Regular neighbors based on regular sensing range
                if distance < CONFIG["regular_sensing_range"]:
                    A[i, j] = 1
                
                # Shadow neighbors if the flag is True and within shadow sensing range
                if shadow and CONFIG["shadow_sensing_range"] is not None and distance < CONFIG["shadow_sensing_range"]:
                    A[i, j] = 1

    # Compute the degree matrix D
    D = np.diag(np.sum(A, axis=1))  # Sum along rows to get the degree of each node

    # Laplacian matrix L = D - A
    L = D - A

    return L


# Assuming you have a function to initialize robots
def initialize_robots(r, leader_index, total_robots):
    # Colors: Red for leader, Green for followers
    colors = [np.array([1, 0, 0])] + [np.array([0, 1, 0])] * (total_robots - 1)
    
    # Initialize each robot with its respective color
    for i in range(total_robots):
        r.set_color(i, colors[i])

def determine_font_size(r, base_font_size):
    """
    Adjusts font size based on the environment size and robot count.

    Parameters:
    - r: Robotarium environment object, which may have attributes related to plotting dimensions.
    - base_font_size: The default or base font size to start from.

    Returns:
    - Adjusted font size.
    """
    # Adjust font size based on the number of robots or other factors
    adjusted_font_size = base_font_size
    
    # Example: if many robots, reduce font size for readability
    if r.number_of_robots > 10:  # Assumes r has an attribute for total robot count
        adjusted_font_size = max(8, base_font_size - 2)  # Reduce font size, minimum of 8
    
    return adjusted_font_size

def update_heading(current_heading, steering_angle, dt = 0.1):
    """
    Update the heading based on the steering angle and time delta.
    
    Parameters:
        current_heading (float): The current heading of the robot.
        steering_angle (float): The steering angle output from the control policy.
        dt (float): Time step for the simulation.
        
    Returns:
        float: Updated heading.
    """
    # Update the heading (ensure the heading stays within [0, 2*pi])
    new_heading = current_heading + steering_angle * dt
    new_heading = new_heading % (2 * np.pi)  # Wrap around if necessary
    return new_heading

def initialize_headings(num_robots, initial_leader_heading=0):
    """
    Initialize headings (orientations) for each robot in the team.

    Parameters:
        num_robots (int): Total number of robots (including the leader).
        initial_leader_heading (float): Initial heading angle of the leader in radians.

    Returns:
        np.ndarray: An array of shape (num_robots,) containing the initial headings.
    """
    # Initialize headings array
    headings = np.zeros(num_robots)

    # Set the leader's initial heading
    headings[0] = initial_leader_heading  # For example, facing upwards (90 degrees)

    # Set initial headings for followers
    for i in range(1, num_robots):
        # Initialize followers' headings (e.g., random, or facing the leader)
        headings[i] = initial_leader_heading + np.random.uniform(-np.pi / 8, np.pi / 8)

    return headings

def topological_neighbors(L, agent, ground_truth):
    """ 
    Returns the regular and shadow neighbors of a particular agent using the graph Laplacian.
    
    L: NxN numpy array (representing the graph Laplacian)
    agent: int (agent: 0 - N-1)
    regular_sensing_range: float (distance for regular neighbors)
    shadow_sensing_range: float (distance for shadow neighbors)
    ground_truth: Nx3 numpy array (positions [x, y, theta] of robots)

    -> 1xM numpy arrays (regular_neighbors, shadow_neighbors)
    """
    # Validate the inputs
    assert isinstance(L, np.ndarray), f"In the topological_neighbors function, the graph Laplacian (L) must be a numpy ndarray. Received type {type(L).__name__}."
    assert isinstance(agent, int), f"In the topological_neighbors function, the agent number (agent) must be an integer. Received type {type(agent).__name__}."
    assert isinstance(CONFIG["regular_sensing_range"], (int, float)), "Sensing ranges must be numeric."
    assert isinstance(CONFIG["shadow_sensing_range"], (int, float)), "Sensing ranges must be numeric."
    
    # Validate agent index
    assert agent >= 0, f"In the topological_neighbors function, the agent number (agent) must be greater than or equal to zero. Received {agent}."
    assert agent < L.shape[0], f"In the topological_neighbors function, the agent number (agent) must be within the dimension of the provided Laplacian (L). Received agent number {agent} and Laplacian size {L.shape[0]} by {L.shape[1]}."

    # Extract the row corresponding to the agent
    row = L[agent, :]

    # Set the self-connection to zero (since the agent is not its own neighbor)
    row[agent] = 0

    # Initialize lists for regular and shadow neighbors
    regular_neighbors = []
    shadow_neighbors = []

    # Loop through all other agents to determine neighbors
    for j in range(L.shape[0]):
        if j != agent:
            # Calculate Euclidean distance between agents i and j
            dist = np.linalg.norm(ground_truth[:, agent] - ground_truth[:, j])

            # Regular neighbor: within the regular sensing range
            if dist < CONFIG["regular_sensing_range"]:
                regular_neighbors.append(j)
            # Shadow neighbor: within the shadow sensing range but outside regular range
            elif dist < CONFIG["shadow_sensing_range"]:
                shadow_neighbors.append(j)
                
    return np.array(regular_neighbors), np.array(shadow_neighbors)

def run_simulation(robotarium_env, CONFIG, x_hat, P_self, P_cross, positions,
                   apply_process_noise, time_update, measurement_function, 
                   measurement_update, attack_detected, topological_neighbors, event_triggered, 
                   generate_initial_positions, range_measurement, bearing_measurement, 
                   shadow_range_measurement,  # New shadow measurement functions
                   leader_control_policy, follower_control_policy, 
                   plot_final_states, do_s_attack_probability, 
                   fdi_attack_probability, max_dos_robots, max_fdi_measurements):
    """
    Runs the simulation of UGVs in the Robotarium environment with limited attacks.

    Parameters:
        robotarium_env: The Robotarium environment instance.
        CONFIG: Configuration parameters for the simulation.
        x_hat: Estimated states of the robots.
        P: Covariance matrices for state estimation.
        positions: Current positions of the robots.
        target_positions: Desired target positions for the robots.
        apply_process_noise: Function to apply process noise to control inputs.
        time_update: Function for the time update step in the Kalman filter.
        measurement_function: Function to calculate the expected measurement.
        measurement_update: Function for the measurement update step in the Kalman filter.
        attack_detected: Function to check if an attack is detected based on the innovation.
        get_neighbors: Function to get neighbors of a robot.
        event_triggered: Function to check if event-triggered communication is needed.
        generate_initial_positions: Function to generate initial robot positions.
        range_measurement: Function for range measurements.
        bearing_measurement: Function for bearing measurements.
        shadow_range_measurement: Function for shadow edge range measurements.
        leader_control_policy: Function defining the leader's control input policy.
        follower_control_policy: Function defining the followers' control input policy.
        animate: Function to run the animation of the simulation.
        plot_final_states: Function to plot the final states of the robots.
        do_s_attack_probability: Probability of a DoS attack occurring.
        fdi_attack_probability: Probability of an FDI attack occurring.
        max_dos_robots: Maximum number of robots to be affected by DoS attack.
        max_fdi_measurements: Maximum number of measurements to be corrupted by FDI attack.
    """
    logging.info("Starting the simulation.")
    ground_truth_positions = []
    estimated_positions = []
    control_command = []
    Event_Trigger = []
    Attack_Detection = []
    Neighbors = []

    initial_positions = generate_initial_positions(CONFIG["TOTAL_ROBOTS"], x_range, y_range, CONFIG["initial_leader_position"])
    x_hat_pred[:, :, 0] = initial_positions
    x_hat[:, :, 0] = initial_positions

    r = robotarium.Robotarium(number_of_robots=CONFIG["TOTAL_ROBOTS"], show_figure=True, initial_conditions=initial_positions, sim_in_real_time=True)
    _,uni_to_si_states = create_si_to_uni_mapping()
    si_to_uni_dyn = create_si_to_uni_dynamics()
    si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
    leader_controller = create_si_position_controller(velocity_magnitude_limit=CONFIG["CONTROL_INPUT_LIMIT"])


    # For computational/memory reasons, initialize the velocity vector
    dxi = np.zeros((2,CONFIG["TOTAL_ROBOTS"]))


    leader_index = 0  # Leader is robot 0
    d_min = CONFIG["SAFE_DISTANCE"]  # Minimum safe distance to avoid collisions


    ground_truth = r.get_poses()


    leader_index = 0  # Leader is robot 0
    follower_indices = [i for i in range(1, CONFIG["TOTAL_ROBOTS"]) if i < ground_truth.shape[1]] 
    d_min = CONFIG["SAFE_DISTANCE"]  # Minimum safe distance to avoid collisions


    # Initialize distances, errors, and rho
    distances, errors = calculate_distances_and_errors(positions)
    rho = compute_rho(positions)

    
    # Initialize empty lists for connection indices
    rows = []
    cols = []

    # Populate rows and cols based on neighbors
    for i in range(CONFIG["TOTAL_ROBOTS"]):
        L = create_laplacian_matrix(CONFIG["TOTAL_ROBOTS"], ground_truth)
        neighbors, _ = topological_neighbors(L, i, ground_truth)
        for neighbor in neighbors:
            rows.append(i)        # Current robot index
            cols.append(neighbor)  # Neighbor robot index

    # Convert rows and cols to numpy arrays for easier indexing
    rows = np.array(rows)
    cols = np.array(cols)

    # Initialize line objects and leader label
    line_leader, = r.axes.plot(
        [ground_truth[0, leader_index], ground_truth[0, follower_indices[0]]],
        [ground_truth[1, leader_index], ground_truth[1, follower_indices[0]]],
        linewidth=0.1, color='r', zorder=-1
    )

    leader_label = r.axes.text(
        ground_truth[0, leader_index], ground_truth[1, leader_index] + 0.15, "Leader",
        fontsize=12, color='k', fontweight='bold',
        horizontalalignment='center', verticalalignment='center', zorder=0
    )

    # Initialize follower lines and labels
    follower_labels = []
    line_followers = []  # Initialize a list to hold Line2D objects for followers

    for jj, follower_index in enumerate(follower_indices):
        if follower_index >= ground_truth.shape[1]:
            print(f"Follower index {follower_index} is out of bounds for x with size {ground_truth.shape[1]}")
            continue

        # Create lines for each follower's connections
        neighbors, _ =  topological_neighbors(L, i, ground_truth)
        for neighbor in neighbors:
            if neighbor < ground_truth.shape[1]:
                line_follower, = r.axes.plot(
                    [ground_truth[0, follower_index], ground_truth[0, neighbor]],
                    [ground_truth[1, follower_index], ground_truth[1, neighbor]],
                    linewidth=0.1, color='b', zorder=-1
                )
                line_followers.append(line_follower)  # Store reference to the line object

        # Add label for each follower
        follower_label = r.axes.text(
            ground_truth[0, follower_index], ground_truth[1, follower_index] + 0.15, f"Follower {jj + 1}",
            fontsize=12, color='b', fontweight='bold',
            horizontalalignment='center', verticalalignment='center', zorder=0
        )
        follower_labels.append(follower_label)

    # Plot waypoints in the Robotarium
    # Generate waypoints
    waypoints = generate_spline_waypoints(CONFIG["initial_leader_position"])

    # Set the target position as the last waypoint
    target_position = waypoints[:, -1]

    # Create the plot using your custom r object
    # Assuming `r` is your plot object that has axes property
    waypoint_lines = r.axes.plot(waypoints[0], waypoints[1], 'g--', linewidth=2.5, label='Leader Path', zorder=-2)

    # Mark the initial position
    initial_marker = r.axes.plot(CONFIG["initial_leader_position"][0], CONFIG["initial_leader_position"][1], 'ro', label='Initial Position')
    r.axes.text(CONFIG["initial_leader_position"][0], CONFIG["initial_leader_position"][1] + 0.05, 'Initial Position', 
             horizontalalignment='center', color='black')

    # Mark the target position
    target_marker = r.axes.plot(target_position[0], target_position[1], 'bo', label='Target Position')
    r.axes.text(target_position[0], target_position[1] + 0.05, 'Target Position', 
             horizontalalignment='center', color='black')

    # Additional plot settings if needed
    r.axes.set_xlim(-2, 2)
    r.axes.set_ylim(-2, 2)
    r.axes.set_title('Robot Path with Waypoints')
    r.axes.set_xlabel('X Coordinate')
    r.axes.set_ylabel('Y Coordinate')
    r.axes.axhline(0, color='black', linewidth=0.5, ls='--')
    r.axes.axvline(0, color='black', linewidth=0.5, ls='--')
    r.axes.grid()
    r.axes.legend()
    r.axes.legend(loc='upper left', bbox_to_anchor=(0.2, 0.2))



    r.step()  # Update Robotarium environment


    # Simulation Loop
    for step in range(CONFIG["num_steps"]):  

        # Step the Robotarium environment
        ground_truth = r.get_poses()
        xi = uni_to_si_states(ground_truth)

        initial_positions = generate_initial_positions(CONFIG["TOTAL_ROBOTS"], x_range, y_range, CONFIG["initial_leader_position"])

        # Recalculate distances, errors, and rho at each step
        distances, errors = calculate_distances_and_errors(ground_truth)
        rho = compute_rho(ground_truth)

        # Determine robots affected by DoS attack based on probability
        dos_attacked_robots = []
        for i in range(CONFIG["TOTAL_ROBOTS"]):
            if np.random.rand() < do_s_attack_probability:
                dos_attacked_robots.append(i)
        
        dos_attacked_robots = dos_attacked_robots[:max_dos_robots]  # Limit to max_dos_robots
        logging.info(f"Robots affected by DoS attack: {dos_attacked_robots}")

        for i in range(CONFIG["TOTAL_ROBOTS"]):   
            
            L = create_laplacian_matrix(CONFIG["TOTAL_ROBOTS"], ground_truth)
            neighbors, shadow_neighbors = topological_neighbors(L, i, ground_truth)
            logging.info(f"Neighbors generated for robot {i}: {neighbors}")

            # Get positions of nearby UGVs (neighbors only)
            nearby_positions = np.array([ground_truth[:, j] for j in neighbors])

            # Compute regular range and bearing measurements
            range_meas = range_measurement(ground_truth[:, i], nearby_positions)
            bearing_meas = bearing_measurement(ground_truth[:, i], nearby_positions)

            # Compute shadow edge measurements
            shadow_neighbors_positions_predict = np.array([x_hat_pred[:2, j, step] for j in shadow_neighbors])
            shadow_ranges = {}  # Correctly initialize as a dictionary
            for j in range(CONFIG["TOTAL_ROBOTS"]):
            # Replace with actual range calculation logic
                shadow_ranges[(i, j)] = (np.random.uniform(1, 10), None)  # Example measurement

            # Later in your code where the error occurred
            shadow_range_meas = [shadow_ranges.get((i, j), (0, 0))[0] for j in shadow_neighbors]

            # Check for FDI attack affecting limited measurements
            fdi_attacked_robots = []
            if np.random.rand() < fdi_attack_probability:
                # Randomly select a limited number of measurements to corrupt
                fdi_indices = np.random.choice(len(range_meas), size=min(max_fdi_measurements, len(range_meas)), replace=False)
                for idx in fdi_indices:
                    range_meas[idx] += np.random.normal(0, 0.5)  # Corrupt range measurement
                    bearing_meas[idx] += np.random.normal(0, 0.5)  # Corrupt bearing measurement

                    # Track which robots are affected by the FDI attack
                    if idx < CONFIG["TOTAL_ROBOTS"]:  # Ensure the index is valid for the number of robots
                        fdi_attacked_robots.append(idx)

            combined_measurements = np.concatenate([range_meas, bearing_meas])
            logging.info(f"Combined measurements generated for robot {i}: {combined_measurements}")

            # Initialize state before the loop if it's not already initialized
            if "state" not in locals():
                state = 0  # Assuming 0 is the starting state or the initial index for waypoints

            headings_leader = 0.0
            headings_follower = initialize_headings(CONFIG["TOTAL_ROBOTS"])
            # Leader control logic
            if i == CONFIG["LEADER_INDEX"]:
                leader_position = xi[:, CONFIG["LEADER_INDEX"]]
                target_position = waypoints[:,state].reshape((2,))
                # Define and update the current heading of the leader robot
                current_heading = headings_leader
                control_input = leader_control_policy(leader_position, current_heading, xi, d_min, CONFIG, waypoints)
                #plt.scatter(ground_truth[0, i], ground_truth[1, i], color='red', label='Leader' if i == CONFIG["LEADER_INDEX"] else "")
                
                # Assuming control_input contains steering angle and velocity
                steering_angle = control_input[0]  # Extract steering angle from control input
                dt = 0.1  # Define time step for the simulation
        
                # Update the leader's heading
                headings_leader= update_heading(current_heading, steering_angle, dt)

                # Check if the leader has reached its target waypoint
                if np.linalg.norm(leader_position - target_position) < CONFIG["close_enough"]:
                    state = (state + 1) % waypoints.shape[1]  # Update state to the next waypoint

            else:
                # Follower control logic
                follower_position = xi[:, i]  # Current position of the follower
                follower_heading = headings_follower[i]  # Current heading of the follower
                desired_trajectory = xi[:, CONFIG["LEADER_INDEX"]]  # Follower aims to follow the leader

                # Call the follower control policy using Stanley Controller
                control_input = follower_control_policy(follower_position, follower_heading, desired_trajectory, xi, d_min, CONFIG)
                
                # Assuming control_input contains [steering_angle, velocity]
                steering_angle, velocity = control_input  # Extract steering angle and velocity

                # Update follower's heading based on steering angle and velocity
                dt = 0.1  # Define time step for the simulation
                headings_follower[i] = update_heading(follower_heading, steering_angle, dt)
                # Plot followers
                #plt.scatter(ground_truth[0, i], ground_truth[1, i], color='green', label='Follower' if i == 0 else "")

            # Apply process noise to the control input for both leader and followers
            dxi[:, i] = apply_process_noise(control_input, CONFIG)  # Apply noise to follower inputs
            if i == CONFIG["LEADER_INDEX"]:
                dxi[:, CONFIG["LEADER_INDEX"]] = apply_process_noise(control_input, CONFIG)  # Apply noise to leader input

            font_size_m = 12
            # Assume num_followers is defined to be the number of followers you expect
            line_follower = [plt.Line2D([], []) for _ in range(len(follower_indices))]  # Initialize the Line2D objects

            # Update follower positions and lines
            for q, follower_label in enumerate(follower_labels):
                # Update follower label position and font size
                follower_label.set_position([xi[0, follower_indices[q]], xi[1, follower_indices[q]] + 0.15])
                follower_label.set_fontsize(determine_font_size(r, font_size_m))
                

                # Ensure the index is valid for line_followers
                if q < len(line_followers):

                    # Check if the current follower is affected by DoS or FDI
                    if follower_indices[q] in dos_attacked_robots or follower_indices[q] in fdi_attacked_robots:
                        line_color = 'r'  # Set color to black if affected
                        line_width = 0.1   # Thicker line for affected robots
                    else:
                        line_color = 'b'  # Set to blue if not affected
                        line_width = 0.1   # Default line width for unaffected robots

                    # Update the data for the follower's line using valid indices
                    if q < len(rows) and q < len(cols):
                        line_followers[q].set_data(
                            [ground_truth[0, rows[q + 1]], ground_truth[0, cols[q + 1]]],
                            [ground_truth[1, rows[q + 1]], ground_truth[1, cols[q + 1]]],
                        )
                         # Change color of follower line
                        line_followers[q].set_color(line_color)
                        line_followers[q].set_linewidth(line_width)

            # Update leader position and line
            leader_label.set_position([xi[0, leader_index], xi[1, leader_index] + 0.15])
            leader_label.set_fontsize(determine_font_size(r, font_size_m))

            # Change color of leader line
            line_leader.set_color(line_color)
            line_leader.set_linewidth(line_width)

            # Check if the leader is affected by DoS or FDI
            if leader_index in dos_attacked_robots or leader_index in fdi_attacked_robots:
                line_color = 'r'  # Set color to black if affected
                line_width = 0.1  # Thicker line for affected leader
            else:
                line_color = 'k'  # Set to red if not affected
                line_width = 0.1  # Default line width for unaffected leader

            # Update the data for the leader's line
            line_leader.set_data(
                [ground_truth[0, leader_index], ground_truth[0, follower_indices[0]]],
                [ground_truth[1, leader_index], ground_truth[1, follower_indices[0]]]
            )

            

            # Plot the waypoints if needed
            waypoint_lines[0].set_xdata(waypoints[0])
            waypoint_lines[0].set_ydata(waypoints[1])

            control_input_noisy = apply_process_noise(control_input, CONFIG)
           

            # Ensure control_input_noisy is 2D before setting velocities
            if control_input_noisy.ndim == 1:
                control_input_noisy = control_input_noisy[:, np.newaxis]

            if step >= 170:
                control_input_noisy = [0,0]  # Set the last 5 samples to zero for all robots
                x_hat[:, i, step-1] = x_hat[:, i, 170]
            
            

            # Check for DoS attack on specific robots
            if i in dos_attacked_robots:
                logging.warning(f"DoS attack detected on robot {i}. Local state estimates blocked.")
                # Block predicted state and increase uncertainty
                x_hat_pred[:, i, step] = np.zeros_like(x_hat_pred[:, i, step])  
                P_self_pred[:, :, i, step] = np.eye(len(P_self_pred[:, :, i, step])) * 1e5  
                P_cross_pred[:, :, i, step] = np.eye(len(P_cross_pred[:, :, i, step])) * 1e5  

            else:
                # Prediction step if no DoS attack
                x_hat_pred[:, i, step], P_self_pred[:, :, i, step], P_cross_pred[:, :, i, step]  = time_update(
                    x_hat[:, i, step - 1], P_self[:, :, i, step - 1], 
                    CONFIG["PROCESS_NOISE_COVARIANCE"], control_input_noisy, 
                    state_transition, F_jacobian, P_cross[:, :, i, step - 1]
                )

            # Count the number of nearby robots excluding the current robot i
            number_of_nearby_robots = len([j for j in neighbors if j != i])
        
            number_of_shadow_robots = len(shadow_neighbors) 

            MEASUREMENT_NOISE_COVARIANCE = np.diag(
               [CONFIG["RANGE_NOISE_STD"]**2]  + 
               [CONFIG["BEARING_NOISE_STD"]**2] 
            )


            # Prepare shadow measurements and covariance
            shadow_measurement = np.concatenate([shadow_range_meas])  # Include shadow measurements
            shadow_R = np.diag(
               [CONFIG["RANGE_NOISE_STD"]**2] * (number_of_shadow_robots ) + 
               [CONFIG["BEARING_NOISE_STD"]**2] * (number_of_shadow_robots)
            )
            
            nearby_positions_predict = np.array([x_hat_pred[:, j, step] for j in neighbors])  
            P_pred_nearby = np.array([P_self_pred[:, :, j, step] for j in neighbors]) 


                # Call the measurement_update function with all required arguments
            x_hat_updated, P_self_updated, previous_innovation, updated_threshold = measurement_update(
                    x_hat_pred[:, i, step], 
                    P_self_pred[:, :, i, step],
                    combined_measurements, 
                    MEASUREMENT_NOISE_COVARIANCE, 
                    nearby_positions_predict,
                    P_pred_nearby,
                    number_of_nearby_robots, 
                    number_of_shadow_robots,
                    shadow_neighbors_positions_predict,
                    i, 
                    previous_innovations[:, i, step-1], 
                    adaptive_thresholds[i],
                    shadow_measurement,  # Include shadow measurements
                    shadow_R,  # Include covariance for shadow measurements
                    H_jacobian,
                    P_cross_pred[:, :, i, step]

                )


            # Store the updated values back to the arrays
            x_hat[:, i, step] = x_hat_updated
            P_self[:, :, i, step] = P_self_updated
            # If previous_innovation is not guaranteed to be (3,), handle it
            if previous_innovation.size == 2:
                # You may want to append a zero or some default value
                previous_innovation = np.append(previous_innovation, 0)
            elif previous_innovation.size > 3:
                previous_innovation = previous_innovation[:3] 
            adaptive_thresholds[i] = updated_threshold
 
            positions[:, i, step] = x_hat[0:2, i, step]  # Update position based on the estimated state
  
            # Update estimated positions for the next step
            ground_truth_positions.append(ground_truth[:, i])
            estimated_positions.append(x_hat[:, i, step])
            control_command.append(control_input[:, np.newaxis]) 
            Neighbors.append(neighbors)


            
        norms = np.linalg.norm(dxi, 2, 0)
        magnitude_limit = 0.15
        idxs_to_normalize = (norms > magnitude_limit)
        dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]
        # Apply control barriers, convert to unicycle, and set velocities
        dxi = si_barrier_cert(dxi, ground_truth[:2, :])
        dxu = si_to_uni_dyn(dxi, ground_truth)
        if step >= 170:
             dxu[:, :] = 0  # Set the last 5 samples to zero for all robots

       
        r.set_velocities(np.arange(CONFIG["TOTAL_ROBOTS"]), dxu)  # Update velocities for all robots
            # Step the simulation forward
        r.step()
    
    ground_truth_positions = np.array(ground_truth_positions)
    estimated_positions = np.array(estimated_positions)
    control_command = np.squeeze(np.array(control_command))
    Event_Trigger = np.squeeze(np.array(Event_Trigger))
    Attack_Detection = np.squeeze(np.array(Attack_Detection))
    Neighbors = np.squeeze(np.array(Neighbors))

    # Finalizing the simulation
    logging.debug(f"Final step {step}: current positions {positions}")
    plot_final_states(ground_truth_positions, estimated_positions, control_command,  CONFIG["TOTAL_ROBOTS"], CONFIG["num_steps"])
    #plot_event_trigger_and_attack_detection(Event_Trigger, Attack_Detection, CONFIG["TOTAL_ROBOTS"], CONFIG["num_steps"], Neighbors)
    input("Press Enter to close...")  # Pause until user input
    # Clean up the Robotarium environment
    logging.info("Simulation completed.")
    #Call at end of script to print debug information and for your script to run on the Robotarium server properly
    r.call_at_scripts_end()



def plot_final_states(ground_truth, estimates, control_command, num_robots, samples_per_robot):
    """Plot the trajectories and final positions of UGVs for ground truth, estimates, and control commands."""

    # Create subplots for trajectories
    nrows = int(np.ceil(num_robots / 2))  # Number of rows needed
    ncols = 2  # Fixed number of columns for the layout
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

    # Define colors for each robot using a colormap
    colors = plt.cm.viridis(np.linspace(0, 1, num_robots))  # Create a colormap with distinct colors

    # Loop through each robot to create individual subplots for trajectories
    for i in range(num_robots):
        start_index = i * samples_per_robot
        end_index = start_index + samples_per_robot
        
        # Plot ground truth trajectory
        axes[i].plot(ground_truth[start_index:end_index, 0], 
                     ground_truth[start_index:end_index, 1], 
                     color=colors[i], alpha=0.5, label='Ground Truth')
        axes[i].scatter(ground_truth[end_index - 1, 0], 
                        ground_truth[end_index - 1, 1], 
                        color=colors[i], marker='o', label='Final Ground Truth Position')

        # Plot estimated trajectory
        axes[i].plot(estimates[start_index:end_index, 0], 
                      estimates[start_index:end_index, 1], 
                      color=colors[i], linestyle='--', alpha=0.5, label='Estimates')
        axes[i].scatter(estimates[end_index - 1, 0], 
                        estimates[end_index - 1, 1], 
                        color=colors[i], marker='x', label='Final Estimate Position')

        # Set titles and labels for each subplot
        if i == 0:
            axes[i].set_title('Leader Robot Trajectory', fontsize=14)
        else:
            axes[i].set_title(f'Follower Robot {i}', fontsize=14)

        axes[i].set_xlabel("X Position", fontsize=12)
        axes[i].set_ylabel("Y Position", fontsize=12)
        axes[i].grid()
        axes[i].axis('equal')
        axes[i].legend()

    # Handle any unused subplots if the number of robots is not a multiple of 2
    for j in range(num_robots, nrows * ncols):
        fig.delaxes(axes[j])  # Remove any empty subplots

    plt.tight_layout(pad=6.0)
    plt.subplots_adjust(top=0.90, hspace=0.8, wspace=0.6)
    plt.suptitle("Event-based EKF for Multi-Robot Localization in Sparse Sensing Graph and Adversarial Environment \n UGV Trajectories and Final Positions", fontsize=16)
    plt.show()

    # Create a new figure for X localization with respect to sample time
    time = np.arange(samples_per_robot)  # Sample time based on number of samples
    fig, axes_x = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
    axes_x = axes_x.flatten()

    # Loop through each robot to create subplots for X coordinates
    for i in range(num_robots):
        start_index = i * samples_per_robot
        end_index = start_index + samples_per_robot
        
        # Plot X coordinate vs sample time
        axes_x[i].plot(time, ground_truth[start_index:end_index, 0], color=colors[i], alpha=0.5, label='Ground Truth X')
        axes_x[i].plot(time, estimates[start_index:end_index, 0], linestyle='--', color=colors[i], alpha=0.5, label='Estimate X')

        # Set titles and labels for each subplot
        if i == 0:
            axes_x[i].set_title('Leader Robot X Coordinate over Time', fontsize=14)
        else:
            axes_x[i].set_title(f'Follower Robot {i} X Coordinate over Time', fontsize=14)

        axes_x[i].set_xlabel("Sample Time", fontsize=12)
        axes_x[i].set_ylabel("X Position", fontsize=12)
        axes_x[i].grid()
        axes_x[i].legend()

    # Handle any unused subplots if the number of robots is not a multiple of 2
    for j in range(num_robots, nrows * ncols):
        fig.delaxes(axes_x[j])  # Remove any empty subplots

    plt.tight_layout(pad=6.0)
    plt.subplots_adjust(top=0.90, hspace=0.8, wspace=0.6)
    plt.suptitle("Event-based EKF for Multi-Robot Localization in Sparse Sensing Graph and Adversarial Environment \n X Coordinate Localization over Time", fontsize=16)
    plt.show()

    # Create a new figure for Y localization with respect to sample time
    fig, axes_y = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
    axes_y = axes_y.flatten()

    # Loop through each robot to create subplots for Y coordinates
    for i in range(num_robots):
        start_index = i * samples_per_robot
        end_index = start_index + samples_per_robot
        
        # Plot Y coordinate vs sample time
        axes_y[i].plot(time, ground_truth[start_index:end_index, 1], color=colors[i], alpha=0.5, label='Ground Truth Y')
        axes_y[i].plot(time, estimates[start_index:end_index, 1], linestyle='--', color=colors[i], alpha=0.5, label='Estimate Y')

        # Set titles and labels for each subplot
        if i == 0:
            axes_y[i].set_title('Leader Robot Y Coordinate over Time', fontsize=14)
        else:
            axes_y[i].set_title(f'Follower Robot {i} Y Coordinate over Time', fontsize=14)

        axes_y[i].set_xlabel("Sample Time", fontsize=12)
        axes_y[i].set_ylabel("Y Position", fontsize=12)
        axes_y[i].grid()
        axes_y[i].legend()

    # Handle any unused subplots if the number of robots is not a multiple of 2
    for j in range(num_robots, nrows * ncols):
        fig.delaxes(axes_y[j])  # Remove any empty subplots

    plt.tight_layout(pad=6.0)
    plt.subplots_adjust(top=0.90, hspace=0.8, wspace=0.6)
    plt.suptitle("Event-based EKF for Multi-Robot Localization in Sparse Sensing Graph and Adversarial Environment \n Y Coordinate Localization over Time", fontsize=16)
    plt.show()

    # Calculate and plot mean square error (MSE) for localization
    mse = np.zeros((num_robots, samples_per_robot))
    for i in range(num_robots):
        start_index = i * samples_per_robot
        end_index = start_index + samples_per_robot
        
        # Calculate MSE for each sample
        mse[i, :] = np.mean((ground_truth[start_index:end_index, :2] - estimates[start_index:end_index, :2])**2, axis=1)

    # Create a new figure for localization error
    fig, axes_mse = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
    axes_mse = axes_mse.flatten()

    # Loop through each robot to create subplots for MSE
    for i in range(num_robots):
        axes_mse[i].plot(time, mse[i, :], color=colors[i], label='Localization Error (MSE)', alpha=0.7)
        
        # Set titles and labels for each subplot
        if i == 0:
            axes_mse[i].set_title('Leader Robot Localization Error (MSE)', fontsize=14)
        else:
            axes_mse[i].set_title(f'Follower Robot {i} Localization Error (MSE)', fontsize=14)

        axes_mse[i].set_xlabel("Sample Time", fontsize=12)
        axes_mse[i].set_ylabel("Mean Square Error", fontsize=12)
        axes_mse[i].grid()
        axes_mse[i].legend()

    # Handle any unused subplots if the number of robots is not a multiple of 2
    for j in range(num_robots, nrows * ncols):
        fig.delaxes(axes_mse[j])  # Remove any empty subplots

    plt.tight_layout(pad=6.0)
    plt.subplots_adjust(top=0.90, hspace=0.8, wspace=0.6)
    plt.suptitle("Event-based EKF for Multi-Robot Localization in Sparse Sensing Graph and Adversarial Environment \n Localization Error in Mean Square (MSE)", fontsize=16)
    plt.show()

    # Calculate and plot the average localization error (MSE) across all robots
    avg_mse = np.mean(mse, axis=0)
    
    fig_avg_mse, ax_avg_mse = plt.subplots(figsize=(12, 6))
    ax_avg_mse.plot(time, avg_mse, color='blue', label='Average Localization Error (MSE)', alpha=0.7)
    
    ax_avg_mse.set_title('Event-based EKF for Multi-Robot Localization in Sparse Sensing Graph and Adversarial Environment \n Average Localization Error (MSE) Across Robots', fontsize=16)
    ax_avg_mse.set_xlabel('Sample Time', fontsize=14)
    ax_avg_mse.set_ylabel('Mean Square Error', fontsize=14)
    ax_avg_mse.grid()
    ax_avg_mse.legend()
    
    plt.tight_layout()
    plt.show()

    # Create a new figure for control commands
    fig, axes_control = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
    axes_control = axes_control.flatten()

    # Loop through each robot to create subplots for control commands
    for i in range(num_robots):
        start_index = i * samples_per_robot
        end_index = start_index + samples_per_robot
        
        # Extract control commands
        linear_velocity = control_command[start_index:end_index, 0]
        angular_velocity = control_command[start_index:end_index, 1]

        # Plot linear velocity
        axes_control[i].plot(time, linear_velocity, color='green', label='Linear Velocity', alpha=0.7)
        if i == 0:
            axes_control[i].set_title('Leader Robot Control Commands', fontsize=14)
        else:
            axes_control[i].set_title(f'Follower Robot {i} Control Commands', fontsize=14)
        axes_control[i].set_xlabel("Sample Time", fontsize=12)
        axes_control[i].set_ylabel("Velocity", fontsize=12)
        axes_control[i].grid()

        # Create a second y-axis for angular velocity
        ax2 = axes_control[i].twinx()
        ax2.plot(time, angular_velocity, color='orange', label='Angular Velocity', alpha=0.7)
        ax2.set_ylabel("Angular Velocity", fontsize=12)
        
        # Combine legends from both axes
        axes_control[i].legend(loc='upper left')
        ax2.legend(loc='upper right')

    # Handle any unused subplots if the number of robots is not a multiple of 2
    for j in range(num_robots, nrows * ncols):
        fig.delaxes(axes_control[j])  # Remove any empty subplots

    # Handle any unused subplots if the number of robots is not a multiple of 2
    for j in range(num_robots, nrows * ncols):
        fig.delaxes(axes_control[j])  # Remove any empty subplots
    
        # Save ground truth and estimates to CSV
    for i in range(num_robots):
        start_index = i * samples_per_robot
        end_index = start_index + samples_per_robot
        
        data = {
            "Time": np.arange(samples_per_robot),
            "GroundTruth_X": ground_truth[start_index:end_index, 0],
            "GroundTruth_Y": ground_truth[start_index:end_index, 1],
            "Estimate_X": estimates[start_index:end_index, 0],
            "Estimate_Y": estimates[start_index:end_index, 1],
        }
        df = pd.DataFrame(data)
        df.to_csv(f"robot_{i}_ground_truth_estimates.csv", index=False)

    for i in range(num_robots):
        start_index = i * samples_per_robot
        end_index = start_index + samples_per_robot
        
        data = {
            "Time": np.arange(samples_per_robot),
            "LinearVelocity": control_command[start_index:end_index, 0],
            "AngularVelocity": control_command[start_index:end_index, 1],
        }
        df = pd.DataFrame(data)
        df.to_csv(f"robot_{i}_control_commands.csv", index=False)

        # Save MSE for each robot
    for i in range(num_robots):
        data = {
            "Time": np.arange(samples_per_robot),
            "MSE": mse[i, :],
        }
        df = pd.DataFrame(data)
        df.to_csv(f"robot_{i}_mse.csv", index=False)

    # Save average MSE across robots
    data = {
        "Time": np.arange(samples_per_robot),
        "AverageMSE": avg_mse,
    }
    df_avg_mse = pd.DataFrame(data)
    df_avg_mse.to_csv("average_mse.csv", index=False)


    plt.tight_layout(pad=6.0)
    plt.subplots_adjust(top=0.90, hspace=0.8, wspace=0.6)
    plt.suptitle("Event-based EKF for Multi-Robot Localization in Sparse Sensing Graph and Adversarial Environment \n Control Commands for Robots", fontsize=16)
    plt.show()



def plot_event_trigger_and_attack_detection(event_trigger_data, attack_detection_data, num_robots, samples_per_robot, neighbors):
    """Plot event trigger and attack detection data for each robot considering neighbors.
    
    Args:
        event_trigger_data: Array of shape (samples_per_robot * num_robots,) containing event trigger data for each robot and each sample.
        attack_detection_data: Array of shape (samples_per_robot * num_robots,) containing attack detection data for each robot and each sample.
        neighbors: Array of shape (samples_per_robot * num_robots,), where each element is an array indicating the neighbors for each robot at each time step.
        samples_per_robot: Integer, number of time samples per robot.
        num_robots: Integer, total number of robots.
    """
    
    num_samples = len(event_trigger_data)
    
    colors = ['red', 'green', 'blue', 'purple', 'magenta', 'orange', 'cyan', 'brown', 'pink', 'gray']

    # Plot event trigger data with neighbors
    fig_event, ax_event = plt.subplots(figsize=(10, 6))
    for i in range(num_samples):
        robot = (i // samples_per_robot) + 1  # Calculate robot number based on sample index
        time_step = i % samples_per_robot     # Time step within each robot's sample
        
        # Plot the event trigger for the main robot
        if np.any(event_trigger_data[i] == 1):
            ax_event.scatter(time_step, robot, color=colors[(robot - 1) % len(colors)], alpha=0.7)

        # Plot event triggers for each neighbor of this robot at this time step
        for neighbor in neighbors[i]:
            if np.any(event_trigger_data[i] == 1):
                ax_event.scatter(time_step, neighbor, color=colors[(neighbor - 1) % len(colors)], marker='o', alpha=0.5, edgecolors='black')

    ax_event.set_title('Triggering Instants of Robots (Event Trigger)', fontsize=16)
    ax_event.set_xlabel('Time, k', fontsize=14)
    ax_event.set_ylabel('Robot Number', fontsize=14)
    ax_event.grid(True, linestyle='--', alpha=0.5)
    ax_event.set_yticks(range(1, num_robots + 1))
    ax_event.set_xlim(0, samples_per_robot - 1)  # Capture whole sample time
    plt.tight_layout()
    plt.show()

    # Plot attack detection data with neighbors
    fig_attack, ax_attack = plt.subplots(figsize=(10, 6))
    for i in range(num_samples):
        robot = (i // samples_per_robot) + 1  # Calculate robot number based on sample index
        time_step = i % samples_per_robot     # Time step within each robot's sample

        # Plot the attack detection for the main robot
        if np.any(attack_detection_data[i] == 1):
            ax_attack.scatter(time_step, robot, color=colors[(robot - 1) % len(colors)], marker='x', alpha=0.7)

        # Plot attack detections for each neighbor of this robot at this time step
        for neighbor in neighbors[i]:
            if np.any(attack_detection_data[i] == 1):
                ax_attack.scatter(time_step, neighbor, color=colors[(neighbor - 1) % len(colors)], marker='x', alpha=0.5, edgecolors='black')

    ax_attack.set_title('Triggering Instants of Robots (Attack Detection)', fontsize=16)
    ax_attack.set_xlabel('Time, k', fontsize=14)
    ax_attack.set_ylabel('Robot Number', fontsize=14)
    ax_attack.grid(True, linestyle='--', alpha=0.5)
    ax_attack.set_yticks(range(1, num_robots + 1))
    ax_attack.set_xlim(0, samples_per_robot - 1)  # Capture whole sample time
    plt.tight_layout()
    plt.show()



# Call the run_simulation function to start
if __name__ == "__main__":
    # Define distances, errors, and rho before calling run_simulation
    # Initialize distances (e.g., with a placeholder array or a calculation)
    distances = np.zeros((CONFIG["TOTAL_ROBOTS"], CONFIG["TOTAL_ROBOTS"]))
    
    # Initialize errors 
    errors = np.zeros((CONFIG["TOTAL_ROBOTS"], CONFIG["TOTAL_ROBOTS"]))

    # Define rho
    rho = np.zeros(CONFIG["TOTAL_ROBOTS"])

    # Call the run_simulation function to start
if __name__ == "__main__":
    run_simulation(
        robotarium_env=robotarium_env,
        CONFIG=CONFIG,
        x_hat=x_hat,
        P_self=P_self,
        P_cross = P_cross,
        positions=positions,
        apply_process_noise=apply_process_noise,
        time_update=time_update,
        measurement_function=measurement_function,
        measurement_update=measurement_update,
        attack_detected=attack_detected,
        topological_neighbors=topological_neighbors,
        event_triggered=event_triggered,
        generate_initial_positions=generate_initial_positions,
        range_measurement=range_measurement,
        bearing_measurement=bearing_measurement,
        shadow_range_measurement=shadow_range_measurement,
        leader_control_policy=leader_control_policy,
        follower_control_policy=follower_control_policy,
        plot_final_states=plot_final_states,
        do_s_attack_probability=0.0,  # Example probability for DoS attack
        fdi_attack_probability=0.0,   # Example probability for FDI attack
        max_dos_robots=2,             # Example maximum number of robots affected by DoS attack
        max_fdi_measurements=2       # Example maximum number of measurements affected by FDI attack
    )

    






