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

# Configure Logging
logging.basicConfig(level=logging.DEBUG,  # Set the log level
                    format='%(asctime)s - %(levelname)s - %(message)s')  # Set the log format

# Configuration Constants
CONFIG = {
    # Total number of robots in the system.
    "TOTAL_ROBOTS": 10,
    
    # Standard deviation of noise added to control inputs.
    "CONTROL_NOISE_STD": 0.05,
    
    # Covariance matrix representing the process noise in the system.
    "PROCESS_NOISE_COVARIANCE": np.eye(2) * 0.01,  # 2x2 identity matrix scaled by 0.01.
    
    # Threshold for triggering events in the control strategy.
    "EVENT_TRIGGER_THRESHOLD": 0.1,
    
    # Small constant to avoid division by zero in calculations.
    "EPSILON": 1e-6,
    
    # Standard deviation of noise in range measurements.
    "RANGE_NOISE_STD": 0.1,
    
    # Standard deviation of noise in bearing measurements.
    "BEARING_NOISE_STD": 0.05,
    
    # Operational area limits for the x-coordinate.
    "x_limits": (-1, 1),
    
    # Operational area limits for the y-coordinate.
    "y_limits": (-1, 1),
    
    # Dimensions of the target area for the robots.
    "TARGET_AREA": (1, 1),
    
    # Number of steps in the simulation or control loop.
    "num_steps": 50,

    # Proportional gain for the PID controller—controls response based on current error.
    "kp_leader": 1.0,
    "kp_follower": 0.8,  # Follower robots may have different dynamics.

    # Integral gain for the PID controller—eliminates steady-state error.
    "ki_leader": 0.1,
    "ki_follower": 0.05,
    
    # Derivative gain for the PID controller—accounts for rate of change of error.
    "kd_leader": 0.05,
    "kd_follower": 0.03,

    # Time interval for each iteration of the control loop.
    "dt": 0.1,
    
    # Initialize the integral term for the PID controller (2D control).
    "integral_leader": np.zeros(2),
    "integral_follower": np.zeros(2),
    
    # Initialize the previous error for the PID controller (2D control).
    "previous_error_leader": np.zeros(2),
    "previous_error_follower": np.zeros(2),

    # Limit for control input to prevent excessive values.
    "CONTROL_INPUT_LIMIT": 2.0,

    # Minimum safe distance between robots to avoid collisions.
    "SAFE_DISTANCE": 0.5,  # Minimum distance robots should maintain to avoid collisions.

    # Gain for the control barrier function (CBF) to ensure collision avoidance.
    "cbf_gain": 1.5,

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
    "close_enough": 0.03,

    # distance between the wheel
    "b": 0.1,
    
    #Step size 
    "delta_steps": 0.01
}



# Global Variables for Attack Detection
consecutive_large_innovations = 0
attack_threshold = 2.0  # Threshold for detecting an attack
benign_threshold = 1.0   # Threshold for benign detection
steps_for_attack = 3     # Steps needed to confirm an attack
state = "normal"         # Initial state

# Initialize Arrays for Innovations and Adaptive Thresholds
previous_innovations = np.zeros((2, CONFIG["TOTAL_ROBOTS"], CONFIG["num_steps"]))  # Store innovations for each robot across all iterations
adaptive_thresholds = np.full((CONFIG["TOTAL_ROBOTS"], CONFIG["num_steps"]), CONFIG["EVENT_TRIGGER_THRESHOLD"])  # Adaptive thresholds for each robot across all iterations

# Define Range Limits
x_range = CONFIG["x_limits"]
y_range = CONFIG["y_limits"]

# Generate Nearby Positions for Demonstration
nearby_positions = np.random.rand(2, CONFIG["TOTAL_ROBOTS"]) * 10  # Random positions in a 10x10 area

# Measurement Noise Covariance
num_nearby = nearby_positions.shape[1]
MEASUREMENT_NOISE_COVARIANCE = np.diag(
    [CONFIG["RANGE_NOISE_STD"]**2]  + 
    [CONFIG["BEARING_NOISE_STD"]**2] 
)

# Randomly Initialize Current Positions for Each UGV across num_steps
positions = np.random.rand(2, CONFIG["TOTAL_ROBOTS"], CONFIG["num_steps"]) * 100  # Current positions in a 100x100 area
target_positions = np.random.rand(2, CONFIG["TOTAL_ROBOTS"], CONFIG["num_steps"]) * 100  # Target positions in a 100x100 area

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
    P_self_pred[:, :, i, 0] = np.eye(3)  # Initial covariance for the first time step
    P_cross_pred[:, :, i, 0] = np.eye(3)  # Initial covariance for the first time step


# Initialize State Estimates and Covariance Matrices
x_hat = np.zeros((3, CONFIG["TOTAL_ROBOTS"], CONFIG["num_steps"]))  # State vector: [x, y, theta]
# Initialize error covariance matrices for each robot and each time step
# P will have the shape (TOTAL_ROBOTS, 3, 3, num_steps)
P_self = np.zeros((3, 3, CONFIG["TOTAL_ROBOTS"],  CONFIG["num_steps"]))  # For time-dependent covariance matrices
P_cross = np.zeros((3, 3, CONFIG["TOTAL_ROBOTS"],  CONFIG["num_steps"]))  # For time-dependent covariance matrices

# Fill initial covariance matrices with identity matrices for the first time step
for i in range(CONFIG["TOTAL_ROBOTS"]):
    P_self[:, :, i, 0] = np.eye(3)  # Initial covariance for the first time step
    P_cross[:, :, i, 0] = np.eye(3)  # Initial covariance for the first time step




# Control inputs (random initial values)
control_inputs = np.random.rand(2, CONFIG["TOTAL_ROBOTS"]) * 0.1  # 2D control inputs (v, omega)

# Initialize leader state
State = 0 

# Define waypoints for the leader
waypoints = np.array([[0.1, 0.9, 0.5, 0.5], [0.1, 0.1, 0.9, 0.9]])






def generate_initial_positions(num_robots, x_range, y_range):
    # Generate random positions for x and y within specified ranges
    x_positions = np.random.uniform(x_range[0], x_range[1], num_robots)
    y_positions = np.random.uniform(y_range[0], y_range[1], num_robots)
    # Random orientations for each robot
    theta_positions = np.random.uniform(0, 2 * np.pi, num_robots)
    
    # Combine positions and orientations into a 3xN array
    initial_positions = np.array([x_positions, y_positions, theta_positions])
    
    return initial_positions



# EKF Functions

def time_update(x_hat, P_self, Q, control_input, f, F_jacobian, P_cross):
    """
    Performs the Extended Kalman Filter (EKF) time update (prediction step).
    
    Parameters:
        x_hat (ndarray): Current state estimate.
        P_self (ndarray): Current state covariance for this robot.
        Q (ndarray): Process noise covariance.
        control_input (ndarray): Control input applied to the system.
        f (function): Nonlinear state transition function.
        F_jacobian (function): Function to compute the Jacobians of f with respect to the state and control input.
        P_cross (ndarray): Cross-covariance matrix between this robot and others.
    
    Returns:
        x_hat_pred (ndarray): Predicted state estimate.
        P_self_pred (ndarray): Predicted state covariance for this robot.
        P_cross_pred (ndarray): Predicted cross-covariance matrix.
    """
    logging.debug("Performing EKF time update.")

    # Predict state using the nonlinear transition function
    x_hat_pred = f(x_hat, control_input)

    # Calculate Jacobians of f with respect to state and control input
    F_x, F_u = F_jacobian(x_hat, control_input)

    # Predict state covariance
    P_self_pred = F_x @ P_self @ F_x.T + F_u @ Q @ F_u.T

    # Predict cross-covariance
    P_cross_pred = F_x @ P_cross @ F_x.T

    logging.info("Time update completed.")
    return x_hat_pred, P_self_pred, P_cross_pred




def measurement_update(x_hat_pred, P_self_pred, measurement, R, h, nearby_positions, nearby_robots, index, 
                       previous_innovation, adaptive_threshold, shadow_measurement, shadow_R, H_jacobian, P_cross_pred):
    """
    Performs the EKF measurement update (correction) with event-triggered communication and attack detection.
    """
    logging.debug(f"Performing EKF measurement update for UGV {index}.")

    # Predicted measurement using nonlinear measurement function
    z_hat_pred = measurement_function(x_hat_pred, nearby_positions, index)


    # Calculate Jacobian of h at the predicted state estimate
    H_r, H_l = H_jacobian(x_hat_pred, nearby_positions)

    # Innovation (residual)
    innovation_regular = measurement - z_hat_pred

    print(R.shape, "R.shape")
    print(H_r.shape, "H_r.shape")
    print(P_self_pred.shape, "P_pred.shape")
    
    P_zz = np.zeros((2, 2))
    # Sum over all neighboring robots
    for i in range(len(H_r)):
        P_zz += H_r[i] @ P_self_pred @ H_r[i].T + H_l[i] @ P_cross_pred @ H_l[i].T

    # Add the measurement noise covariance
    P_zz += R

    # Regularization for P_zz
    if P_zz.size == 0:
        logging.warning("P_zz is empty. Skipping update.")
        return x_hat_pred, P_self_pred, previous_innovation, adaptive_threshold  # Handle appropriately
    if np.linalg.cond(P_zz) > 1e10:
        P_zz += CONFIG["EPSILON"] * np.eye(P_zz.shape[0])
        
    # Regularization for P_zz
    if np.linalg.cond(P_zz) > 1e10:
        P_zz += CONFIG["EPSILON"] * np.eye(P_zz.shape[0])

    
    P_regular = np.zeros((3, 2))
    # Calculate Kalman gain
    for i in range(len(H_r)):
        P_regular += (P_self_pred @ H_r[i].T + P_cross_pred @ H_l[i].T) 

    K_regular = P_regular @ np.linalg.inv(P_zz)
    print(K_regular.shape, "K_regular.shape")
    

    print(innovation_regular[:2].shape, "innovation_regular.shape")
    print(x_hat_pred.shape, "x_hat_pred.shape")

    update = np.zeros((3, ))

    # Update state estimate with the regular measurement
    # Update state estimate with the regular measurement
    for i in range(1, len(H_r)):  # Iterate from 1 to 49
        index = (i - 1) * 2  # Calculate the starting index for two elements
        if index + 2 <= len(innovation_regular):  # Check to avoid out of bounds
            update += K_regular @ innovation_regular[index:index + 2]  # Take two elements
    
    x_hat_updated = x_hat_pred + update
    
    print(x_hat_updated.shape, "x_hat_updated.shape")

    # Update covariance
    P_self_updated = P_self_pred - K_regular @ P_zz @ K_regular.T
    
    print(shadow_measurement.shape, "shadow_measurement.shape")
    # Handle shadow measurements
    #if shadow_measurement.size > 0:
        #shadow_z_hat_pred = measurement_function(x_hat_pred, nearby_positions, index)
        #Hr_shadow, Hl_shadow = H_jacobian(x_hat_pred, nearby_positions)

        #print(shadow_measurement.shape, "shadow_measurement.shape")
        #shadow_z_hat_pred = shadow_z_hat_pred[:84]

        #print(shadow_z_hat_pred.shape, "shadow_z_hat_pred.shape")


        #shadow_innovation = shadow_measurement - shadow_z_hat_pred
        #P_zz_shadow = Hr_shadow[0] @ P_pred @ Hr_shadow[0].T + shadow_R

        #if np.linalg.cond(P_zz_shadow) > 1e10:
            #P_zz_shadow += CONFIG["EPSILON"] * np.eye(P_zz_shadow.shape[0])

        #K_shadow = 0.5 * P_pred @ Hr_shadow[0].T @ np.linalg.inv(P_zz_shadow)

        #print(K_shadow.shape, "K_shadow.shape")

        #print(shadow_innovation.shape, "shadow_innovation.shape")
        #shadow_innovation = shadow_innovation[:, np.newaxis]

        #shadow_innovation = shadow_innovation[:84]

        # Incorporate shadow measurement update
        #x_hat_updated += K_shadow @ shadow_innovation
        #P_updated -= K_shadow @ P_zz_shadow @ K_shadow.T

    # Regularize the updated covariance to ensure symmetry and positive definiteness
    P_self_updated = (P_self_updated + P_self_updated.T) / 2 + CONFIG["EPSILON"] * np.eye(x_hat_pred.shape[0])



    # Check for attack detection and perform event-triggered communication
    if not attack_detected(innovation_regular):
        event_trigger, updated_threshold = event_triggered(
            innovation_regular, previous_innovation, adaptive_threshold
        )

        if event_trigger.any():
            logging.info(f"Event triggered for UGV {index}. Communicating state to neighbors.")
            return x_hat_updated, P_self_updated, innovation_regular, updated_threshold
        else:
            logging.info(f"No event triggered for UGV {index}. Using predicted state.")
            return x_hat_pred, P_self_pred, previous_innovation, adaptive_threshold
    else:
        logging.warning(f"Attack detected for UGV {index}. Ignoring the measurement update.")
        return x_hat_pred, P_self_pred, previous_innovation, adaptive_threshold

# Jacobian Functions

def F_jacobian(x_hat, control_input):
    """
    Calculate the Jacobians of the state transition function with respect to state and control inputs.
    """
    # Jacobian with respect to state x (F_x)
    F_x = np.array([[1, 0, -((control_input[1] - control_input[0]) / (2 * CONFIG["b"])) * np.sin(x_hat[2])],
                    [0, 1,  ((control_input[1] - control_input[0]) / (2 * CONFIG["b"])) * np.cos(x_hat[2])],
                    [0, 0, 1]])

    # Jacobian with respect to control input u (F_u)
    F_u = np.array([[0.5 * np.cos(x_hat[2] + (control_input[1] - control_input[0]) / (2 * CONFIG["b"])) , 
                     0.5 * np.cos(x_hat[2] + (control_input[1] - control_input[0]) / (2 * CONFIG["b"]))],
                    [0.5 * np.sin(x_hat[2] + (control_input[1] - control_input[0]) / (2 * CONFIG["b"])),
                     0.5 * np.sin(x_hat[2] + (control_input[1] - control_input[0]) / (2 * CONFIG["b"]))],
                    [1 / CONFIG["b"], -1 / CONFIG["b"]]])
    
    return F_x, F_u




def H_jacobian(current_pose, nearby_pose):
    """
    Computes the Jacobians H_r and H_l for the measurement model.
    current_pose should be a 1D array of shape (3,) [rx, ry, theta].
    nearby_pose should be a 2D array of shape (n, 2) where n is the number of nearby landmarks.
    """
    # Ensure current_pose is a 1D array of shape (3,)
    if current_pose.ndim != 1 or current_pose.shape[0] != 3:
        raise ValueError("current_pose must be a 1D array of shape (3,).")
    
    # Unpack current pose
    rx, ry = current_pose[0], current_pose[1]
    
    H_r_list = []
    H_l_list = []
    
    for nearby in nearby_pose:
        lx, ly = nearby[0], nearby[1]
        
        q = np.sqrt((lx - rx) ** 2 + (ly - ry) ** 2)
        
        if q == 0:
            raise ValueError("The distance q cannot be zero, as it leads to division by zero in H_jacobian.")
        
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
    # Estimates the next position and orientation of a 2 wheeled robot
    # INPUT: 
    # previousPose: 1x3 array [previousX, previousY, previousTheta]
    # ut: 1x2 array [DL, DR]
    
    previousX = previousPose[0]
    previousY = previousPose[1]
    previousTheta = previousPose[2]

    DL = ut[0]
    DR = ut[1]


    # Calculate the new x coordinate
    currentX = previousX + ((DR + DL) / 2) * np.cos(previousTheta + ((DR - DL) / (2 * CONFIG["b"])))

    # Calculate the new y coordinate
    currentY = previousY + ((DR + DL) / 2) * np.sin(previousTheta + ((DR - DL) / (2 * CONFIG["b"])))

    # Calculate the new Theta
    currentTheta = previousTheta + (DR - DL) / CONFIG["b"]

    # Normalize results between [-pi, pi]
    currentTheta = normalizeAngle(currentTheta)

    print(np.array([currentX, currentY, currentTheta]).shape, "np.array([currentX, currentY, currentTheta]).shape")

    # Create the output as a numpy array
    return np.array([currentX, currentY, currentTheta]).flatten()

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
    if normalizedAngle > np.pi:
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


def leader_control_policy(current_position, target_position, positions, min_distance, config):
    """
    Control policy for the leader robot with Control Barrier Function to avoid collisions.
    
    Parameters:
        current_position (np.ndarray): Current position of the leader robot.
        target_position (np.ndarray): Target position for the leader robot.
        positions (np.ndarray): Positions of all robots (shape: (2, N)).
        min_distance (float): Minimum safe distance.
        config (dict): Configuration parameters.
        
    Returns:
        np.ndarray: Control input for the leader robot.
    """
    # PID control to track the target position
    control_input = pid_control(current_position, target_position, config, "leader")

    # Check distances to all other robots and apply CBF if necessary
    for i in range(1, config["TOTAL_ROBOTS"]):  # Avoid checking itself
        control_input = apply_cbf_control(control_input, current_position, positions[:, i], min_distance, config)
    
    return control_input


def follower_control_policy(current_position, desired_trajectory, positions, min_distance, config):
    """
    Control policy for follower robots with Control Barrier Function to avoid collisions.
    
    Parameters:
        current_position (np.ndarray): Current position of the follower robot.
        desired_trajectory (np.ndarray): Desired trajectory (e.g., leader's position).
        positions (np.ndarray): Positions of all robots (shape: (2, N)).
        min_distance (float): Minimum safe distance.
        config (dict): Configuration parameters.
        
    Returns:
        np.ndarray: Control input for the follower robot.
    """
    # PID control to track the desired trajectory
    control_input = pid_control(current_position, desired_trajectory, config, "follower")

    # Check distances to all other robots and apply CBF if necessary
    for i in range(config["TOTAL_ROBOTS"]):  # Avoid checking itself
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

def range_measurement(own_position, nearby_positions):
    """
    Calculate the range measurement between the UGV's current position and nearby positions.
    """
    own_position = own_position.reshape(-1)  # Ensure own_position is shape (2,)
    
    # Check if there are any nearby positions
    if nearby_positions.shape[0] == 0:
        # If there are no nearby positions, return an empty array or handle accordingly
        return np.array([])  # Returning an empty array or you can handle this case as needed

    ranges_to_nearby = np.linalg.norm(nearby_positions - own_position, axis=1)  # Shape: (N,)
    ranges = np.concatenate([ranges_to_nearby])  # Combine distances
    return ranges


def bearing_measurement(own_position, own_orientation, nearby_positions):
    """
    Calculates the bearing measurements to nearby UGVs and the initial position.
    """
    # Check if there are any nearby positions
    if nearby_positions.shape[0] == 0:
        # If there are no nearby positions, return an empty array or handle accordingly
        return np.array([])  # Returning an empty array or handle this case as needed

    vectors_to_nearby = nearby_positions - own_position  # Shape: (N, 2)
    angles_to_nearby = np.arctan2(vectors_to_nearby[:, 1], vectors_to_nearby[:, 0])
    bearings = angles_to_nearby - own_orientation

    bearings = np.arctan2(np.sin(bearings), np.cos(bearings))
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






def measurement_function(robot_position, nearby_positions, index):
    """
    Constructs a measurement vector from the ground truth.
    """
    current_position = robot_position[:2]  # Shape: (2,)
    
    ranges = range_measurement(current_position, nearby_positions)
    bearings = bearing_measurement(current_position, robot_position[2], nearby_positions)
    
    measurement = np.concatenate([ranges, bearings])
    return measurement

def event_triggered(innovation, prev_innovation, threshold=None, decay_factor=0.95):
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





def attack_detected(innovation, threshold=1.0):
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
    return consecutive_large_innovations > 5



def get_neighbors(i, total_robots, shadow=False):
    """Simulate neighbors and shadow neighbors for robot i."""
    neighbors = []
    shadow_neighbors = []
    for j in range(total_robots):
        if i != j:  # Exclude self
            if np.linalg.norm(x_hat[:, i] - x_hat[:, j]) < 1.0:
                neighbors.append(j)
            elif shadow and np.linalg.norm(x_hat[:, i] - x_hat[:, j]) < 2.0:
                shadow_neighbors.append(j)
    return neighbors, shadow_neighbors


# Assuming you have a function to initialize robots
def initialize_robots(r, leader_index, total_robots):
    # Colors: Red for leader, Green for followers
    colors = [np.array([1, 0, 0])] + [np.array([0, 1, 0])] * (total_robots - 1)
    
    # Initialize each robot with its respective color
    for i in range(total_robots):
        r.set_color(i, colors[i])



def run_simulation(robotarium_env, CONFIG, x_hat, P_self, P_cross, positions, target_positions, 
                   apply_process_noise, time_update, measurement_function, 
                   measurement_update, attack_detected, get_neighbors, event_triggered, 
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


    leader_index = 0  # Leader is robot 0
    follower_indices = [i for i in range(1, CONFIG["TOTAL_ROBOTS"])]
    d_min = CONFIG["SAFE_DISTANCE"]  # Minimum safe distance to avoid collisions


    # Initialize distances, errors, and rho
    distances, errors = calculate_distances_and_errors(positions)
    rho = compute_rho(positions)

    initial_positions = generate_initial_positions(CONFIG["TOTAL_ROBOTS"], x_range, y_range)

    r = robotarium.Robotarium(number_of_robots=CONFIG["TOTAL_ROBOTS"], show_figure=True, initial_conditions=initial_positions, sim_in_real_time=True)
    _,uni_to_si_states = create_si_to_uni_mapping()
    si_to_uni_dyn = create_si_to_uni_dynamics()
    si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
    leader_controller = create_si_position_controller(velocity_magnitude_limit=CONFIG["CONTROL_INPUT_LIMIT"])


    # For computational/memory reasons, initialize the velocity vector
    dxi = np.zeros((2,CONFIG["TOTAL_ROBOTS"]))
    # Set initial axis limits
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)


    # Simulation Loop
    for step in range(CONFIG["num_steps"]):  
        logging.debug(f"Step {step + 1}/{CONFIG['num_steps']}")

        # Step the Robotarium environment
        ground_truth = r.get_poses()
        logging.debug(f"Ground truth generated: {ground_truth}")



        # Step the simulation forward
        r.step()
        #plt.pause(0.01)  # Pause for a brief moment to allow the plot to update

        xi = uni_to_si_states(ground_truth)


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
            neighbors, shadow_neighbors = get_neighbors(i, CONFIG["TOTAL_ROBOTS"], shadow=True)
            logging.info(f"Neighbors generated for robot {i}: {neighbors}")
            # Get positions of nearby UGVs (neighbors only)
            nearby_positions = np.array([x_hat[:2, j, step] for j in neighbors]) 


            # Compute regular range and bearing measurements
            range_meas = range_measurement(x_hat[:2, i, step - 1], nearby_positions)
            bearing_meas = bearing_measurement(x_hat[:2, i, step - 1], x_hat[2, i, step - 1], nearby_positions)

            # Compute shadow edge measurements
            shadow_neighbors_positions = np.array([x_hat[:2, j, step] for j in shadow_neighbors])
            shadow_ranges = {}  # Correctly initialize as a dictionary
            for j in range(CONFIG["TOTAL_ROBOTS"]):
            # Replace with actual range calculation logic
                shadow_ranges[(i, j)] = (np.random.uniform(1, 10), None)  # Example measurement

            # Later in your code where the error occurred
            shadow_range_meas = [shadow_ranges.get((i, j), (0, 0))[0] for j in shadow_neighbors]

            # Check for FDI attack affecting limited measurements
            if np.random.rand() < fdi_attack_probability:
                # Randomly select a limited number of measurements to corrupt
                fdi_indices = np.random.choice(len(range_meas), size=min(max_fdi_measurements, len(range_meas)), replace=False)
                for idx in fdi_indices:
                    range_meas[idx] += np.random.normal(0, 0.5)  # Corrupt range measurement
                    bearing_meas[idx] += np.random.normal(0, 0.5)  # Corrupt bearing measurement

            combined_measurements = np.concatenate([range_meas, bearing_meas])
            logging.info(f"Combined measurements generated for robot {i}: {combined_measurements}")

            # Initialize state before the loop if it's not already initialized
            if "state" not in locals():
                state = 0  # Assuming 0 is the starting state or the initial index for waypoints


            # Leader control logic
            if i == CONFIG["LEADER_INDEX"]:
                leader_position = xi[:, CONFIG["LEADER_INDEX"]]
                target_position = waypoints[:,state].reshape((2,))
                control_input = leader_control_policy(leader_position, target_position, xi, d_min, CONFIG)
                #plt.scatter(ground_truth[0, i], ground_truth[1, i], color='red', label='Leader' if i == CONFIG["LEADER_INDEX"] else "")


                # Check if the leader has reached its target waypoint
                if np.linalg.norm(leader_position - target_position) < CONFIG["close_enough"]:
                    state = (state + 1) % waypoints.shape[1]  # Update state to the next waypoint

            else:
                # Follower control logic
                desired_trajectory = xi[:, CONFIG["LEADER_INDEX"]]  # Follower aims to follow the leader
                control_input = follower_control_policy(xi[:, i], desired_trajectory, xi, d_min, CONFIG)
                # Plot followers
                #plt.scatter(ground_truth[0, i], ground_truth[1, i], color='green', label='Follower' if i == 0 else "")


            # Apply process noise to the control input for both leader and followers
            dxi[:, i] = apply_process_noise(control_input, CONFIG)  # Apply noise to follower inputs
            if i == CONFIG["LEADER_INDEX"]:
                dxi[:, CONFIG["LEADER_INDEX"]] = apply_process_noise(control_input, CONFIG)  # Apply noise to leader input

            # Apply control barriers, convert to unicycle, and set velocities
            dxu = si_to_uni_dyn(dxi, ground_truth)  # Convert to unicycle dynamics
            r.set_velocities(np.arange(CONFIG["TOTAL_ROBOTS"]), dxu)  # Update velocities for all robots


            control_input_noisy = apply_process_noise(control_input, CONFIG)


            # Ensure control_input_noisy is 2D before setting velocities
            if control_input_noisy.ndim == 1:
                control_input_noisy = control_input_noisy[:, np.newaxis]

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

            # Update estimated states and covariance only if DoS attack is not active
            if i not in dos_attacked_robots:
            # Count the number of nearby robots excluding the current robot i
                number_of_nearby_robots = len([j for j in neighbors if j != i]) + len(shadow_neighbors)  # Include shadow neighbors

            # Prepare shadow measurements and covariance
                shadow_measurement = np.concatenate([shadow_range_meas])  # Include shadow measurements
                shadow_R = MEASUREMENT_NOISE_COVARIANCE  # Use the same covariance for shadow measurements

                # Call the measurement_update function with all required arguments
                x_hat_updated, P_self_updated, previous_innovation, updated_threshold = measurement_update(
                    x_hat_pred[:, i, step], 
                    P_self_pred[:, :, i, step], 
                    combined_measurements, 
                    MEASUREMENT_NOISE_COVARIANCE, 
                    measurement_function(x_hat[:, i, step-1], nearby_positions, i), 
                    nearby_positions, 
                    number_of_nearby_robots, 
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
            print(x_hat[:, i, step], "x_hat[:, i, step]")
 
            positions[:, i, step] = x_hat[0:2, i, step]  # Update position based on the estimated state
  

            # Update estimated positions for the next step
            ground_truth_positions.append(ground_truth[:, i])
            estimated_positions.append(x_hat[:, i, step])
            control_command.append(control_input[:, np.newaxis])



    ground_truth_positions = np.array(ground_truth_positions)
    estimated_positions = np.array(estimated_positions)
    control_command = np.squeeze(np.array(control_command))

    # Finalizing the simulation
    logging.debug(f"Final step {step}: current positions {positions}")
    
    plot_final_states(ground_truth_positions, estimated_positions, control_command,  CONFIG["TOTAL_ROBOTS"], CONFIG["num_steps"])

    
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

    plt.tight_layout(pad=6.0)
    plt.subplots_adjust(top=0.90, hspace=0.8, wspace=0.6)
    plt.suptitle("Event-based EKF for Multi-Robot Localization in Sparse Sensing Graph and Adversarial Environment \n Control Commands for Robots", fontsize=16)
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
        target_positions=target_positions,
        apply_process_noise=apply_process_noise,
        time_update=time_update,
        measurement_function=measurement_function,
        measurement_update=measurement_update,
        attack_detected=attack_detected,
        get_neighbors=get_neighbors,
        event_triggered=event_triggered,
        generate_initial_positions=generate_initial_positions,
        range_measurement=range_measurement,
        bearing_measurement=bearing_measurement,
        shadow_range_measurement=shadow_range_measurement,
        leader_control_policy=leader_control_policy,
        follower_control_policy=follower_control_policy,
        plot_final_states=plot_final_states,
        do_s_attack_probability=0.1,  # Example probability for DoS attack
        fdi_attack_probability=0.2,   # Example probability for FDI attack
        max_dos_robots=2,             # Example maximum number of robots affected by DoS attack
        max_fdi_measurements=3        # Example maximum number of measurements affected by FDI attack
    )

    





