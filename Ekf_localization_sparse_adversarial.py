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
    "TOTAL_ROBOTS": 50,
    
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
    "num_steps": 20,

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
previous_innovations = np.zeros((3, CONFIG["TOTAL_ROBOTS"], CONFIG["num_steps"]))  # Store innovations for each robot across all iterations
adaptive_thresholds = np.full((CONFIG["TOTAL_ROBOTS"], CONFIG["num_steps"]), CONFIG["EVENT_TRIGGER_THRESHOLD"])  # Adaptive thresholds for each robot across all iterations

# Define Range Limits
x_range = CONFIG["x_limits"]
y_range = CONFIG["y_limits"]

# Generate Nearby Positions for Demonstration
nearby_positions = np.random.rand(2, CONFIG["TOTAL_ROBOTS"]) * 10  # Random positions in a 10x10 area

# Measurement Noise Covariance
num_nearby = nearby_positions.shape[1]
MEASUREMENT_NOISE_COVARIANCE = np.diag(
    [CONFIG["RANGE_NOISE_STD"]**2] * (num_nearby ) + 
    [CONFIG["BEARING_NOISE_STD"]**2] * (num_nearby)
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
P_pred = np.zeros((3, 3, CONFIG["TOTAL_ROBOTS"],  CONFIG["num_steps"]))  # For time-dependent covariance matrices

# Fill initial covariance matrices with identity matrices for the first time step
for i in range(CONFIG["TOTAL_ROBOTS"]):
    P_pred[:, :, i, 0] = np.eye(3)  # Initial covariance for the first time step

# Initialize State Estimates and Covariance Matrices
x_hat = np.zeros((3, CONFIG["TOTAL_ROBOTS"], CONFIG["num_steps"]))  # State vector: [x, y, theta]
# Initialize error covariance matrices for each robot and each time step
# P will have the shape (TOTAL_ROBOTS, 3, 3, num_steps)
P = np.zeros((3, 3, CONFIG["TOTAL_ROBOTS"],  CONFIG["num_steps"]))  # For time-dependent covariance matrices

# Fill initial covariance matrices with identity matrices for the first time step
for i in range(CONFIG["TOTAL_ROBOTS"]):
    P[:, :, i, 0] = np.eye(3)  # Initial covariance for the first time step



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

def time_update(x_hat, P, Q, control_input, f, F_jacobian):
    """
    Performs the EKF time update (prediction).
    """
    logging.debug("Performing EKF time update.")

    # Predict state using the nonlinear function

    x_hat = x_hat[:, np.newaxis]  # Converts from (3,) to (3, 1)
    x_hat_pred = f(x_hat, control_input).flatten()

    print(x_hat.shape, "x_hat.shape")
    print(Q.shape, "Q.shape")
    print(P.shape, "P.shape")

    print(x_hat_pred, "x_hat_pred")

    # Calculate Jacobian of f at the current state estimate
    F_x, F_u = F_jacobian(x_hat, control_input)

    print(F_x.shape, "F_x.shape")
    print(F_u.shape, "F_u.shape")

    # Predict covariance using the Jacobian
    P_pred = F_x @ P @ F_x.T + F_u @ Q @ F_u.T

    print(P_pred.shape, "P_pred.shape")

    logging.info("Time update completed.")
    return x_hat_pred, P_pred



def measurement_update(x_hat_pred, P_pred, measurement, R, h, nearby_positions, nearby_robots, index, 
                       previous_innovation, adaptive_threshold, shadow_measurement, shadow_R, H_jacobian):
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

    # Innovation covariance
    P_zz = H_r @ P_pred @ H_r.T + R 

    # Regularization for P_zz
    if np.linalg.cond(P_zz) > 1e10:
        P_zz += CONFIG["EPSILON"] * np.eye(P_zz.shape[0])

    # Calculate Kalman gain
    K_regular = P_pred @ H_r.T @ np.linalg.inv(P_zz)

    # Update state estimate with the regular measurement
    x_hat_updated = x_hat_pred + K_regular @ innovation_regular

    # Update covariance
    P_updated = P_pred - K_regular @ P_zz @ K_regular.T

    # Handle shadow measurements
    if shadow_measurement is not None:
        shadow_z_hat_pred = measurement_function(x_hat_pred, nearby_positions, index)
        H_shadow = H_jacobian(x_hat_pred, nearby_positions)

        shadow_innovation = shadow_measurement - shadow_z_hat_pred
        P_zz_shadow = H_shadow @ P_pred @ H_shadow.T + shadow_R

        if np.linalg.cond(P_zz_shadow) > 1e10:
            P_zz_shadow += CONFIG["EPSILON"] * np.eye(P_zz_shadow.shape[0])

        K_shadow = 0.5 * P_pred @ H_shadow.T @ np.linalg.inv(P_zz_shadow)

        # Incorporate shadow measurement update
        x_hat_updated += K_shadow @ shadow_innovation
        P_updated -= K_shadow @ P_zz_shadow @ K_shadow.T

    # Regularize the updated covariance to ensure symmetry and positive definiteness
    P_updated = (P_updated + P_updated.T) / 2 + CONFIG["EPSILON"] * np.eye(P_updated.shape[0])

    # Check for attack detection and perform event-triggered communication
    if not attack_detected(innovation_regular):
        event_trigger, updated_threshold = event_triggered(
            innovation_regular, previous_innovation, adaptive_threshold
        )

        if event_trigger.any():
            logging.info(f"Event triggered for UGV {index}. Communicating state to neighbors.")
            return x_hat_updated, P_updated, innovation_regular, updated_threshold
        else:
            logging.info(f"No event triggered for UGV {index}. Using predicted state.")
            return x_hat_pred, P_pred, previous_innovation, adaptive_threshold
    else:
        logging.warning(f"Attack detected for UGV {index}. Ignoring the measurement update.")
        return x_hat_pred, P_pred, previous_innovation, adaptive_threshold

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
    """
    # Ensure current_pose and nearby_pose are 1D arrays
    lx, ly = nearby_pose[0], nearby_pose[1]
    rx, ry = current_pose[0], current_pose[1]

    print(lx.shape, "lx.shape")
    print(ly.shape, "ly.shape")
    print(rx.shape, "rx.shape")
    print(ry.shape, "ry.shape")



    
    q = np.sqrt((lx - rx) ** 2 + (ly - ry) ** 2)
    
    # Check for division by zero
    if q.any() == 0:
        raise ValueError("The distance q cannot be zero, as it leads to division by zero in H_jacobian.")

    # Calculate Jacobian H_r
    H_r = (1 / q) * np.array([
        [-(lx - rx), -(ly - ry), 0],  # First row
        [(ly - ry) / q, -(lx - rx) / q, -q]  # Second row
    ])
    
    # Calculate Jacobian H_l
    H_l = (1 / q) * np.array([
        [lx - rx, ly - ry, 0],  # First row
        [-(ly - ry) / q, (lx - rx) / q, 0]  # Second row
    ])
    
    return H_r, H_l




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

    # Create the output as a numpy array
    return np.array([[currentX], [currentY], [currentTheta]])

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



def run_simulation(robotarium_env, CONFIG, x_hat, P, positions, target_positions, 
                   apply_process_noise, time_update, measurement_function, 
                   measurement_update, attack_detected, get_neighbors, event_triggered, 
                   generate_initial_positions, range_measurement, bearing_measurement, 
                   shadow_range_measurement,  # New shadow measurement functions
                   leader_control_policy, follower_control_policy, 
                   animate, plot_final_states, do_s_attack_probability, 
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
                P_pred[:, :, i, step] = np.eye(len(P_pred[:, :, i, step])) * 1e5  
            else:
                # Prediction step if no DoS attack
                x_hat_pred[:, i, step], P_pred[:, :, i, step] = time_update(
                    x_hat[:, i, step - 1], P[:, :, i, step - 1], 
                    CONFIG["PROCESS_NOISE_COVARIANCE"], control_input_noisy, 
                    state_transition, F_jacobian
                )

            # Update estimated states and covariance only if DoS attack is not active
            if i not in dos_attacked_robots:
            # Count the number of nearby robots excluding the current robot i
                number_of_nearby_robots = len([j for j in neighbors if j != i]) + len(shadow_neighbors)  # Include shadow neighbors

            # Prepare shadow measurements and covariance
                shadow_measurement = np.concatenate([shadow_range_meas])  # Include shadow measurements
                shadow_R = MEASUREMENT_NOISE_COVARIANCE  # Use the same covariance for shadow measurements

                # Call the measurement_update function with all required arguments
                x_hat_updated, P_updated, previous_innovation, updated_threshold = measurement_update(
                    x_hat_pred[:, i, step], 
                    P_pred[:, :, i, step], 
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
                    H_jacobian
                )

            # Store the updated values back to the arrays
            x_hat[:, i, step] = x_hat_updated
            P[:, :, i, step] = P_updated
            previous_innovations[:, i, step] = previous_innovation
            adaptive_thresholds[i] = updated_threshold
            print(x_hat[:, i, step], "x_hat[:, i, step]")
 
            positions[:, i, step] = x_hat[0:2, i, step]  # Update position based on the estimated state
  

            # Update estimated positions for the next step
            ground_truth_positions.append(ground_truth[:, i])
            estimated_positions.append(x_hat[:, i, step])
        print(estimated_positions, "estimated_positions")
        print(ground_truth_positions, "ground_truth_positions")


    # Finalizing the simulation
    logging.debug(f"Final step {step}: current positions {positions}")

     # Animate the results
    #animate(np.array(ground_truth_positions), np.array(estimated_positions), CONFIG["num_steps"])
    
    #plot_final_states(np.array(ground_truth_positions), np.array(estimated_positions))

    
    input("Press Enter to close...")  # Pause until user input

    # Clean up the Robotarium environment
    logging.info("Simulation completed.")

    #Call at end of script to print debug information and for your script to run on the Robotarium server properly
    r.call_at_scripts_end()



def animate(ugvs, estimates, num_steps):
    """Real-time animation of UGV positions comparing ground truth and estimates."""
    fig, ax = plt.subplots()
    colors = ['b', 'g']  # Define colors for UGVs (ground truth and estimates)

    # Set axis limits and labels
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_title("UGV Positions Over Time")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True)  # Add grid for better visibility

    # Create a scatter plot for ground truth and estimates
    ground_truth_scatter = ax.scatter([], [], color='b', label='Ground Truth')
    estimates_scatter = ax.scatter([], [], color='g', label='Estimates')

    ax.legend()

    # Initialize function for the animation
    def init():
        ground_truth_scatter.set_offsets(np.empty((0, 2)))  # Empty 2D array
        estimates_scatter.set_offsets(np.empty((0, 2)))  # Empty 2D array
        return ground_truth_scatter, estimates_scatter

    # Update function for each frame
    def update(frame):
        if frame < len(ugvs):
            ground_truth_scatter.set_offsets(np.array(ugvs[frame]).reshape(-1, 2))  # Ensure 2D shape
        if frame < len(estimates):
            estimates_scatter.set_offsets(np.array(estimates[frame]).reshape(-1, 2))  # Ensure 2D shape
        return ground_truth_scatter, estimates_scatter

    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, repeat=False)

    plt.show()  # Display the animation

def plot_final_states(ground_truth, estimates):
    """Plot the final positions of UGVs for ground truth and estimates."""
    plt.figure(figsize=(8, 6))
    
    # Check if inputs are not empty
    if ground_truth.size == 0 or estimates.size == 0:
        print("No data to plot.")
        return
    
    plt.scatter(ground_truth[:, 0], ground_truth[:, 1], color='b', label='Ground Truth', marker='o')
    plt.scatter(estimates[:, 0], estimates[:, 1], color='g', label='Estimates', marker='x')
    
    plt.title("Final Positions of UGVs")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid()
    plt.legend()
    plt.axis('equal')
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
        P=P,
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
        animate=animate,
        plot_final_states=plot_final_states,
        do_s_attack_probability=0.1,  # Example probability for DoS attack
        fdi_attack_probability=0.2,   # Example probability for FDI attack
        max_dos_robots=2,             # Example maximum number of robots affected by DoS attack
        max_fdi_measurements=3        # Example maximum number of measurements affected by FDI attack
    )

    





