import rps.robotarium as robotarium
import numpy as np
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import logging

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
    "PROCESS_NOISE_COVARIANCE": np.eye(3) * 0.01,  # 3x3 identity matrix scaled by 0.01.
    
    # Threshold for triggering events in the control strategy.
    "EVENT_TRIGGER_THRESHOLD": 0.1,
    
    # Small constant to avoid division by zero in calculations.
    "EPSILON": 1e-6,
    
    # Standard deviation of noise in range measurements.
    "RANGE_NOISE_STD": 0.1,
    
    # Standard deviation of noise in bearing measurements.
    "BEARING_NOISE_STD": 0.05,
    
    # Operational area limits for the x-coordinate.
    "x_limits": (0, 10),
    
    # Operational area limits for the y-coordinate.
    "y_limits": (0, 10),
    
    # Dimensions of the target area for the robots.
    "TARGET_AREA": (100, 100),
    
    # Number of steps in the simulation or control loop.
    "num_steps": 100,

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
    "FOLLOWER_INDICES": [i for i in range(1, 10)],  # Automatically set for TOTAL_ROBOTS=10.

    # Desired trajectory for the leader to follow.
    # This can be dynamically updated during the simulation if necessary.
    "desired_trajectory": np.array([5.0, 5.0]),  # Example initial target position for the leader.
    
    # Placeholder for storing follower indices for easier referencing in follower control.
    "FOLLOWER_INDEX": 1  # Default for follower; update dynamically during the simulation if needed.
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

def generate_initial_positions(num_robots, x_range, y_range):
    """Generates random initial positions for the robots."""
    logging.debug("Generating initial positions for robots.")
    initial_positions = np.random.rand(num_robots, 2) * [x_range[1] - x_range[0], y_range[1] - y_range[0]] + [x_range[0], y_range[0]]
    logging.info(f"Initial positions generated: {initial_positions}")
    return initial_positions

def generate_cubature_points(P, x_hat):
    """Generates cubature points."""
    logging.debug("Generating cubature points.")
    n = x_hat.shape[0]
    P += CONFIG["EPSILON"] * np.eye(n)  # Ensure positive definiteness
    S = cholesky(P, lower=True)
    
    cubature_points = np.zeros((2 * n, n))
    for i in range(n):
        cubature_points[i] = np.sqrt(n) * S[:, i] + x_hat
        cubature_points[i + n] = -np.sqrt(n) * S[:, i] + x_hat
    logging.info("Cubature points generated successfully.")
    return cubature_points

def time_update(x_hat, P, Q, control_input, f):
    """Performs the CKF time update (prediction)."""
    logging.debug("Performing time update.")
    cubature_points = generate_cubature_points(P, x_hat)

    # Propagate cubature points through the transition function f
    propagated_points = np.array([f(point, control_input) for point in cubature_points])

    # Calculate predicted state mean and covariance
    x_hat_pred = np.mean(propagated_points, axis=0)
    P_pred = np.cov(propagated_points, rowvar=False) + Q
    logging.info("Time update completed.")
    return x_hat_pred, P_pred

def measurement_update(x_hat_pred, P_pred, measurement, R, h, nearby_positions, initial_positions, nearby_robots, index, 
                       previous_innovation, adaptive_threshold, shadow_measurement, shadow_R):
    """
    Performs the CKF measurement update (correction) with event-triggered communication and attack detection,
    considering both regular and shadow edge measurements.
    """
    logging.debug(f"Performing measurement update for UGV {index}.")
    
    n = x_hat_pred.shape[0]
    cubature_points = generate_cubature_points(P_pred, x_hat_pred)

    # Propagate cubature points through the measurement function h for regular measurements
    measurement_points = np.array([h(point, nearby_positions, initial_positions, index) for point in cubature_points])

    # Calculate predicted measurement mean and covariance for regular measurements
    z_hat_pred = np.mean(measurement_points, axis=0)
    covariance_matrix = np.cov(measurement_points, rowvar=False)

    # Adjust the dimensions of R for regular measurements
    num_features = min(covariance_matrix.shape[0], 2 * nearby_robots)
    R_resized = np.resize(R, (num_features, num_features))  # Resize R to match covariance_submatrix
    P_zz = covariance_matrix[:num_features, :num_features] + R_resized

    # Calculate cross-covariance matrix P_xz for regular measurements
    P_xz = np.zeros((n, num_features))  # Initialize with the shape (state_dim, measurement_dim)
    for i in range(2 * n):
        outer_product = np.outer(cubature_points[i] - x_hat_pred, measurement_points[i, :num_features] - z_hat_pred[:num_features])
        P_xz += outer_product
    P_xz /= (2 * n)

    # Regularization for P_zz
    if np.linalg.cond(P_zz) > 1e10:
        P_zz += CONFIG["EPSILON"] * np.eye(P_zz.shape[0])

    # Calculate Kalman gain for regular measurements
    K_regular = P_xz @ np.linalg.inv(P_zz)

    # Compute the innovation (residual) for regular measurements
    innovation_regular = measurement[:num_features] - z_hat_pred[:num_features]

    # Handle shadow measurements
    # Propagate cubature points through the measurement function h for shadow measurements
    shadow_measurement_points = np.array([h(point, nearby_positions, initial_positions, index) for point in cubature_points])

    # Calculate predicted measurement mean and covariance for shadow measurements
    shadow_z_hat_pred = np.mean(shadow_measurement_points, axis=0)
    shadow_covariance_matrix = np.cov(shadow_measurement_points, rowvar=False)

    # Adjust the dimensions of shadow_R
    shadow_R_resized = np.resize(shadow_R, (num_features, num_features))  # Resize shadow R to match covariance_submatrix
    P_zz_shadow = shadow_covariance_matrix[:num_features, :num_features] + shadow_R_resized

    # Calculate cross-covariance matrix P_xz for shadow measurements
    P_xz_shadow = np.zeros((n, num_features))  # Initialize with the shape (state_dim, measurement_dim)
    for i in range(2 * n):
        outer_product = np.outer(cubature_points[i] - x_hat_pred, shadow_measurement_points[i, :num_features] - shadow_z_hat_pred[:num_features])
        P_xz_shadow += outer_product
    P_xz_shadow /= (2 * n)

    # Regularization for P_zz_shadow
    if np.linalg.cond(P_zz_shadow) > 1e10:
        P_zz_shadow += CONFIG["EPSILON"] * np.eye(P_zz_shadow.shape[0])

    # Calculate Kalman gain for shadow measurements (half of the regular)
    K_shadow = 0.5 * P_xz_shadow @ np.linalg.inv(P_zz_shadow)

    # Combine updates from regular and shadow measurements
    x_hat_updated = x_hat_pred + K_regular @ innovation_regular + K_shadow @ (shadow_measurement - shadow_z_hat_pred)
    
    # Update covariance
    P_updated = P_pred - K_regular @ P_zz @ K_regular.T - K_shadow @ P_zz_shadow @ K_shadow.T

    # Ensure covariance matrix remains symmetric and positive definite
    P_updated = (P_updated + P_updated.T) / 2
    P_updated += CONFIG["EPSILON"] * np.eye(n)

    # Check for attack detection
    if not attack_detected(innovation_regular):
        # Event-triggered communication
        event_trigger, updated_threshold = event_triggered(
            innovation_regular, previous_innovation, adaptive_threshold
        )

        if event_trigger:
            logging.info(f"Event triggered for UGV {index}. Communicating state to neighbors.")
            # Return updated values
            return x_hat_updated, P_updated, innovation_regular, updated_threshold
        else:
            logging.info(f"No event triggered for UGV {index}. Using predicted state.")
            # No event triggered, return the predicted values
            return x_hat_pred, P_pred, previous_innovation, adaptive_threshold
    else:
        logging.warning(f"Attack detected for UGV {index}. Ignoring the measurement update.")
        # In case of an attack, ignore the measurement update and use the predicted state
        return x_hat_pred, P_pred, previous_innovation, adaptive_threshold



def state_transition(x, u):
    """Example state transition function."""
    x_new = np.copy(x)
    x_new[0] += u[0] * np.cos(x[2])  # x-position update
    x_new[1] += u[0] * np.sin(x[2])  # y-position update
    x_new[2] += u[1]  # Theta update
    return x_new

def control_barrier_function(x_i, x_j, d_min):
    """
    Computes the control barrier function value between robots i and j.
    """
    distance_squared = np.sum((x_i - x_j)**2)
    return distance_squared - d_min**2

def apply_cbf_control(input_control, x_i, x_j, d_min, CONFIG):
    """
    Modifies the control input using Control Barrier Function to avoid collisions.
    """
    h_ij = control_barrier_function(x_i, x_j, d_min)

    # If the barrier function is violated (too close), adjust the control input
    if h_ij < 0:
        # Apply a repulsive force to prevent collision
        direction = (x_i - x_j) / (np.linalg.norm(x_i - x_j) + 1e-6)  # normalize direction
        repulsive_force = CONFIG["cbf_gain"] * direction / (np.linalg.norm(x_i - x_j) + 1e-6)
        input_control += repulsive_force
    
    # Ensure the control input stays within limits
    return np.clip(input_control, -CONFIG["CONTROL_INPUT_LIMIT"], CONFIG["CONTROL_INPUT_LIMIT"])

def leader_control_policy(current_position, target_position, positions, d_min, CONFIG):
    """
    Control policy for the leader robot with Control Barrier Function to avoid collisions.
    """
    # Regular PID control to track the target position
    control_input = pid_control(current_position, target_position, CONFIG, "leader")

    # Check distances to all other robots and apply CBF if necessary
    for i in range(1, CONFIG["TOTAL_ROBOTS"]):  # Avoid checking itself
        control_input = apply_cbf_control(control_input, current_position, positions[:, i], d_min, CONFIG)
    
    return control_input

def follower_control_policy(current_position, desired_trajectory, positions, d_min, CONFIG):
    """
    Control policy for follower robots with Control Barrier Function to avoid collisions.
    """
    # Regular PID control to track the desired trajectory (leader's position)
    control_input = pid_control(current_position, desired_trajectory, CONFIG, "follower")

    # Check distances to all other robots and apply CBF if necessary
    for i in range(CONFIG["TOTAL_ROBOTS"]):  # Avoid checking itself
        if i != CONFIG["FOLLOWER_INDEX"]:  # Avoid checking itself
            control_input = apply_cbf_control(control_input, current_position, positions[:, i], d_min, CONFIG)
    
    return control_input

def pid_control(current_position, target_position, CONFIG, robot_type):
    """
    General PID control logic used for both leader and follower robots.
    """
    integral = CONFIG[f"integral_{robot_type}"]
    previous_error = CONFIG[f"previous_error_{robot_type}"]

    # Calculate the error
    error = target_position - current_position

    # Proportional term
    proportional = CONFIG[f"kp_{robot_type}"] * error

    # Integral term
    integral += CONFIG[f"ki_{robot_type}"] * error * CONFIG["dt"]

    # Derivative term
    derivative = CONFIG[f"kd_{robot_type}"] * (error - previous_error) / CONFIG["dt"]

    # Calculate control input
    control_input = proportional + integral + derivative

    # Update previous error in CONFIG
    CONFIG[f"previous_error_{robot_type}"] = error

    # Limit control input to a specified range
    return np.clip(control_input, -CONFIG["CONTROL_INPUT_LIMIT"], CONFIG["CONTROL_INPUT_LIMIT"])



def apply_process_noise(control_input):
    """
    Applies process noise to the control input.
    """
    noise = np.random.normal(0, CONFIG["CONTROL_NOISE_STD"], size=control_input.shape)
    control_input_noisy = control_input + noise
    return control_input_noisy

def range_measurement(own_position, nearby_positions, initial_positions):
    """
    Calculate the range measurement between the UGV's current position and nearby positions.
    """
    own_position = own_position.reshape(-1)  # Ensure (2,)
    range_to_initial = np.linalg.norm(own_position - initial_positions)  # Scalar
    ranges_to_nearby = np.linalg.norm(nearby_positions - own_position, axis=1)  # Shape: (N,)
    ranges = np.concatenate([[range_to_initial], ranges_to_nearby])
    return ranges

def bearing_measurement(own_position, own_orientation, nearby_positions, initial_position):
    """
    Calculates the bearing measurements to nearby UGVs and the initial position.
    """
    vector_to_initial = initial_position - own_position
    angle_to_initial = np.arctan2(vector_to_initial[1], vector_to_initial[0])
    initial_bearing = angle_to_initial - own_orientation
    initial_bearing = np.arctan2(np.sin(initial_bearing), np.cos(initial_bearing))

    vectors_to_nearby = nearby_positions - own_position  # Shape: (N, 2)
    angles_to_nearby = np.arctan2(vectors_to_nearby[:, 1], vectors_to_nearby[:, 0])
    bearings = angles_to_nearby - own_orientation

    bearings = np.arctan2(np.sin(bearings), np.cos(bearings))
    return np.concatenate(([initial_bearing], bearings))

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
                if dist < threshold:
                    rho += 1.0 / dist  # Closer robots contribute more to rho

    return rho


def shadow_range_measurement(positions, distances, errors, rho):
    """
    Calculate the shadow range measurement considering noise and distance constraints.
    
    :param positions: A dictionary of node positions {'v_i': (x, y), ...}
    :param distances: A dictionary of known distances {'d_ij': distance, ...}
    :param errors: A dictionary of errors {'eta_ij': error, ...}
    :param rho: Minimum distance threshold for shadow edges
    :return: A dictionary with shadow range measurements and their uncertainties
    """
    shadow_ranges = {}
    
    for (i, j) in distances.keys():
        d_ih = distances.get((i, 'h'), 0)
        d_jh = distances.get((j, 'h'), 0)
        d_ij = distances.get((i, j), 0)
        
        eta_ih = errors.get((i, 'h'), 0)
        eta_jh = errors.get((j, 'h'), 0)

        # Check conditions for shadow edges
        if d_ij <= rho or d_ij > 2 * rho:
            continue  # Not a valid shadow edge
        
        # Calculate angles using the cosine rule
        alpha_khj = np.arccos((d_jh**2 + d_ih**2 - d_ij**2) / (2 * d_jh * d_ih))
        
        # Calculate estimated shadow edge length
        d_ij_estimated = np.sqrt(d_ih**2 + d_jh**2 - 2 * d_ih * d_jh * np.cos(alpha_khj))

        # Noise adjustment for shadow range
        eta_ij = (
            2 * d_jh * d_ih * np.cos(alpha_khj) +
            eta_jh + eta_ih -
            2 * np.sqrt((d_jh**2 + eta_jh) * (d_ih**2 + eta_ih)) * np.cos(alpha_khj)
        )
        
        shadow_ranges[(i, j)] = (d_ij_estimated, eta_ij)
    
    return shadow_ranges


def shadow_bearing_measurement(positions):
    """
    Calculate the shadow bearing measurement between nodes.
    
    :param positions: A dictionary of node positions {'v_i': (x, y), ...}
    :return: A dictionary with shadow bearing measurements
    """
    shadow_bearings = {}
    
    for (i, j) in positions.keys():
        pos_i = positions[i]
        pos_j = positions[j]
        
        # Calculate bearing angle
        dy = pos_j[1] - pos_i[1]
        dx = pos_j[0] - pos_i[0]
        
        # Bearing angle in radians
        shadow_bearing = np.arctan2(dy, dx)  # Angle in radians
        
        shadow_bearings[(i, j)] = shadow_bearing
    
    return shadow_bearings



def measurement_function(robot_position, nearby_positions, initial_positions, index):
    """
    Constructs a measurement vector from the ground truth.
    """
    current_position = robot_position[:2]  # Shape: (2,)
    initial_position = initial_positions[index]  # Extract specific initial position for the current UGV
    
    ranges = range_measurement(current_position, nearby_positions, initial_position)
    bearings = bearing_measurement(current_position, robot_position[2], nearby_positions, initial_position)
    
    measurement = np.concatenate([ranges, bearings])
    return measurement

def event_triggered(innovation, prev_innovation, threshold=None, decay_factor=0.95):
    """
    Checks if the event-triggered condition is met with an adaptive threshold.
    """
    threshold = threshold or CONFIG["EVENT_TRIGGER_THRESHOLD"]
    innovation_norm = np.linalg.norm(innovation)
    adaptive_threshold = max(threshold, decay_factor * np.linalg.norm(prev_innovation))

    logging.debug(f"Innovation norm: {innovation_norm}, Adaptive threshold: {adaptive_threshold}")

    # Event triggering logic
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


def run_simulation(robotarium_env, CONFIG, x_hat, P, positions, target_positions, 
                   apply_process_noise, time_update, measurement_function, 
                   measurement_update, attack_detected, get_neighbors, event_triggered, 
                   generate_initial_positions, range_measurement, bearing_measurement, 
                   shadow_range_measurement, shadow_bearing_measurement,  # New shadow measurement functions
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
        shadow_bearing_measurement: Function for shadow edge bearing measurements.
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

    # Simulation Loop
    for step in range(CONFIG["num_steps"]):  
        logging.debug(f"Step {step + 1}/{CONFIG['num_steps']}")

        # Get ground truth poses before stepping
        ground_truth = robotarium_env.get_poses()
        logging.debug(f"Ground truth generated: {ground_truth}")

        # Step the Robotarium environment
        robotarium_env.step()
        plt.pause(0.01)  # Pause for a brief moment to allow the plot to update

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
            initial_positions = generate_initial_positions(CONFIG["TOTAL_ROBOTS"], x_range, y_range)

            # Compute regular range and bearing measurements
            range_meas = range_measurement(x_hat[:2, i, step - 1], nearby_positions, initial_positions[i])
            bearing_meas = bearing_measurement(x_hat[:2, i, step - 1], x_hat[2, i, step - 1], nearby_positions, initial_positions[i])

            # Compute shadow edge measurements
            shadow_neighbors_positions = np.array([x_hat[:2, j, step] for j in shadow_neighbors])
            shadow_ranges = shadow_range_measurement(positions, distances, errors, rho)
            shadow_range_meas = [shadow_ranges.get((i, j), (0, 0))[0] for j in shadow_neighbors]
            shadow_bearing_meas = shadow_bearing_measurement(positions)

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
                    state_transition
                )

            # Check for FDI attack affecting limited measurements
            if np.random.rand() < fdi_attack_probability:
                # Randomly select a limited number of measurements to corrupt
                fdi_indices = np.random.choice(len(range_meas), size=min(max_fdi_measurements, len(range_meas)), replace=False)
                for idx in fdi_indices:
                    range_meas[idx] += np.random.normal(0, 0.5)  # Corrupt range measurement
                    bearing_meas[idx] += np.random.normal(0, 0.5)  # Corrupt bearing measurement

            combined_measurements = np.concatenate([range_meas, bearing_meas])
            logging.info(f"Combined measurements generated for robot {i}: {combined_measurements}")

            # Leader control
            if i == leader_index:
                leader_position = positions[:, leader_index, step - 1]
                target_position = target_positions[:, leader_index, step - 1]
                control_input = leader_control_policy(leader_position, target_position, positions, d_min, CONFIG)
            else:
                # Follower control
                desired_trajectory = positions[:, leader_index, step]  # Desired trajectory shared with followers
                control_input = follower_control_policy(positions[:, i, step - 1], desired_trajectory, positions, d_min, CONFIG)

            control_input_noisy = apply_process_noise(control_input)
            positions[:, i, step] = x_hat[0:2, i, step]  # Update position based on the estimated state

            # Update estimated states and covariance only if DoS attack is not active
            if i not in dos_attacked_robots:
                # Count the number of nearby robots excluding the current robot i
                number_of_nearby_robots = len([j for j in neighbors if j != i]) + len(shadow_neighbors)  # Include shadow neighbors
                x_hat[:, i, step], P[:, :, i, step], previous_innovations[i], adaptive_thresholds[i] = measurement_update(
                    x_hat_pred[:, i, step], P_pred[:, :, i, step], 
                    combined_measurements, MEASUREMENT_NOISE_COVARIANCE, 
                    measurement_function(x_hat[:, i, step], nearby_positions, initial_positions[i], i), 
                    nearby_positions, initial_positions[i], number_of_nearby_robots, i, 
                    previous_innovations[i], adaptive_thresholds[i],
                    np.concatenate([shadow_range_meas, shadow_bearing_meas]),  # Include shadow measurements
                    MEASUREMENT_NOISE_COVARIANCE  # Use the same covariance for shadow measurements
                )

            # Update estimated positions for the next step
            ground_truth_positions.append(ground_truth[:, i])
            estimated_positions.append(x_hat[:, i, step])

        # Update the Robotarium environment with the new positions
        robotarium_env.set_robot_positions(positions[:, :, step])

    # Finalizing the simulation
    logging.debug(f"Final step {step}: current positions {positions}")

    # Convert to numpy arrays for further processing
    ground_truth_positions = np.array(ground_truth_positions)
    estimated_positions = np.array(estimated_positions)

    # Run the animation
    animate(ground_truth_positions, estimated_positions, CONFIG["num_steps"])

    # Plot final states with comparisons
    plot_final_states(ground_truth_positions, estimated_positions)

    # Clean up the Robotarium environment
    robotarium_env.destroy()
    logging.info("Simulation completed.")



def animate(ugvs, estimates, num_steps):
    """Real-time animation of UGV positions comparing ground truth and estimates."""
    fig, ax = plt.subplots()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Define a color map for UGVs
    markers = ['o', 's', '^', 'D', 'p', '*', 'X']  # Different markers for UGVs
    
    def update(frame):
        ax.clear()  # Clear previous frame
        ax.set_xlim(-10, 10)  # Set limits based on your simulation
        ax.set_ylim(-10, 10)
        ax.set_title("UGV Positions Over Time")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        
        # Plot each UGV's ground truth and estimated positions
        for i, (ugv, estimate) in enumerate(zip(ugvs, estimates)):
            # Plot ground truth
            ax.plot(ugv[:frame, 0], ugv[:frame, 1], color=colors[i % len(colors)], 
                    marker=markers[i % len(markers)], label=f'UGV {i} Ground Truth', alpha=0.6)
            # Plot estimated positions
            ax.plot(estimate[:frame, 0], estimate[:frame, 1], color=colors[i % len(colors)], 
                    linestyle='--', label=f'UGV {i} Estimate', alpha=0.8)
            # Plot differences between estimates and ground truth
            ax.scatter(estimate[frame-1, 0], estimate[frame-1, 1], color='k', 
                       marker='x', s=100, label=f'UGV {i} Difference' if frame == 1 else "")
            # Optionally, draw lines between the estimate and the ground truth
            ax.plot([ugv[frame-1, 0], estimate[frame-1, 0]], 
                    [ugv[frame-1, 1], estimate[frame-1, 1]], color='orange', linestyle=':')

        ax.legend(loc='upper right')
    
    ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=100)
    plt.show()



def plot_final_states(ugvs, estimates):
    """Plot the final states of UGVs with comparisons to ground truth."""
    plt.figure()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    markers = ['o', 's', '^', 'D', 'p', '*', 'X']
    
    for i, (ugv, estimate) in enumerate(zip(ugvs, estimates)):
        # Plot ground truth
        plt.plot(ugv[:, 0], ugv[:, 1], color=colors[i % len(colors)], 
                 marker=markers[i % len(markers)], label=f'UGV {i} Ground Truth', alpha=0.6)
        # Plot estimated positions
        plt.plot(estimate[:, 0], estimate[:, 1], color=colors[i % len(colors)], 
                 linestyle='--', label=f'UGV {i} Estimate', alpha=0.8)
        # Highlight the final positions
        plt.scatter(ugv[-1, 0], ugv[-1, 1], color='black', marker='o', s=100, edgecolor='k', label=f'Final Ground Truth UGV {i}')
        plt.scatter(estimate[-1, 0], estimate[-1, 1], color='orange', marker='x', s=100, edgecolor='k', label=f'Final Estimate UGV {i}')

    plt.title("Final UGV States with Comparison")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid()
    plt.axis('equal')  # Keep aspect ratio
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
        shadow_range_measurement=shadow_range_measurement,  # Include shadow measurements
        shadow_bearing_measurement=shadow_bearing_measurement,
        leader_control_policy=leader_control_policy,
        follower_control_policy=follower_control_policy,
        animate=animate,
        plot_final_states=plot_final_states,
        do_s_attack_probability=0.1,  # Example probability for DoS attack
        fdi_attack_probability=0.2,     # Example probability for FDI attack
        max_dos_robots=2,               # Example maximum number of robots affected by DoS attack
        max_fdi_measurements=3,         # Example maximum number of measurements affected by FDI attack
        distances=distances,             # Pass distances to the simulation
        errors=errors,                   # Pass errors to the simulation
        rho=rho                          # Pass rho to the simulation
    )



