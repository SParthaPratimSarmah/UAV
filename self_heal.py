import random
import math
import matplotlib.pyplot as plt
import numpy as np

# Function to calculate users covered by a UAV at a given position
def users_covered_by_uav(uav_pos, users, coverage_radius, covered_users):
    new_covered_users = []
    for user in users:
        if user not in covered_users and distance(uav_pos, user.position) <= coverage_radius:
            new_covered_users.append(user)
    return new_covered_users

# Function to estimate the minimum number of UAVs
def estimate_uav_count(m, n, coverage_radius, k=0.75):
    area_covered_per_uav = np.pi * (coverage_radius ** 2)
    grid_area = m * n
    estimated_uavs = grid_area / (area_covered_per_uav * k)
    return math.floor(max(1, int(estimated_uavs)))  # Ensure at least 1 UAV is estimated

# Ant Colony Optimization for UAV Placement with limited UAVs
def self_healing_placement(m, n, users, num_ants, iterations, coverage_radius, max_uavs, threshold=0.95, k=15, best_record=0, alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=100):

    if iterations <=0 or max_uavs <= 0:
        return [], 0

    curr_iter = 0
    best_iter = 0
    pheromone = np.ones((m, n))  # Initialize pheromone matrix
    best_solution = None
    best_coverage = 0

    for iteration in range(iterations):
        curr_iter += 1
        all_solutions = [] 
        all_coverages = []

        # Each ant builds a solution (places UAVs)
        for ant in range(num_ants):
            uav_positions = []
            covered_users = set()

            # Limit the number of UAVs placed by each ant
            for _ in range(max_uavs):

                
                probabilities = np.zeros((m, n))

                # Calculate the probability of placing a UAV at each position
                for i in range(m):
                    for j in range(n):
                        heuristic_value = len(users_covered_by_uav([i, j], users, coverage_radius, covered_users))
                        probabilities[i, j] = (pheromone[i, j] ** alpha) * (heuristic_value ** beta)

                # Normalize the probabilities
                probabilities /= probabilities.sum()

                # Select the next UAV position based on probabilities
                uav_pos = np.unravel_index(np.argmax(np.random.multinomial(1, probabilities.flatten())), (m, n))
                uav_positions.append(uav_pos)

                # Update covered users with the new UAV placement
                new_covered_users = set(users_covered_by_uav(uav_pos, users, coverage_radius, covered_users))
                covered_users.update(new_covered_users)

                # Stop placing UAVs if 90% coverage is reached
                if len(covered_users) / len(users) >= threshold:
                    break

            # Record the solution and coverage
            all_solutions.append(uav_positions)
            all_coverages.append(len(covered_users) / len(users))

            # Update the best solution if coverage is better
            if len(covered_users) / len(users) > best_coverage:
                best_solution = uav_positions
                best_coverage = len(covered_users) / len(users)
                best_iter = curr_iter

            if curr_iter - best_iter > k:
                break

        # Update pheromones
        pheromone *= (1 - evaporation_rate)  # Evaporate pheromones
        for i in range(len(all_solutions)):
            for pos in all_solutions[i]:
                pheromone[pos[0], pos[1]] += Q * all_coverages[i]  # Deposit pheromones based on solution quality

        print(f"Iteration {iteration + 1}: Best coverage = {best_coverage:.2f}")

        # Stop if we have achieved the coverage threshold
        if best_coverage >= threshold:
            break

        if curr_iter - best_iter > k:
                break

    if best_coverage >= best_record:
        print(f"---------Searching for {max_uavs-1} UAVs ----------")
        rec_solution, rec_coverage = self_healing_placement(m, n, users, num_ants, iterations-curr_iter, coverage_radius, max_uavs-1, threshold, k, best_coverage, alpha, beta, evaporation_rate, Q)

        if rec_coverage >= best_coverage:
            return rec_solution, rec_coverage
    
    return best_solution, best_coverage

# Function to calculate angle between two points in degrees
def calculate_angle(uav1_pos, uav2_pos):
    delta_y = uav2_pos[1] - uav1_pos[1]
    delta_x = uav2_pos[0] - uav1_pos[0]
    angle = math.degrees(math.atan2(delta_y, delta_x))
    return angle if angle >= 0 else angle + 360


# Function to check if a UAV is within beamforming angle
def within_beamforming_angle(uav1_angle, uav2_angle, beamwidth=30):
    # Check if the angle between UAVs is within the beamwidth (30 degrees here as an example)
    return abs(uav1_angle - uav2_angle) <= beamwidth / 2


# Function to calculate distance between two points
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def move_users(users, m, n, step_size=0.5):
    for user in users:
        # Update x and y positions
        user.position[0] += random.uniform(-step_size, step_size)
        user.position[1] += random.uniform(-step_size, step_size)

        # Ensure users stay within grid boundaries
        user.position[0] = max(0, min(user.position[0], m))
        user.position[1] = max(0, min(user.position[1], n))


def visualize_grid(m, n, users, selected_uavs, coverage_radius, time_step):
    plt.figure(figsize=(10, 10))

    # Plot users
    for user in users:
        plt.plot(user.position[0], user.position[1], 'bo')

    # Plot UAV coverage
    for uav in selected_uavs:
        circle = plt.Circle(uav.position, uav.coverage_radius, color='r', alpha=0.3)
        plt.gca().add_patch(circle)
        plt.plot(uav.position[0], uav.position[1], 'rx', markersize=10)

    # Plot UAV backhaul connections
    for uav in selected_uavs:
        for connected_uav in uav.connected_uavs:
            plt.plot([uav.position[0], connected_uav.position[0]], [uav.position[1], connected_uav.position[1]], 'g--')

    # Set plot limits and labels
    plt.xlim(0, m)
    plt.ylim(0, n)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    plt.title(f'UAV Coverage and Backhaul on Grid at Time Step {time_step}')
    plt.show()


class UAV:
    def __init__(self, position, coverage_radius, backhaul_range, beamwidth):
        self.position = position  # [x, y]
        self.coverage_radius = coverage_radius
        self.backhaul_range = backhaul_range
        self.beamwidth = beamwidth  # Directional beamwidth in degrees
        self.covered_users = set()  # Set of User objects
        self.connected_uavs = set()  # Set of UAV objects
        self.beam_angle = 0  # Direction in which UAV is pointing its beam

    def update_coverage(self, users):
        self.covered_users = set()
        for user in users:
            if distance(self.position, user.position) <= self.coverage_radius:
                self.covered_users.add(user)
                user.connect_to_uav(self)

    def update_backhaul_connections(self, uavs):
        potential_connections = []
        for uav in uavs:
            if uav != self and distance(self.position, uav.position) <= self.backhaul_range:
                potential_connections.append(uav)

        # Sort potential connections by distance (closest first)
        potential_connections.sort(key=lambda uav: distance(self.position, uav.position))

        # Update the beam angle towards the closest UAV
        if potential_connections:
            closest_uav = potential_connections[0]
            self.beam_angle = calculate_angle(self.position, closest_uav.position)

            # Connect only to UAVs within the beamforming angle
            for uav in potential_connections:
                uav_angle = calculate_angle(self.position, uav.position)
                if within_beamforming_angle(self.beam_angle, uav_angle, self.beamwidth):
                    self.connected_uavs.add(uav)

# User Class
class User:
    def __init__(self, position):
        self.position = position  # position is a list [x, y]
        self.connected_uav = None

    def connect_to_uav(self, uav):
        self.connected_uav = uav

    @staticmethod
    def generate_random_users(m, n, num_users):
        users = []
        for _ in range(num_users):
            x = random.uniform(0, m)
            y = random.uniform(0, n)
            users.append(User([x, y]))
        return users

# Simulation parameters
m, n = 10, 10  # Grid size
num_users = 50  # Number of users
coverage_radius = 2.0  # Coverage radius of each UAV
backhaul_range = 12.0  # Backhaul connection range between UAVs
time_steps = 10  # Number of time steps to simulate
min_threshold = 0.90  # Minimum threshold value of user coverage
num_ants = 20  # Number of ants
iterations = 100  # Maximum number of iterations
max_uavs_per_ant=estimate_uav_count(m,n,coverage_radius,0.75)
threshold=0.95 # Threshold value of user coverage
k=15 # Allowed repition with same coverage
# Initialize the list of operational UAVs

# Initialize users and UAVs
users = User.generate_random_users(m, n, num_users)
uavs, _ = self_healing_placement(m, n, users, num_ants, iterations, coverage_radius,max_uavs_per_ant,threshold,k)
uavs = [UAV(uav, coverage_radius, backhaul_range, 30) for uav in uavs]

for uav in uavs:
    uav.update_coverage(users)
    uav.update_backhaul_connections(uavs)

# Modify the time-stepped simulation to include UAV failure handling
failed_uavs = set()

# Initialize the list of operational UAVs
operational_uavs = uavs[:]  # Start with all UAVs as operational

# Initialize lists to store time-step data
coverage_over_time = []
uavs_over_time = []

# Time-stepped simulation
# Ensure coverage is captured correctly and percentages are displayed
for time_step in range(time_steps):
    # Simulate UAV failure at random
    if random.random() < 0.2:  # Adjust probability as needed
        num_failures = random.randint(1, 2)  # Fail 1 or 2 UAVs randomly
        failed_uav_indices = random.sample(range(len(uavs)), num_failures)
        
        for idx in failed_uav_indices:
            failed_uavs.add(uavs[idx])

        # Update the list of operational UAVs after failure
        operational_uavs = [uav for uav in uavs if uav not in failed_uavs]

        print(f"Time Step {time_step}: UAV failure detected. Adjusting positions...")

        # Update coverage and connections for operational UAVs
        for uav in operational_uavs:
            uav.update_coverage(users)
            uav.update_backhaul_connections(operational_uavs)

    # Move users
    move_users(users, m, n)

    # Recalculate UAV coverage after user movement and failure adjustments
    for uav in operational_uavs:
        uav.update_coverage(users)
        uav.update_backhaul_connections(operational_uavs)

    # Calculate the coverage percentage
    covered_users = set([u for uav in operational_uavs for u in uav.covered_users])
    current_coverage = (len(covered_users) / num_users) * 100  # Convert to percentage

    # Print time step information with required metrics
    print(f"Time Step {time_step}: UAV Count = {len(operational_uavs)}, Coverage = {current_coverage:.2f}%")

    # Record the coverage and UAV count for this time step
    coverage_over_time.append(current_coverage)
    uavs_over_time.append(len(operational_uavs))

    # Check if coverage drops below the minimum threshold
    if current_coverage < min_threshold * 100:  # Compare with min_threshold as a percentage
        print(f"Coverage below threshold at time step {time_step}, running local search...")

        # Rerun the ant colony or a simple repositioning algorithm to restore coverage
        uavs, _ = self_healing_placement(m, n, users, num_ants, iterations, coverage_radius, max_uavs_per_ant, threshold, k)
        uavs = [UAV(uav, coverage_radius, backhaul_range, 30) for uav in uavs]

        # Reset failed UAVs and update operational UAVs
        failed_uavs.clear()
        operational_uavs = uavs[:]

    # Visualize the grid and UAV coverage
    visualize_grid(m, n, users, operational_uavs, coverage_radius, time_step)

# Plot the graph after the simulation ends
# Plot Coverage vs Time Step
plt.figure(figsize=(12, 6))
plt.plot(range(time_steps), coverage_over_time, label="Coverage (%)", color="b")
plt.xlabel("Time Steps")
plt.ylabel("Coverage (%)")
plt.title("User Coverage Over Time")
plt.legend()
plt.grid(True)
plt.show()

# Plot Number of UAVs vs Time Step
plt.figure(figsize=(12, 6))
plt.plot(range(time_steps), uavs_over_time, label="Number of UAVs", color="r")
plt.xlabel("Time Steps")
plt.ylabel("Number of UAVs")
plt.title("UAV Count Over Time")
plt.legend()
plt.grid(True)
plt.show()
