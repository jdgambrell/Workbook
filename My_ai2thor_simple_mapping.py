from ai2thor.controller import Controller
import math
import time
import numpy as np
import cv2
import open3d as o3d


# Rotation is along the z-axis here, Locobot model cannot move left/right, and up/down motion is still not integrated yet

#action: movements along left/right (+1,-1), forward/back (+1,-1), rotate

# Initialize the controller
controller = Controller(
    agentMode="locobot",  # Define the agent mode (locobot for this example)
    visibilityDistance=5.0,  # How far the agent can see
    scene="FloorPlan210",  # Choose a scene to load
    movementGaussianSigma=0.005,  # Movement noise
    rotateGaussianSigma=0.5,  # Rotation noise
    renderDepthImage=True,  # Option to render depth images                             -- May take a full frame to update
    renderInstanceSegmentation=False,  # Option to render instance segmentation
    width=300,  # Image width for rendering
    height=300,  # Image height for rendering
    fieldOfView=60,  # Camera field of view in degrees
    use_gpu=True,  # Use GPU for rendering
    gridSize=0.05,  # Step sizes
    snapToGrid=False,
    rotateStepDegrees=10
)

# Function for Keyboard control (arrows for movement, and r/e for rotation)
def keyboardMotion(controller, event):
	current_r = event.metadata['agent']['rotation']['y']
	print("Enter command (w/s/r/l): ", end='', flush=True)
	cc = input().strip().lower()
	rotate_unit = 10
	steps = 1
	if cc=='w':
		event = controller.step(dict(action='MoveAhead', moveMagnitude=0.25*steps))
	if cc=='s':
		event = controller.step(dict(action='MoveBack', moveMagnitude=0.25*steps))
	if cc=='r':
		event = controller.step(dict(action='Rotate', rotation = current_r+rotate_unit))
	if cc=='l':
		event = controller.step(dict(action='Rotate', rotation = current_r-rotate_unit))

	return event


# Initialize event -- This is done so that event has a value the first time it is passed
event = controller.step(action='GetReachablePositions')


# Get and save the agent's initial position and rotation
agent_position = event.metadata['agent']['position']
print(f"Initial Agent Position: {agent_position}")
agent_rotation = event.metadata['agent']['rotation']
print(f"Initial Agent Rotation: {agent_rotation}")


# Requires pressing enter key to move on in the script
input("start")
time.sleep(0.5)


# Move ahead and log new position
event = controller.step(dict(action='MoveAhead', moveMagnitude=0.25))
time.sleep(0.5)

agent_position = event.metadata['agent']['position']
print(f"New Agent Position: {agent_position}")
agent_rotation = event.metadata['agent']['rotation']
print(f"New Agent Rotation: {agent_rotation}")


# To fix issue with display lagging a step behind, the fix is usually to request 
# another no-op action (like Pass) to ensure the image updates with the new position:
event = controller.step(action='Pass')
event.cv2image



################################################################################################################

# Camera intrinsic matrix
K = np.array([[150, 0, 150], 
              [0, 150, 150], 
              [0,   0,   1]])

fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]



# The global point cloud
global_pcd = o3d.geometry.PointCloud()

# Define a few positions and rotations to move the agent through
poses = [
    {'x': 1, 'z': -1, 'rotation': 0},
    {'x': 1.25, 'z': -1, 'rotation': 45},
    {'x': 1.25, 'z': -0.75, 'rotation': 90},
]

for pose in poses:
    event = controller.step(action='Teleport', position={'x': pose['x'], 'y': 0.9, 'z': pose['z']}, 
                    rotation={'x': 0, 'y': pose['rotation'], 'z': 0})
    event = controller.step(action='Pass')
	
    
    print(f"New Pose {pose}")
    agent_position = event.metadata['agent']['position']
    print(f"New Agent Position: {agent_position}")
    agent_rotation = event.metadata['agent']['rotation']
    print(f"New Agent Rotation: {agent_rotation}")

    rgb = np.array(event.frame)
    depth = np.array(event.depth_frame)

    # Skip empty depth frames
    if depth is None or depth.size == 0:
        continue

    height, width = depth.shape
    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')

    z = depth
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # ---- Calculate transformation matrix from pose ----       --> "APPLY TRANSFORMATION TO WORLD FRAME"
    rotation_y = math.radians(pose['rotation'])
    cos_r, sin_r = math.cos(rotation_y), math.sin(rotation_y)
    transform = np.array([
        [ cos_r, 0, -sin_r, pose['x']],
        [     0, 1,      0,     0.9],
        [ sin_r, 0,  cos_r, pose['z']],
        [     0, 0,      0,      1]
    ])

    pcd.transform(transform)
    global_pcd += pcd

# Visualize final accumulated map
o3d.visualization.draw_geometries([global_pcd])


# Save the map
#o3d.io.write_point_cloud("scene_map.ply", global_pcd)




'''
#Point cloud from single step

rgb = np.array(event.frame)
depth = np.array(event.depth_frame)

# Generate pixel coordinate grid
height, width = depth.shape
i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')

# Convert depth to 3D camera coordinates
z = depth
x = (i - cx) * z / fx
y = (j - cy) * z / fy

points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
colors = rgb.reshape(-1, 3) / 255.0  # normalize colors

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Optional: transform using robot pose (assumed as identity here)
robot_pose = np.eye(4)  # Replace with actual pose matrix if known
pcd.transform(robot_pose)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])
'''




'''
# Loop for keyboard-based movement
while (1):
	
    event = keyboardMotion(controller, event)
    event = controller.step(action='Pass')          # Force an update so image matches latest agent state
    event.cv2image
    time.sleep(0.1)
'''







