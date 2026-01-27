# ===============================
# Dataset / Pose Settings
# ===============================

# Maximum number of persons considered per frame
MAX_PERSONS = 6

# Number of consecutive frames in one temporal window
WINDOW_SIZE = 5

# Minimum normalized joint movement required to be considered meaningful
MAX_CHANGE_RATIO = 0.025


# ===============================
# Pose Keypoints Configuration
# ===============================

# Upper-body keypoints used for gesture analysis
REQUIRED_KEYPOINTS = [
    "neck",
    "right shoulder",
    "right elbow",
    "right wrist",
    "left shoulder",
    "left elbow",
    "left wrist",
]


# Flattened feature names corresponding to REQUIRED_KEYPOINTS (x, y only)
TRAIN_COLS = [
    "neck_x", "neck_y",
    "right shoulder_x", "right shoulder_y",
    "right elbow_x", "right elbow_y",
    "right wrist_x", "right wrist_y",
    "left shoulder_x", "left shoulder_y",
    "left elbow_x", "left elbow_y",
    "left wrist_x", "left wrist_y",
]


# ===============================
# Model & Training Settings
# ===============================

# Path to trained model (model file is NOT included in this repository)
MODEL_PATH = "Models/best_model.h5"

# Train–validation split ratio
TRAIN_VAL_SPLIT = 0.8