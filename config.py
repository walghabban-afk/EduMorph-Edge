"""
config.py - EduMorph-Edge configuration.

Paper: EduMorph-Edge: AI-Driven Adaptive Personalization for Student
       Engagement and Learning Outcomes in Cloud-Supported Classrooms
       (W. Alghabban and R. Salama)

Hyperparameters follow Appendix A of the paper. Reported study results are
kept in paper_values.py and are used only by verify_statistics.py.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

# ---------------------------------------------------------------------------
# Engagement-classifier labelling set (Section III-A, Appendix A)
# 240 students, 14,400 five-minute windows (60 per student on average),
# split at the student level: 68% training, 32% held-out test.
# Model selection uses a validation subset drawn from the training students.
# ---------------------------------------------------------------------------
LABEL_SET_STUDENTS = 240
WINDOWS_PER_STUDENT = 60
TRAIN_FRACTION = 0.68
VAL_FRACTION_OF_TRAIN = 0.15
ENGAGE_LABELS = {0: "Disengaged", 1: "Transitioning", 2: "Engaged"}

# ---------------------------------------------------------------------------
# RCT design (Section III-C)
# ---------------------------------------------------------------------------
N_STUDENTS_EXP = 150
N_STUDENTS_CTRL = 150
N_SCHOOLS = 9
INTERVENTION_WEEKS = 12
SUBJECTS = ["Mathematics", "Science", "English", "Social Studies"]

# ---------------------------------------------------------------------------
# MSC-Trans engagement classifier (Appendix A.3, architecture of [30])
# ---------------------------------------------------------------------------
FACE_AU_DIM = 128          # on-device facial action-unit vector (ResNet-18 + projection)
CLICK_DIM = 64             # aggregated clickstream sequence length
TEXT_DIM = 768             # DistilBERT CLS embedding
MSC_CNN_KERNELS = [3, 5, 7]
MSC_TEXT_MODEL = "distilbert-base-uncased"
MSC_ATTENTION_HEADS = 4
MSC_EMBED_DIM = 512
MSC_N_CLASSES = 3
MSC_DROPOUT = 0.1
MSC_EPOCHS = 50
MSC_WARMUP_EPOCHS = 5
MSC_BATCH_SIZE = 32
MSC_LR = 2e-5
MSC_WEIGHT_DECAY = 0.01
MSC_REPORTED_MACRO_F1 = 0.89   # Table A.II (deployment data, not synthetic data)

# ---------------------------------------------------------------------------
# MAML content recommender (Appendix A.1, [33])
# ---------------------------------------------------------------------------
MAML_K_STEPS = 5
MAML_INNER_LR = 0.01
MAML_OUTER_LR = 3e-4
MAML_BETA1 = 0.9
MAML_BETA2 = 0.999

# ---------------------------------------------------------------------------
# DRQN learning-path optimizer (Appendix A.2, [34]; PER [47])
# ---------------------------------------------------------------------------
DRQN_LSTM_HIDDEN = 256
DRQN_LSTM_LAYERS = 2
DRQN_MLP_LAYERS = 3
DRQN_GAMMA = 0.99
DRQN_REPLAY_BUFFER = 50000
DRQN_BATCH_SIZE = 64
DRQN_TARGET_UPDATE = 1000
DRQN_PER_ALPHA = 0.6
DRQN_PER_BETA_START = 0.4
DRQN_PER_BETA_END = 1.0
DRQN_REWARD_ALPHA = 0.5
DRQN_REWARD_BETA = 0.3
DRQN_REWARD_GAMMA_TIME = 0.2

# ---------------------------------------------------------------------------
# LinUCB intervention bandit (Appendix A.4, [31])
# ---------------------------------------------------------------------------
LINUCB_ALPHA = 0.5
LINUCB_N_ARMS = 7
LINUCB_DEBIAS_LAMBDA = 0.3

# ---------------------------------------------------------------------------
# Privacy (Appendix B)
# ---------------------------------------------------------------------------
K_ANONYMITY_K = 5
DP_EPSILON = 1.2

RANDOM_SEED = 42
