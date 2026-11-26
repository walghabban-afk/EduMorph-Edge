"""
model.py - EduMorph-Edge AI engine components (Appendix A of the paper).

  1. MSCTrans            - multimodal engagement classifier following the
                           MSC-Trans architecture [30]
  2. MAMLRecommender     - gradient-based meta-learning content recommender [33]
  3. DRQNPathOptimizer   - deep recurrent Q-network for POMDP path planning [34]
                           with prioritized experience replay [47]
  4. DebiasedLinUCB      - LinUCB contextual bandit [31] on a representation
                           trained with a gradient-reversal adversary
"""

import copy
import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from config import (
    FACE_AU_DIM, CLICK_DIM, TEXT_DIM, MSC_CNN_KERNELS, MSC_ATTENTION_HEADS,
    MSC_EMBED_DIM, MSC_N_CLASSES, MSC_DROPOUT, MSC_TEXT_MODEL, MSC_LR,
    MAML_K_STEPS, MAML_INNER_LR, MAML_OUTER_LR, MAML_BETA1, MAML_BETA2,
    DRQN_LSTM_HIDDEN, DRQN_LSTM_LAYERS, DRQN_GAMMA, DRQN_REPLAY_BUFFER,
    DRQN_BATCH_SIZE, DRQN_TARGET_UPDATE, DRQN_PER_ALPHA, DRQN_PER_BETA_START,
    DRQN_PER_BETA_END, DRQN_REWARD_ALPHA, DRQN_REWARD_BETA, DRQN_REWARD_GAMMA_TIME,
    LINUCB_ALPHA, LINUCB_N_ARMS, LINUCB_DEBIAS_LAMBDA,
)


# ===========================================================================
# 1. MSC-Trans engagement classifier (Appendix A.3)
# ===========================================================================

class ClickstreamEncoder(nn.Module):
    """1D-CNN over the clickstream sequence, kernel sizes {3, 5, 7}, max-pooling,
    output dimension 256."""

    def __init__(self, kernels=MSC_CNN_KERNELS, channels=96, out_dim=256):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(1, channels, k, padding=k // 2) for k in kernels])
        self.proj = nn.Linear(channels * len(kernels), out_dim)

    def forward(self, x):                      # x: (B, CLICK_DIM)
        x = x.unsqueeze(1)                     # (B, 1, L)
        feats = [F.relu(conv(x)).max(dim=-1).values for conv in self.convs]
        return self.proj(torch.cat(feats, dim=1))   # (B, 256)


class MSCTrans(nn.Module):
    """Three input streams fused by a cross-modal transformer.

    face : (B, 128) facial action-unit vector computed on the student device
           (ResNet-18 backbone + projection, run locally; Appendix B)
    click: (B, 64)  clickstream sequence -> 1D-CNN -> 256
    text : (B, 768) DistilBERT CLS embedding -> linear projection -> 256

    Each stream becomes one token of width 512; a 2-layer transformer encoder
    with 4 attention heads lets every modality attend to the others, and the
    pooled representation is classified into {Disengaged, Transitioning, Engaged}.
    """

    def __init__(self, embed_dim=MSC_EMBED_DIM, n_heads=MSC_ATTENTION_HEADS,
                 n_classes=MSC_N_CLASSES, dropout=MSC_DROPOUT):
        super().__init__()
        self.face_proj = nn.Linear(FACE_AU_DIM, embed_dim)
        self.click_enc = ClickstreamEncoder()
        self.click_proj = nn.Linear(256, embed_dim)
        self.text_down = nn.Linear(TEXT_DIM, 256)
        self.text_proj = nn.Linear(256, embed_dim)
        self.modality = nn.Parameter(torch.zeros(3, embed_dim))
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
                                           dim_feedforward=4 * embed_dim,
                                           dropout=dropout, batch_first=True)
        self.fusion = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(dropout),
                                  nn.Linear(embed_dim, n_classes))

    def forward(self, face, click, text):
        tokens = torch.stack([
            self.face_proj(face),
            self.click_proj(self.click_enc(click)),
            self.text_proj(self.text_down(text)),
        ], dim=1) + self.modality              # (B, 3, D)
        fused = self.fusion(tokens).mean(dim=1)  # (B, D)
        return self.head(fused)                # logits (B, 3)


def finetune_text_encoder(texts, labels, out_dir, epochs=3, lr=MSC_LR, batch_size=16,
                          model_name=MSC_TEXT_MODEL, device="cpu"):
    """Fine-tune DistilBERT on the three engagement labels (Appendix A.3) and
    save it to `out_dir`; pass `out_dir` as `model_name` to encode_text().
    Requires the `transformers` package and the pretrained weights."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    clf = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=MSC_N_CLASSES).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr)
    texts, labels = list(texts), torch.as_tensor(labels, dtype=torch.long)
    clf.train()
    for _ in range(epochs):
        perm = torch.randperm(len(texts))
        for i in range(0, len(texts), batch_size):
            idx = perm[i:i + batch_size]
            batch = tok([texts[j] for j in idx], padding=True, truncation=True, return_tensors="pt").to(device)
            loss = clf(**batch, labels=labels[idx].to(device)).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
    clf.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir


def encode_text(texts, device="cpu", model_name=MSC_TEXT_MODEL):
    """Return DistilBERT CLS embeddings (N, 768) for a list of strings, from the
    pretrained model or from a fine-tuned one saved by finetune_text_encoder().
    Requires the `transformers` package and the weights."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = AutoModel.from_pretrained(model_name).to(device).eval()
    with torch.no_grad():
        batch = tok(list(texts), padding=True, truncation=True, return_tensors="pt").to(device)
        return enc(**batch).last_hidden_state[:, 0, :]


# ===========================================================================
# 2. MAML content recommender (Appendix A.1)
# ===========================================================================

class BaseRecommender(nn.Module):
    """Base model f_theta mapping a student-context vector to item scores."""

    def __init__(self, input_dim=FACE_AU_DIM, n_items=100):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(),
                                 nn.Linear(128, n_items))

    def forward(self, x):
        return self.net(x)


class MAMLRecommender:
    """Second-order MAML: K = 5 inner SGD steps (alpha = 0.01) per student,
    outer Adam update (beta1 = 0.9, beta2 = 0.999, eta = 3e-4) - Eq. (A.1)."""

    def __init__(self, input_dim=FACE_AU_DIM, n_items=100, device="cpu"):
        self.device = device
        self.model = BaseRecommender(input_dim, n_items).to(device)
        self.outer_opt = torch.optim.Adam(self.model.parameters(), lr=MAML_OUTER_LR,
                                          betas=(MAML_BETA1, MAML_BETA2))

    def _adapted_params(self, sx, sy, create_graph):
        params = dict(self.model.named_parameters())
        for _ in range(MAML_K_STEPS):
            loss = F.cross_entropy(functional_call(self.model, params, (sx,)), sy)
            grads = torch.autograd.grad(loss, list(params.values()), create_graph=create_graph)
            params = {k: p - MAML_INNER_LR * g for (k, p), g in zip(params.items(), grads)}
        return params

    def outer_step(self, task_batch):
        """task_batch: iterable of (support_x, support_y, query_x, query_y)."""
        self.outer_opt.zero_grad()
        losses = []
        for sx, sy, qx, qy in task_batch:
            sx, sy, qx, qy = (t.to(self.device) for t in (sx, sy, qx, qy))
            params = self._adapted_params(sx, sy, create_graph=True)
            losses.append(F.cross_entropy(functional_call(self.model, params, (qx,)), qy))
        meta_loss = torch.stack(losses).mean()
        meta_loss.backward()
        self.outer_opt.step()
        return float(meta_loss.item())

    def personalize(self, sx, sy):
        """Return item scores function adapted to one student."""
        params = {k: v.detach() for k, v in self._adapted_params(
            sx.to(self.device), sy.to(self.device), create_graph=False).items()}
        return lambda x: functional_call(self.model, params, (x.to(self.device),))


# ===========================================================================
# 3. DRQN learning-path optimizer (Appendix A.2, Table A.I)
# ===========================================================================

KNOWLEDGE_STATE_DIM = 32   # IRT-based knowledge-state vector length


class DRQNNet(nn.Module):
    """2-layer LSTM encoder (hidden 256) and 3-layer MLP Q-head (512-256-|A|)."""

    def __init__(self, state_dim=KNOWLEDGE_STATE_DIM + MSC_N_CLASSES, n_actions=50):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, DRQN_LSTM_HIDDEN, DRQN_LSTM_LAYERS, batch_first=True)
        self.q = nn.Sequential(nn.Linear(DRQN_LSTM_HIDDEN, 512), nn.ReLU(),
                               nn.Linear(512, 256), nn.ReLU(),
                               nn.Linear(256, n_actions))

    def forward(self, x, hidden=None):        # x: (B, T, state_dim)
        out, hidden = self.lstm(x, hidden)
        return self.q(out[:, -1, :]), hidden


def make_state(knowledge_state, engagement_class):
    """Concatenate the knowledge-state vector with a one-hot engagement class."""
    one_hot = F.one_hot(torch.as_tensor(engagement_class), MSC_N_CLASSES).float()
    return torch.cat([torch.as_tensor(knowledge_state, dtype=torch.float32), one_hot], dim=-1)


class PrioritizedReplayBuffer:
    """Prioritized experience replay [47]: priority exponent 0.6, importance-
    sampling exponent annealed from 0.4 to 1.0."""

    def __init__(self, capacity=DRQN_REPLAY_BUFFER, alpha=DRQN_PER_ALPHA):
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def push(self, transition):
        max_p = max(self.priorities) if self.priorities else 1.0
        self.buffer.append(transition)
        self.priorities.append(max_p)

    def sample(self, batch_size, beta):
        pr = np.asarray(self.priorities, dtype=np.float64) ** self.alpha
        probs = pr / pr.sum()
        idx = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        w = (len(self.buffer) * probs[idx]) ** (-beta)
        w = w / w.max()
        return [self.buffer[i] for i in idx], idx, torch.tensor(w, dtype=torch.float32)

    def update_priorities(self, idx, td_errors):
        for i, e in zip(idx, td_errors):
            self.priorities[i] = float(abs(e)) + 1e-6

    def __len__(self):
        return len(self.buffer)


class DRQNPathOptimizer:
    """Reward r = alpha*d_accuracy + beta*d_engagement - gamma*time (Eq. A.3);
    Bellman target y = r + gamma * max_a' Q_target(s', a') (Eq. A.4)."""

    def __init__(self, state_dim=KNOWLEDGE_STATE_DIM + MSC_N_CLASSES, n_actions=50,
                 device="cpu", anneal_steps=50000):
        self.device = device
        self.n_actions = n_actions
        self.online = DRQNNet(state_dim, n_actions).to(device)
        self.target = copy.deepcopy(self.online).eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=1e-4)
        self.buffer = PrioritizedReplayBuffer()
        self.steps = 0
        self.beta = DRQN_PER_BETA_START
        self.beta_inc = (DRQN_PER_BETA_END - DRQN_PER_BETA_START) / anneal_steps

    @staticmethod
    def reward(d_acc, d_eng, time_cost):
        return DRQN_REWARD_ALPHA * d_acc + DRQN_REWARD_BETA * d_eng - DRQN_REWARD_GAMMA_TIME * time_cost

    def select_action(self, state_seq, epsilon=0.1):
        """state_seq: (T, state_dim) history of states."""
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            q, _ = self.online(state_seq.unsqueeze(0).to(self.device))
        return int(q.argmax(dim=1).item())

    def push(self, state_seq, action, reward, next_state_seq, done):
        self.buffer.push((state_seq, action, reward, next_state_seq, float(done)))

    def update(self):
        if len(self.buffer) < DRQN_BATCH_SIZE:
            return 0.0
        batch, idx, w = self.buffer.sample(DRQN_BATCH_SIZE, self.beta)
        self.beta = min(DRQN_PER_BETA_END, self.beta + self.beta_inc)
        s, a, r, ns, d = zip(*batch)
        s = nn.utils.rnn.pad_sequence(list(s), batch_first=True).to(self.device)
        ns = nn.utils.rnn.pad_sequence(list(ns), batch_first=True).to(self.device)
        a = torch.tensor(a, dtype=torch.long, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)
        q, _ = self.online(s)
        q = q.gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nq, _ = self.target(ns)
            y = r + DRQN_GAMMA * nq.max(dim=1).values * (1 - d)
        loss = (w.to(self.device) * F.smooth_l1_loss(q, y, reduction="none")).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.buffer.update_priorities(idx, (y - q).detach().cpu().numpy())
        self.steps += 1
        if self.steps % DRQN_TARGET_UPDATE == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())


# ===========================================================================
# 4. LinUCB intervention bandit with adversarial debiasing (Appendix A.4)
# ===========================================================================

class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class DebiasedEncoder(nn.Module):
    """Context encoder trained to predict intervention reward while a
    gradient-reversal adversary removes demographic information:
    L_total = L_reward - lambda * L_demographic  (Eq. A.6, lambda = 0.3)."""

    def __init__(self, in_dim, rep_dim=32, n_arms=LINUCB_N_ARMS, n_groups=2,
                 lam=LINUCB_DEBIAS_LAMBDA):
        super().__init__()
        self.lam = lam
        self.enc = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, rep_dim))
        self.reward_head = nn.Linear(rep_dim, n_arms)
        self.adv_head = nn.Sequential(nn.Linear(rep_dim, 32), nn.ReLU(), nn.Linear(32, n_groups))

    def forward(self, x):
        return self.enc(x)

    def loss(self, x, arm, reward, group):
        z = self.enc(x)
        pred = self.reward_head(z).gather(1, arm.unsqueeze(1)).squeeze(1)
        l_reward = F.mse_loss(pred, reward)
        l_demo = F.cross_entropy(self.adv_head(_GradReverse.apply(z, self.lam)), group)
        # Minimising l_reward + l_demo with the reversed gradient trains the
        # adversary to predict the group and the encoder to defeat it.
        return l_reward + l_demo, l_reward.detach(), l_demo.detach()


class LinUCBBandit:
    """LinUCB (alpha_UCB = 0.5) over K = 7 intervention arms (Eq. A.5)."""

    INTERVENTION_NAMES = ["hint", "difficulty_reduction", "peer_prompt", "teacher_alert",
                          "worked_example", "summary_card", "break_recommendation"]

    def __init__(self, context_dim, n_arms=LINUCB_N_ARMS, alpha=LINUCB_ALPHA):
        self.alpha = alpha
        self.A = [np.eye(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]

    def select_arm(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        scores = []
        for A, b in zip(self.A, self.b):
            A_inv = np.linalg.inv(A)
            theta = A_inv @ b
            scores.append(theta @ x + self.alpha * math.sqrt(x @ A_inv @ x))
        return int(np.argmax(scores))

    def update(self, arm, x, reward):
        x = np.asarray(x, dtype=np.float64).ravel()
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x

    def intervention_name(self, arm):
        return self.INTERVENTION_NAMES[arm]


class DebiasedLinUCB:
    """LinUCB operating on the debiased representation of the student context."""

    def __init__(self, in_dim, rep_dim=32, device="cpu"):
        self.device = device
        self.encoder = DebiasedEncoder(in_dim, rep_dim).to(device)
        self.opt = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
        self.bandit = LinUCBBandit(rep_dim)

    def represent(self, x):
        with torch.no_grad():
            return self.encoder(torch.as_tensor(x, dtype=torch.float32, device=self.device)).cpu().numpy()

    def select_arm(self, x):
        return self.bandit.select_arm(self.represent(x))

    def update(self, arm, x, reward):
        self.bandit.update(arm, self.represent(x), reward)

    def train_encoder(self, x, arm, reward, group, epochs=20):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        arm = torch.as_tensor(arm, dtype=torch.long, device=self.device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        group = torch.as_tensor(group, dtype=torch.long, device=self.device)
        for _ in range(epochs):
            loss, lr_, ld_ = self.encoder.loss(x, arm, reward, group)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return float(lr_), float(ld_)

    def intervention_name(self, arm):
        return self.bandit.intervention_name(arm)
