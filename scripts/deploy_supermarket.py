"""
Hierarchical VLM Planner + VLA/Nav Executor — closed-loop deployment
============================================================================
Scope (single-developer portfolio project):
  1. VLM plans: high-level command + TEXT map -> subtask sequence.
  2. Subtasks run SEQUENTIALLY (receding-horizon VLA, eval_libero-style).
  3. Navigation is decomposed into MULTIPLE straight-line segments.

Completion check (the closed loop):
  A cheap sensor TRIGGER (arrival / heading / gripper) flags "maybe done".
  - nav/turn: the trigger IS the done signal (arrival/heading are unambiguous).
  - VLA subtasks: the trigger fires ONE VLM judge call (head before/after
    frames) returning a verdict that drives the scheduler state machine:
      IN_PROGRESS -> change nothing, keep running (cooldown before re-asking)
      SUCCESS     -> advance to the next subtask
      RETRY       -> re-run the same subtask (world intact, e.g. grasp slipped)
      REPLAN      -> regenerate the remaining plan from the current state
      ABORT       -> stop (unrecoverable; also the max_steps timeout)
  This mirrors G0's System-2 (2 Hz replanning) but event-triggered instead of
  fixed-rate, to keep VLM API cost down. Set --no-vlm-judge for the cheaper
  sensor-only path (trigger == success, no recovery).

Usage:
  python scripts/deploy_supermarket.py \
    --command "restock Nivea from warehouse A1 to shelf B3 layer 2" \
    --map-file configs/maps/store_layout1.json \
    --vlm-provider qwen --vlm-api-key $DASHSCOPE_KEY \
    --vla-ckpt /path/to/fetch_lora_ckpt --mode simulate
"""
import os
import json
import time
import enum
import logging
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Deploy")
# ============================================================
# 1. Task taxonomy — two families: VLA manipulation, navigation
# ============================================================
class TaskFamily(enum.Enum):
    VLA = "vla"      # pick_to_basket / restock_basket_to_shelf / pick_from_floor
    NAV = "nav"      # navigate_to / turn_to


TASK_FAMILY_MAP = {
    "pick_to_basket":           TaskFamily.VLA,
    "restock_basket_to_shelf":  TaskFamily.VLA,
    "pick_from_floor":          TaskFamily.VLA,
    "navigate_to":              TaskFamily.NAV,
    "turn_to":                  TaskFamily.NAV,
}
VALID_TYPES = list(TASK_FAMILY_MAP.keys())


class JudgeVerdict(enum.Enum):
    """VLM Judge only answers: "is the subtask done?"
    Complex recovery decisions (retry / replan / abort) are handled by the
    Scheduler based on context (retry count, replan budget, etc.), not by VLM.

      IN_PROGRESS  subtask not yet at terminal state -> keep running
      SUCCESS      terminal state reached -> advance to next subtask
      FAIL         subtask failed (grasp slipped, item dropped, etc.)
                   -> Scheduler decides: retry, replan, or abort
    """
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAIL = "fail"


@dataclass
class Subtask:
    type: str                 # one of VALID_TYPES
    instruction: str          # natural-language string fed to the VLA as `task`
    target: str = ""          # "x,y" for navigate_to; degrees for turn_to; shelf id otherwise
    max_steps: int = 400
    status: str = "pending"   # pending | running | completed | failed
    retries: int = 0          # RETRY verdicts spent on this subtask (vs Scheduler.max_retries)

    @property
    def family(self) -> TaskFamily:
        return TASK_FAMILY_MAP[self.type]

    def to_dict(self) -> Dict:
        return {"type": self.type, "instruction": self.instruction,
                "target": self.target, "status": self.status}
# --- section: storemap ---


# ============================================================
# 2. Store map — TEXT representation (ASCII grid + legend + waypoints)
# ============================================================
class StoreMap:
    """Renders the store as TEXT for the VLM (coordinates parse far more
    reliably from an ASCII grid + waypoint list than from pixels).

    waypoints are the corridor intersection points the VLM chains together
    into a multi-segment route: A -> wp1 -> wp2 -> shelf is three navigate_to
    subtasks, each one straight segment between adjacent waypoints."""

    def __init__(self, m: Dict):
        self.grid_ascii: str = m["grid_ascii"]
        self.legend: Dict[str, str] = m["legend"]
        self.waypoints: Dict[str, List[float]] = m.get("waypoints", {})
        self.shelf_approach: Dict[str, Dict] = m.get("shelf_approach", {})

    @classmethod
    def from_file(cls, path: str) -> "StoreMap":
        with open(path) as f:
            return cls(json.load(f))

    def render(self, robot_xy: List[float], robot_yaw_deg: float) -> str:
        lines = [
            "=== STORE MAP (top-down, each cell = 1.0 m) ===",
            "X -> east (right), Y -> north (up)",
            self.grid_ascii,
            "",
            "=== LEGEND ===",
        ]
        for code, desc in self.legend.items():
            lines.append(f"  {code}: {desc}")
        lines.append("")
        lines.append("=== CORRIDOR WAYPOINTS (chain these into straight segments) ===")
        for name, xy in self.waypoints.items():
            lines.append(f"  {name}: ({xy[0]}, {xy[1]})")
        lines.append("")
        lines.append("=== SHELF APPROACH POINTS (stand here to manipulate) ===")
        for name, info in self.shelf_approach.items():
            lines.append(f"  {name}: pos=({info['pos'][0]}, {info['pos'][1]}), face={info['face_deg']}deg")
        lines.append("")
        lines.append(f"=== ROBOT NOW: at ({robot_xy[0]:.1f}, {robot_xy[1]:.1f}) "
                     f"facing {robot_yaw_deg:.0f}deg ===")
        return "\n".join(lines)
# --- section: prompts ---


# ============================================================
# 3. Prompt — single planning call, emphasises multi-segment nav
# ============================================================
SYSTEM_PROMPT = """You are the task planner for a retail-store mobile manipulation robot (Fetch).
Decompose a high-level human command into a sequence of atomic subtasks.

=== ATOMIC SUBTASK TYPES (use these EXACT type strings) ===
Manipulation (executed by a VLA policy):
  pick_to_basket           Pick an item from a shelf into the robot's basket.
  restock_basket_to_shelf  Take an item from the basket, place it on a shelf layer.
  pick_from_floor          Pick a fallen item from the floor onto a shelf.
Navigation (executed AUTONOMOUSLY by the nav module — global A* path + local obstacle avoidance):
  navigate_to              Drive to a target SHELF. target = the shelf id (e.g. "C-C0",
                           "CS-1"). The nav module computes the collision-free route by
                           itself — you do NOT plan waypoints or read pixels.
  turn_to                  Rotate in place to an absolute heading. target = degrees
                           (0=east, 90=north, 180=west, 270=south).

=== RULES ===
1. type MUST be one of: pick_to_basket, restock_basket_to_shelf, pick_from_floor,
   navigate_to, turn_to.
2. You do NOT plan routes. For navigation, emit a SINGLE navigate_to per destination
   with target = the shelf id. The nav module handles global path + obstacle avoidance.
3. Use the SHELF INVENTORY to find which shelf holds the target item, and which shelf
   is the destination. Emit navigate_to with that shelf's id as the target.
4. Before any manipulation the robot must first navigate_to that shelf. After arrival
   the nav module AUTOMATICALLY faces the shelf — do NOT emit turn_to for this.
5. Output ONLY a JSON array. No prose, no markdown fences.

Each subtask object has keys: "type", "instruction", "target"."""

USER_TMPL = """{map_block}

=== HUMAN COMMAND ===
{command}

Decompose the command into the full subtask sequence from the robot's current
position. For navigation emit ONE navigate_to per destination with target = the
shelf id (the nav module plans the route and avoids obstacles itself — do NOT emit
waypoints or pixel coordinates). Output ONLY the JSON array."""

# Judge prompt: called when a cheap sensor trigger fires mid-subtask. The VLM
# looks at the head camera BEFORE/AFTER frames and decides the verdict that
# drives the scheduler state machine.
JUDGE_SYSTEM_PROMPT = """You verify a retail robot's subtask from camera frames.
You receive 3 images in this order:
  1. Head camera BEFORE the action (reference for what changed)
  2. Head camera AFTER the action
  3. Left wrist camera AFTER the action (shows gripper state)

For manipulation tasks (pick/place), the wrist camera is CRITICAL — it shows
whether the gripper actually holds or released the item. The head camera alone
may miss fine-grained gripper state.

verdict (use these EXACT strings):
  in_progress  The subtask's terminal goal is NOT yet reached.
               The robot is still working correctly. (Most common — default to this.)
  success      The subtask's terminal goal IS achieved.
               For pick: wrist cam shows item secured in gripper / basket.
               For place: wrist cam shows item released on shelf surface.
  fail         The subtask failed (grasp slipped, item dropped, wrong action, etc.)

Output ONLY JSON: {"verdict": "...", "reason": "<short>"}"""

JUDGE_USER_TMPL = """High-level goal: {coarse_task}
Current subtask: {subtask}
Images: head_before, head_after, left_wrist_after.
Has this subtask reached its terminal goal?
Output ONLY the JSON verdict."""
# --- section: planner ---


# ============================================================
# 4. VLM planner — ONE call, returns the whole subtask sequence
# ============================================================
class VLMPlanner:
    def __init__(self, provider: str = "qwen", api_key: str = "", model: str = "",
                 base_url: str = ""):
        self.provider = provider.lower()
        self.api_key = api_key or os.environ.get("VLM_API_KEY", "")
        self.base_url = base_url or os.environ.get("VLM_BASE_URL", "")
        self.model = model or {"qwen": "qwen-vl-max", "gemini": "gemini-2.0-flash",
                               "openai": "gpt-4o",
                               "openai_compatible": "qwen-vl-max",
                               "anthropic": "claude-sonnet-4-5"}.get(self.provider, "qwen-vl-max")
        self._init_client()

    def _init_client(self):
        self.client = None
        if self.provider in ("gemini", "openai"):
            import openai
            base_url = {"gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
                        "openai": None}[self.provider]
            self.client = openai.OpenAI(api_key=self.api_key, base_url=base_url)
        elif self.provider == "openai_compatible":
            import openai
            if not self.base_url:
                raise ValueError("--vlm-base-url is required for openai_compatible provider")
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        elif self.provider == "anthropic":
            import anthropic
            # Native Anthropic Messages API. For PackyCode use
            # --vlm-base-url https://api.packycode.com  (NO /v1 suffix; the SDK
            # appends /v1/messages itself). Omit base_url to hit Anthropic direct.
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self.client = anthropic.Anthropic(**kwargs)
        elif self.provider == "qwen":
            try:
                from dashscope import MultiModalConversation
                self.client = MultiModalConversation
            except ImportError:
                logger.warning("dashscope not installed; qwen calls will fail")

    def plan(self, command: str, map_block: str,
             image: Optional[np.ndarray] = None) -> List[Subtask]:
        user = USER_TMPL.format(map_block=map_block, command=command)
        raw = self._call(user, image)
        return self._parse(raw)

    def judge(self, coarse_task: str, subtask: str,
              head_before: Optional[np.ndarray] = None,
              head_after: Optional[np.ndarray] = None,
              left_wrist_after: Optional[np.ndarray] = None) -> "JudgeVerdict":
        """Verify a subtask from 3 images: head_before, head_after, left_wrist_after.
        VLM only answers "is the subtask done?" — returns IN_PROGRESS / SUCCESS / FAIL.
        Recovery decisions (retry / replan / abort) are handled by the Scheduler.
        On any error defaults to IN_PROGRESS (conservative: keep running)."""
        user = JUDGE_USER_TMPL.format(coarse_task=coarse_task, subtask=subtask)
        imgs = [im for im in (head_before, head_after, left_wrist_after)
                if im is not None]
        try:
            raw = self._call(user, imgs, system=JUDGE_SYSTEM_PROMPT)
            return self._parse_verdict(raw)
        except Exception as e:                       # network / API / parse
            logger.warning(f"judge call failed ({e}); defaulting to IN_PROGRESS")
            return JudgeVerdict.IN_PROGRESS

    @staticmethod
    def _parse_verdict(raw: str) -> "JudgeVerdict":
        import re
        m = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
        try:
            v = json.loads(m.group() if m else raw.strip()).get("verdict", "").lower()
        except (json.JSONDecodeError, AttributeError):
            logger.warning(f"judge reply not JSON: {raw[:120]}; -> IN_PROGRESS")
            return JudgeVerdict.IN_PROGRESS
        try:
            return JudgeVerdict(v)
        except ValueError:
            logger.warning(f"unknown verdict '{v}'; -> IN_PROGRESS")
            return JudgeVerdict.IN_PROGRESS

    # ---- internal ----
    def _call(self, prompt: str, image=None, system: str = SYSTEM_PROMPT) -> str:
        # `image` may be a single ndarray, a list of ndarrays, or None.
        images = image if isinstance(image, list) else ([] if image is None else [image])
        if self.provider == "qwen" and self.client is not None:
            return self._call_qwen(prompt, images, system)
        if self.provider in ("gemini", "openai", "openai_compatible") and self.client is not None:
            return self._call_openai(prompt, images, system)
        if self.provider == "anthropic" and self.client is not None:
            return self._call_anthropic(prompt, images, system)
        raise RuntimeError(f"VLM provider '{self.provider}' has no usable client")
# --- section: planner-api ---

    @staticmethod
    def _encode(image: np.ndarray) -> str:
        import cv2, base64
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8) if image.max() <= 1.0 \
                    else image.astype(np.uint8)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", bgr)
        return base64.b64encode(buf).decode("utf-8")

    def _call_qwen(self, prompt: str, images: List[np.ndarray], system: str) -> str:
        content = [{"text": prompt}]
        for im in images:                                   # 0, 1, or 2 frames
            content.insert(0, {"image": f"data:image/jpeg;base64,{self._encode(im)}"})
        resp = self.client.call(
            model=self.model, api_key=self.api_key,
            messages=[{"role": "system", "content": [{"text": system}]},
                      {"role": "user", "content": content}],
        )
        return resp.output.choices[0].message.content[0]["text"]

    def _call_openai(self, prompt: str, images: List[np.ndarray], system: str) -> str:
        content = [{"type": "text", "text": prompt}]
        for im in images:
            content.insert(0, {"type": "image_url",
                               "image_url": {"url": f"data:image/jpeg;base64,{self._encode(im)}"}})
        resp = self.client.chat.completions.create(
            model=self.model, temperature=0.1, max_tokens=1024,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": content}],
        )
        return resp.choices[0].message.content

    def _call_anthropic(self, prompt: str, images: List[np.ndarray], system: str) -> str:
        # Native Anthropic Messages API: system is a TOP-LEVEL param (not a
        # message), and images use source.base64 blocks rather than image_url.
        content = []
        for im in images:
            content.append({"type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg",
                                       "data": self._encode(im)}})
        content.append({"type": "text", "text": prompt})
        resp = self.client.messages.create(
            model=self.model, max_tokens=1024, temperature=0.1,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        # 取第一个文本块: 模型可能返回 ToolUseBlock 等非文本块在前, 不能直接 [0].text
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                return block.text
            if hasattr(block, "text"):
                return block.text
        return ""

    def _parse(self, raw: str) -> List[Subtask]:
        import re
        m = re.search(r"\[.*\]", raw.strip(), re.DOTALL)
        text = m.group() if m else raw.strip()
        try:
            items = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"VLM plan not valid JSON: {raw[:200]}")
            raise ValueError(f"plan parse error: {e}")
        plan: List[Subtask] = []
        for it in items:
            t = str(it.get("type", "")).lower()
            if t not in VALID_TYPES:
                logger.warning(f"dropping unknown subtask type: {t}")
                continue
            fam = TASK_FAMILY_MAP[t]
            plan.append(Subtask(
                type=t,
                instruction=it.get("instruction", f"{t} {it.get('target','')}"),
                target=str(it.get("target", "")),
                max_steps=300 if fam == TaskFamily.VLA else 600,
            ))
        return plan
# --- section: executors ---


# ============================================================
# 5. Executors — VLA (Fetch 15-DoF), Nav (point-goal), Turn (yaw PD)
# ============================================================
class VLAExecutor:
    """Wraps the LoRA-finetuned G0Plus policy for Fetch (15-DoF action).
    Real deployment would wrap the ROS2 VLA node; here it runs the policy
    directly. Falls back to a mock action when no checkpoint is given."""

    def __init__(self, ckpt_path: str = "",
                 cfg_path: str = "configs/task/robobenchmart/fetch_lora_finetune.yaml",
                 device: str = "cuda"):
        self.ckpt_path = ckpt_path
        self.cfg_path = cfg_path
        self.device = device
        self.policy = None
        self.processor = None
        # High-level goal passed as `coarse_task`. G0's executor was trained on
        # "[High]: {coarse_task}, [Low]: {task}" (see base_processor.py), so the
        # scheduler sets this once to the full human command and each subtask's
        # instruction becomes the low-level `task`.
        self.coarse_task = ""
        self._load()

    def set_coarse_task(self, coarse_task: str):
        self.coarse_task = coarse_task or ""

    def _load(self):
        if not self.ckpt_path:
            logger.warning("VLA: no checkpoint -> mock executor")
            return
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        from galaxea_fm.utils.config_resolvers import register_default_resolvers
        from galaxea_fm.utils.load_pretrained_resumed import load_checkpoint_for_eval
        register_default_resolvers()
        cfg = OmegaConf.load(self.cfg_path)
        OmegaConf.resolve(cfg)
        policy = instantiate(cfg.model.model_arch)
        policy, stats = load_checkpoint_for_eval(self.ckpt_path, policy, device="cpu")
        self.policy = policy.to(self.device).eval()
        self.processor = instantiate(cfg.data.processor)
        self.processor.set_normalizer_from_stats(stats)
        self.processor.eval()
        logger.info(f"VLA: loaded Fetch policy from {self.ckpt_path}")

    def act(self, subtask: Subtask, obs: Dict) -> np.ndarray:
        if self.policy is None:
            return np.random.randn(16, 15).astype(np.float32) * 0.01
        import torch
        from galaxea_fm.utils.pytorch_utils import dict_apply
        sample = {
            "images": {"head_rgb": obs["head_rgb"],
                       "left_wrist_rgb": obs["left_wrist_rgb"],
                       "right_wrist_rgb": obs["right_wrist_rgb"]},
            "state": obs["state"],
            "task": subtask.instruction,
            "coarse_task": self.coarse_task,   # high-level goal -> "[High]: ..." channel
            "state_is_pad": torch.tensor([False]),
            "image_is_pad": torch.tensor([False]),
            "action_is_pad": torch.tensor([False] * 32),
            "idx": torch.tensor(0),
        }
        sample = self.processor.preprocess(sample)
        batch = dict_apply(sample, lambda x: x.unsqueeze(0).to(self.device)
                           if isinstance(x, torch.Tensor) else x)
        with torch.no_grad():
            batch = self.policy.predict_action(batch)
        batch = dict_apply(batch, lambda x: x.cpu() if hasattr(x, "cpu") else x)
        batch = self.processor.postprocess(batch)
        action = batch["action"]
        if isinstance(action, dict):
            action = action["default"]
        return np.asarray(action).reshape(-1, 15)
# --- section: nav-turn ---


class NavExecutor:
    """Point-goal navigation for ONE straight segment.

    Supports two backends:
      - 'navdp':  NavDP diffusion policy (needs RGB + depth, stateful)
      - 'motionplanner': FetchMotionPlanningSapienSolver from RoboBenchMart
      - 'mock':   no-op (for simulate mode)

    The local planner drives to the goal (x, y), turning and avoiding
    obstacles within the segment, then reports arrival.
    """

    def __init__(self, backend: str = "mock", device: str = "cuda",
                 navdp_ckpt: str = "", env=None):
        self.backend = backend
        self.device = device
        self.controller = None
        self._goal_world = None
        self._arrived = False
        self._step_count = 0
        self._max_nav_steps = 600   # safety timeout
        # 终点朝向: 导航模块到达 approach 位置后, 自动转到货架 face_deg(不靠 VLM)
        self.shelf_approach: Dict[str, Dict] = {}
        self._face_deg = None          # 目标朝向(度); None=不需对齐
        self._reached_pos = False      # 位置已到(尚未对齐朝向)
        self._aligned = False          # 朝向已对齐
        self._yaw_kp, self._yaw_kd, self._yaw_prev = 1.5, 0.3, 0.0

        if backend == "navdp" and navdp_ckpt:
            sys.path.append("/home/lh/VLA/RoboBenchMart-main")
            from dsynth.navigation.navdp_controller import NavDPController
            self.controller = NavDPController(
                model_path=navdp_ckpt, device=device,
            )
            logger.info(f"Nav: NavDP controller loaded from {navdp_ckpt}")
        elif backend == "motionplanner" and env is not None:
            sys.path.append("/home/lh/VLA/RoboBenchMart-main")
            from dsynth.planning.motionplanner import FetchMotionPlanningSapienSolver
            unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
            self.controller = FetchMotionPlanningSapienSolver(unwrapped)
            logger.info("Nav: MotionPlanner controller loaded")
        else:
            logger.info("Nav: mock controller (no-op)")

    def reset(self):
        """Reset navigation state for a new segment."""
        self._goal_world = None
        self._arrived = False
        self._step_count = 0
        if self.controller is not None and hasattr(self.controller, 'reset'):
            self.controller.reset()

    def set_goal(self, obs_provider, target: str) -> np.ndarray:
        # target 可为货架 id(如 "C-C0")或 "x,y"。货架 id -> 查 approach 位置 + face_deg,
        # 到达位置后由导航模块自动转到 face_deg(终点朝向不靠 VLM)。
        self._face_deg = None
        info = self.shelf_approach.get(target)
        if info is not None:
            x, y = float(info["pos"][0]), float(info["pos"][1])
            self._face_deg = float(info.get("face_deg")) if info.get("face_deg") is not None else None
        else:
            x, y = [float(v) for v in target.split(",")]
        self._goal_world = np.array([x, y, 0.0])
        self._arrived = False
        self._reached_pos = False
        self._aligned = False
        self._yaw_prev = 0.0
        self._step_count = 0
        # Support both bound methods and callable objects
        env = getattr(obs_provider, "__self__", None) or (
            obs_provider if hasattr(obs_provider, "set_nav_goal") else None
        )
        if env is not None and hasattr(env, "set_nav_goal"):
            env.set_nav_goal(np.array([x, y]))
        if self.controller is not None and hasattr(self.controller, 'reset'):
            self.controller.reset()
        return np.array([x, y])

    def act(self, subtask: Subtask, obs: Dict) -> np.ndarray:
        """Compute navigation action for one step."""
        self._step_count += 1

        # 阶段2: 位置已到, 由导航模块把朝向对齐到货架 face_deg(不靠 VLM 发 turn_to)
        if self._reached_pos and self._face_deg is not None and not self._aligned:
            target = np.radians(self._face_deg)
            cur = float(obs.get("robot_yaw", 0.0))
            err = (target - cur + np.pi) % (2 * np.pi) - np.pi
            if abs(err) < np.radians(5.0):       # 朝向到位
                self._aligned = True
                self._arrived = True
                return np.zeros((16, 15), dtype=np.float32)
            omega = float(np.clip(self._yaw_kp * err + self._yaw_kd * (err - self._yaw_prev),
                                  -1.0, 1.0))
            self._yaw_prev = err
            a = np.zeros((16, 15), dtype=np.float32)
            a[:, 2] = omega                       # base_yaw channel
            return a

        # 阶段1: 驱动到 approach 位置
        if self._check_arrival(obs):
            self._reached_pos = True
            if self._face_deg is None:            # 无需对齐朝向 -> 直接完成
                self._arrived = True
                return np.zeros((16, 15), dtype=np.float32)
            self._yaw_prev = 0.0                  # 进入对齐阶段, 清 PD 状态
            return np.zeros((16, 15), dtype=np.float32)

        # Safety timeout
        if self._step_count > self._max_nav_steps:
            logger.warning(f"Nav: max steps ({self._max_nav_steps}) reached, forcing arrival")
            self._arrived = True
            return np.zeros((16, 15), dtype=np.float32)

        if self.backend == "navdp" and self.controller is not None:
            return self._act_navdp(obs)
        elif self.backend == "motionplanner" and self.controller is not None:
            return self._act_motionplanner(obs)
        else:
            return np.zeros((16, 15), dtype=np.float32)

    def _check_arrival(self, obs: Dict) -> bool:
        if obs.get("nav_arrived") is True:
            return True
        if self._goal_world is not None:
            cur = np.asarray(obs.get("robot_position", [0.0, 0.0, 0.0]))[:2]
            goal = self._goal_world[:2]
            return float(np.linalg.norm(cur - goal)) < 0.3
        return False

    def _act_navdp(self, obs: Dict) -> np.ndarray:
        """NavDP: plan trajectory then pure-pursuit control."""
        import torch as _torch

        # Get robot pose matrix
        robot_pose = obs.get("robot_pose_matrix")
        if robot_pose is None:
            return np.zeros((16, 15), dtype=np.float32)

        # Compute goal in robot frame
        goal_world = np.array([self._goal_world[0], self._goal_world[1], 0.0])
        goal_robot = self.controller.compute_goal_in_robot_frame(goal_world, robot_pose)

        # Get depth image
        depth = obs.get("depth")
        head_rgb = obs.get("head_rgb_raw")
        if depth is None or head_rgb is None:
            return np.zeros((16, 15), dtype=np.float32)

        # Plan (every step for NavDP since it's stateful)
        try:
            self.controller.plan(head_rgb, depth, goal_robot, robot_pose)
        except Exception as e:
            logger.warning(f"NavDP plan failed: {e}")
            return np.zeros((16, 15), dtype=np.float32)

        # Check if NavDP says stop
        if self.controller.is_stopped:
            logger.info("Nav: NavDP reports no safe path (stopped)")
            self._arrived = True
            return np.zeros((16, 15), dtype=np.float32)

        # Pure-pursuit velocity
        linear_vel, angular_vel = self.controller.compute_base_velocity(robot_pose)

        # Convert to Fetch 15-DoF action
        action = np.zeros((16, 15), dtype=np.float32)
        action[:, 0] = linear_vel    # base_x velocity
        action[:, 2] = angular_vel   # base_yaw velocity
        return action

    def _act_motionplanner(self, obs: Dict) -> np.ndarray:
        """MotionPlanner: use drive_base for the whole segment at once."""
        # MotionPlanner drives the base directly via env.step(),
        # so we signal arrival after one call (it's blocking).
        # This is handled differently — see apply_action integration.
        if self._step_count == 1 and self._goal_world is not None:
            try:
                target_pos = self._goal_world.copy()
                self.controller.drive_base(target_pos=target_pos[:3])
            except Exception as e:
                logger.warning(f"MotionPlanner drive_base failed: {e}")
            self._arrived = True
        return np.zeros((16, 15), dtype=np.float32)

    @property
    def arrived(self) -> bool:
        return self._arrived


class TurnExecutor:
    """Rotate in place to an absolute heading (degrees). Used only after arrival,
    to face a shelf for manipulation. Simple PD on base yaw."""

    def __init__(self, kp: float = 1.5, kd: float = 0.3):
        self.kp, self.kd, self.prev = kp, kd, 0.0

    def reset(self):
        """Clear PD state. Called at the start of each turn_to subtask so the
        derivative term does not carry error from a previous turn."""
        self.prev = 0.0

    def act(self, subtask: Subtask, obs: Dict) -> np.ndarray:
        target = np.radians(float(subtask.target))
        cur = obs.get("robot_yaw", 0.0)
        err = self._wrap(target - cur)
        omega = float(np.clip(self.kp * err + self.kd * (err - self.prev), -1.0, 1.0))
        self.prev = err
        a = np.zeros((16, 15), dtype=np.float32)
        a[:, 2] = omega   # base_yaw channel
        return a

    @staticmethod
    def _wrap(a: float) -> float:
        while a > np.pi:  a -= 2 * np.pi
        while a < -np.pi: a += 2 * np.pi
        return a
# --- section: scheduler ---


# ============================================================
# 6. Sequential scheduler (open-loop)
#    plan ONCE -> run each subtask to its sensor "done" signal -> next
# ============================================================
class Scheduler:
    def __init__(self, vlm: VLMPlanner, store_map: StoreMap,
                 vla: VLAExecutor, nav: NavExecutor, turn: TurnExecutor,
                 replan_steps: int = 5, use_vlm_judge: bool = True,
                 judge_cooldown: int = 20, max_retries: int = 2, max_replans: int = 2,
                 judge_fn=None):
        self.vlm = vlm
        self.map = store_map
        self.vla = vla
        self.nav = nav
        self.turn = turn
        # VLA receding horizon: infer once, execute this many steps of the
        # action chunk, then re-infer. Matches eval_libero.py replan_steps=5.
        self.replan_steps = replan_steps
        # Closed-loop completion checking. When True, a cheap sensor TRIGGER on a
        # VLA subtask fires ONE VLM judge call (head before/after frames) that
        # returns a JudgeVerdict driving the run() state machine. When False, the
        # trigger itself counts as success (the old open-loop, sensor-only path).
        self.use_vlm_judge = use_vlm_judge
        self.judge_cooldown = judge_cooldown   # steps to wait after IN_PROGRESS before re-asking
        self.max_retries = max_retries         # per-subtask RETRY budget
        self.max_replans = max_replans         # whole-task REPLAN budget
        # Verdict source. Defaults to the VLM; tests / offline sim inject a
        # callable (coarse_task, subtask, before, after) -> JudgeVerdict.
        self.judge_fn = judge_fn or (vlm.judge if vlm is not None else None)

    def run(self, command: str, obs_provider) -> Dict:
        pose = self._pose(obs_provider)
        map_block = self.map.render(pose["xy"], pose["yaw_deg"])

        # 把货架 approach 数据(pos + face_deg)交给导航模块: navigate_to 的 target 是
        # 货架 id, 由导航模块查坐标、走 A*、到位后自动转到 face_deg(终点朝向不靠 VLM)。
        self.nav.shelf_approach = self.map.shelf_approach

        # The full human command is the high-level goal for the VLA executor.
        # G0 was trained on "[High]: {coarse_task}, [Low]: {task}", so feeding
        # the command here lets each subtask inherit the overall intent.
        self.vla.set_coarse_task(command)

        logger.info("=" * 64)
        logger.info(f"COMMAND: {command}")
        plan = self.vlm.plan(command, map_block)
        if not plan:
            return {"command": command, "status": "failed", "reason": "empty plan"}
        self._log_plan(plan)

        # State machine over the subtask sequence. VLM Judge returns only
        # IN_PROGRESS / SUCCESS / FAIL. The Scheduler decides recovery:
        #   FAIL + retries left  -> RETRY (re-run same subtask)
        #   FAIL + retries exhausted + replans left -> REPLAN
        #   FAIL + all budgets exhausted -> ABORT
        completed = 0
        replans = 0
        i = 0
        while i < len(plan):
            subtask = plan[i]
            logger.info(f"\n--- [{i+1}/{len(plan)}] {subtask.type}: {subtask.instruction} ---")
            verdict = self._execute(subtask, obs_provider)

            if verdict is JudgeVerdict.SUCCESS:
                subtask.status = "completed"
                completed += 1

                # 导航不再让 VLM 介入: 子任务规划好后, 全局路径(A*)+ 局部避障由
                # 导航模块自动处理。VLM 只负责高层规划(理解意图、拆子任务、选目标),
                # 不在到达每个路点后重规划路线。
                i += 1
                continue

            if verdict is JudgeVerdict.FAIL:
                # Scheduler decides recovery based on context, not VLM
                if subtask.retries < self.max_retries:
                    subtask.retries += 1
                    logger.warning(f"  FAIL -> RETRY {subtask.retries}/{self.max_retries}: {subtask.instruction}")
                    continue                              # same i: re-run this subtask
                if replans < self.max_replans:
                    replans += 1
                    logger.warning(f"  FAIL (retries exhausted) -> REPLAN {replans}/{self.max_replans}")
                    pose = self._pose(obs_provider)
                    fresh = self.vlm.plan(command, self.map.render(pose["xy"], pose["yaw_deg"]))
                    if fresh:
                        plan = plan[:i] + fresh           # keep done prefix, swap the rest
                        self._log_plan(plan)
                        continue                          # same i: start of the new tail
                logger.error(f"  FAIL (all budgets exhausted) -> ABORT: {subtask.instruction}")
                subtask.status = "failed"
                return self._summary(command, "failed", plan, completed)

            # Should not reach here, but defensive
            logger.error(f"  unexpected verdict: {verdict}")
            subtask.status = "failed"
            return self._summary(command, "failed", plan, completed)

        return self._summary(command, "success", plan, completed)

    def _log_plan(self, plan: List[Subtask]):
        for i, s in enumerate(plan):
            logger.info(f"  [{i+1}] {s.family.value:3s} | {s.type:22s} | {s.instruction}")

    # ---- run ONE subtask; return a JudgeVerdict that run() acts on ----
    def _execute(self, subtask: Subtask, obs_provider) -> "JudgeVerdict":
        if subtask.type == "navigate_to":
            self.nav.set_goal(obs_provider, subtask.target)
            executor, is_vla = self.nav, False
        elif subtask.type == "turn_to":
            self.turn.reset()                 # fix: PD state must not leak across turns
            executor, is_vla = self.turn, False
        else:
            executor, is_vla = self.vla, True

        # Support both bound methods (SimEnv.get_obs) and callable objects (RealEnvObsProvider)
        env = getattr(obs_provider, "__self__", None) or (
            obs_provider if hasattr(obs_provider, "apply_action") else None
        )
        if env is not None and hasattr(env, "set_subtask"):
            env.set_subtask(subtask)

        # VLA subtasks get the VLM judge; nav/turn use the sensor signal directly
        # (arrival / heading are unambiguous, no vision needed).
        judging = is_vla and self.use_vlm_judge and self.judge_fn is not None
        step, last_judge = 0, 0
        obs = obs_provider()
        head_before = obs.get("head_rgb_raw") if judging else None

        # Receding horizon: VLA infers once, executes `replan_steps` of its chunk,
        # then re-infers (matches eval_libero.py). Nav/turn re-infer every step.
        while step < subtask.max_steps:
            chunk = executor.act(subtask, obs)            # (T, 15)
            n_exec = min(self.replan_steps if is_vla else 1, len(chunk))
            for i in range(n_exec):
                if env is not None and hasattr(env, "apply_action"):
                    env.apply_action(chunk[i])
                step += 1
                obs = obs_provider()                      # POST-action obs
                if self._trigger(subtask, obs):
                    if not judging:                       # sensor-only path
                        logger.info(f"  done in {step} steps (sensor)")
                        return JudgeVerdict.SUCCESS
                    if step - last_judge >= self.judge_cooldown:
                        last_judge = step                 # cooldown gate: avoid VLM spam
                        v = self.judge_fn(
                            self.vla.coarse_task, subtask.instruction,
                            head_before, obs.get("head_rgb_raw"),
                            obs.get("left_wrist_rgb_raw"),
                        )
                        logger.info(f"  trigger@{step} -> judge={v.value}")
                        if v is JudgeVerdict.IN_PROGRESS:
                            head_before = obs.get("head_rgb_raw")
                        else:
                            return v                      # SUCCESS or FAIL
                if step >= subtask.max_steps:
                    break
        logger.warning(f"  hit max_steps ({subtask.max_steps}) -> FAIL")
        return JudgeVerdict.FAIL
# --- section: scheduler-done ---

    # ---- cheap sensor TRIGGER (no VLM). For nav/turn it IS the done signal;
    #      for VLA it only flags "maybe terminal" -> the VLM judge decides. ----
    def _trigger(self, subtask: Subtask, obs: Dict) -> bool:
        t = subtask.type
        if t == "navigate_to":
            if obs.get("nav_arrived") is True:
                return True
            cur = np.asarray(obs.get("robot_position", [0.0, 0.0]))[:2]
            goal = np.asarray(obs.get("nav_goal", [1e9, 1e9]))[:2]
            return float(np.linalg.norm(cur - goal)) < 0.3
        if t == "turn_to":
            cur = obs.get("robot_yaw", 0.0)
            tgt = np.radians(float(subtask.target))
            return abs(TurnExecutor._wrap(tgt - cur)) < np.radians(5)
        if t in ("pick_to_basket", "pick_from_floor"):
            # Key checkpoint: gripper must be closed AND arm must have moved
            # toward the shelf (not just closed in rest position).
            gripper_closed = bool(obs.get("gripper_closed", False))
            if not gripper_closed:
                return False
            # Additional check: arm is not in rest position
            # (joint values significantly different from zero/home)
            qpos = obs.get("state", {}).get("default", None)
            if qpos is not None:
                arm_joints = np.asarray(qpos)[3:10]  # arm joints (indices 3-9)
                arm_moved = float(np.linalg.norm(arm_joints)) > 0.3
                return arm_moved
            return gripper_closed  # fallback if no state
        if t == "restock_basket_to_shelf":
            # Key checkpoint: gripper must be open AND arm must have reached
            # toward the shelf (not just opened at rest).
            gripper_open = bool(obs.get("gripper_open", False))
            if not gripper_open:
                return False
            qpos = obs.get("state", {}).get("default", None)
            if qpos is not None:
                arm_joints = np.asarray(qpos)[3:10]
                arm_moved = float(np.linalg.norm(arm_joints)) > 0.3
                return arm_moved
            return gripper_open
        return False

    # ---- helpers ----
    def _pose(self, obs_provider) -> Dict:
        obs = obs_provider()
        xy = np.asarray(obs.get("robot_position", [0.0, 0.0]))[:2]
        return {"xy": [float(xy[0]), float(xy[1])],
                "yaw_deg": float(np.degrees(obs.get("robot_yaw", 0.0)))}

    def _summary(self, command: str, status: str,
                 plan: List[Subtask], completed: int) -> Dict:
        logger.info("=" * 64)
        logger.info(f"RESULT: {status}  ({completed}/{len(plan)} subtasks completed)")
        return {"command": command, "status": status,
                "completed": completed, "total": len(plan),
                "plan": [s.to_dict() for s in plan]}
# --- section: realenv ---


# ============================================================
# 7a. Real RoboBenchMart environment obs provider
#     Integrates StoreMapProvider for VLM planning with top-down map.
# ============================================================
class RealEnvObsProvider:
    """Bridges RoboBenchMart simulation environment to the Scheduler.

    - Uses StoreMapProvider to auto-extract top-down map, shelf inventory,
      and coordinate table from the simulation environment.
    - Only the robot position is updated in real-time; the static map
      (shelf labels, waypoints, coordinates) is built once at init.
    - Provides obs dict compatible with Scheduler/VLAExecutor/NavExecutor.
    """

    def __init__(self, env, camera_height: float = 8.0, image_size: int = 1024):
        """
        Args:
            env: ManiSkill gym environment from RoboBenchMart.
            camera_height: Height for top-down camera (meters).
            image_size: Resolution of top-down image.
        """
        import sys
        sys.path.append("/home/lh/VLA/RoboBenchMart-main")
        from dsynth.navigation.map_utils import StoreMapProvider

        self.env = env.unwrapped if hasattr(env, 'unwrapped') else env
        self.map_provider = StoreMapProvider(
            self.env, camera_height=camera_height, image_size=image_size
        )
        self.map_provider.initialize()
        self._nav_goal = np.array([0.0, 0.0])

        # Discover sensor names from environment (avoid hardcoding)
        self._sensor_names = self._discover_sensors()

    # ---- Scheduler hooks ----
    def set_subtask(self, subtask: Subtask):
        pass  # could be used for logging

    def set_nav_goal(self, goal: np.ndarray):
        self._nav_goal = goal.copy()

    def _discover_sensors(self) -> Dict[str, str]:
        """Discover sensor names from the environment to avoid hardcoding.

        Returns dict mapping logical role -> actual sensor name:
          head_camera, left_wrist_camera, right_wrist_camera
        """
        names = {}
        try:
            sensor_data = self.env.scene.get_sensor_data()
            all_keys = list(sensor_data.keys()) if isinstance(sensor_data, dict) else []
            logger.info(f"Available sensors: {all_keys}")

            # Head camera: look for base_camera, render_camera, or any camera
            # that is NOT a hand/wrist camera
            for k in all_keys:
                k_lower = k.lower()
                if 'hand' in k_lower or 'wrist' in k_lower or 'gripper' in k_lower:
                    continue
                if 'camera' in k_lower or 'rgb' in k_lower:
                    names.setdefault('head_camera', k)
                    break

            # Hand/wrist cameras
            for k in all_keys:
                k_lower = k.lower()
                if 'hand' in k_lower or 'wrist' in k_lower or 'gripper' in k_lower:
                    if 'left' not in names.values() and 'right' not in names.values():
                        names.setdefault('left_wrist_camera', k)
                    else:
                        names.setdefault('right_wrist_camera', k)

            # Fallbacks for common ManiSkill sensor names
            if 'head_camera' not in names:
                for candidate in ['left_base_camera_link', 'base_camera',
                                  'render_camera', 'fetch_head']:
                    if candidate in all_keys:
                        names['head_camera'] = candidate
                        break
            if 'left_wrist_camera' not in names:
                for candidate in ['fetch_hand', 'hand_camera', 'left_hand_camera',
                                  'left_wrist_camera']:
                    if candidate in all_keys:
                        names['left_wrist_camera'] = candidate
                        break
            if 'right_wrist_camera' not in names:
                for candidate in ['right_hand_camera', 'right_wrist_camera',
                                  'fetch_right_hand']:
                    if candidate in all_keys:
                        names['right_wrist_camera'] = candidate
                        break

        except Exception as e:
            logger.warning(f"Sensor discovery failed: {e}")

        logger.info(f"Mapped sensors: {names}")
        return names

    def apply_action(self, action: np.ndarray):
        """Apply action to the simulation environment."""
        action = np.asarray(action).flatten()
        if len(action) >= 15:
            full_action = action[:15]
        elif len(action) >= 2:
            full_action = np.zeros(15, dtype=np.float32)
            full_action[0] = action[0]  # base_x
            full_action[2] = action[1] if len(action) > 1 else 0.0  # base_yaw
        else:
            return
        try:
            obs, reward, terminated, truncated, info = self.env.step(full_action)
            if terminated or truncated:
                logger.warning("Episode ended (terminated/truncated), resetting")
                self.env.reset()
        except Exception as e:
            logger.warning(f"env.step failed: {e}")

    # ---- Main obs interface ----
    def __call__(self) -> Dict:
        """Return obs dict compatible with Scheduler."""
        return self.get_obs()

    def get_obs(self) -> Dict:
        """Get current observation from the simulation environment."""
        import torch

        # Sensor data from ManiSkill
        sensor_data = self.env.scene.get_sensor_data()

        # RGB images
        head_rgb_raw = None
        left_wrist_rgb_raw = None
        right_wrist_rgb_raw = None
        depth_raw = None

        # Head camera
        head_key = self._sensor_names.get('head_camera')
        if head_key and head_key in sensor_data:
            try:
                cam = sensor_data[head_key]
                head_rgb_raw = cam['rgb'][0].cpu().numpy()  # (H, W, 3)
                if 'depth' in cam:
                    depth_raw = cam['depth'][0, :, :, 0].cpu().numpy()  # (H, W)
            except Exception as e:
                logger.warning(f"Failed to read head camera ({head_key}): {e}")

        # Left wrist camera
        left_key = self._sensor_names.get('left_wrist_camera')
        if left_key and left_key in sensor_data:
            try:
                left_wrist_rgb_raw = sensor_data[left_key]['rgb'][0].cpu().numpy()
            except Exception as e:
                logger.warning(f"Failed to read left wrist camera ({left_key}): {e}")

        # Right wrist camera
        right_key = self._sensor_names.get('right_wrist_camera')
        if right_key and right_key in sensor_data:
            try:
                right_wrist_rgb_raw = sensor_data[right_key]['rgb'][0].cpu().numpy()
            except Exception as e:
                logger.warning(f"Failed to read right wrist camera ({right_key}): {e}")

        # Fallback: create dummy images if sensors unavailable
        if head_rgb_raw is None:
            head_rgb_raw = np.zeros((360, 640, 3), dtype=np.uint8)
        if left_wrist_rgb_raw is None:
            left_wrist_rgb_raw = np.zeros((128, 128, 3), dtype=np.uint8)
        # Fetch has only one gripper — right wrist is a copy of left wrist
        right_wrist_rgb_raw = left_wrist_rgb_raw.copy()

        # Robot state
        robot_pos = self.env.agent.base_link.pose.p[0].cpu().numpy()
        robot_mat = self.env.agent.base_link.pose.to_transformation_matrix()[0].cpu().numpy()
        robot_yaw = float(np.arctan2(robot_mat[1, 0], robot_mat[0, 0]))

        # Joint state
        qpos = self.env.agent.get_qpos()[0].cpu().numpy()

        # Gripper state (last joint of Fetch)
        gripper_q = qpos[-1] if len(qpos) > 0 else 0.0
        gripper_closed = gripper_q < 0.01
        gripper_open = gripper_q > 0.04

        # Resize images for VLA (224x224)
        try:
            import cv2
            head_rgb = cv2.resize(head_rgb_raw, (224, 224)).transpose(2, 0, 1).astype(np.float32) / 255.0
            left_wrist = cv2.resize(left_wrist_rgb_raw, (224, 224)).transpose(2, 0, 1).astype(np.float32) / 255.0
            # Fetch has one gripper — right wrist is a copy of left wrist
            right_wrist = left_wrist.copy()
        except Exception:
            head_rgb = np.zeros((3, 224, 224), np.float32)
            left_wrist = np.zeros((3, 224, 224), np.float32)
            right_wrist = np.zeros((3, 224, 224), np.float32)

        return {
            "head_rgb": head_rgb,
            "left_wrist_rgb": left_wrist,
            "right_wrist_rgb": right_wrist,
            "head_rgb_raw": head_rgb_raw,
            "left_wrist_rgb_raw": left_wrist_rgb_raw,
            "right_wrist_rgb_raw": right_wrist_rgb_raw,
            "depth": depth_raw,
            "state": {"default": qpos.astype(np.float32)},
            "robot_position": robot_pos,
            "robot_yaw": robot_yaw,
            "robot_pose_matrix": robot_mat,
            "nav_goal": self._nav_goal,
            "nav_arrived": bool(np.linalg.norm(robot_pos[:2] - self._nav_goal) < 0.3),
            "gripper_closed": gripper_closed,
            "gripper_open": gripper_open,
        }

    # ---- VLM planning helpers ----
    def get_map_info(self):
        """Get MapInfo from StoreMapProvider (with updated robot position)."""
        return self.map_provider.get_map_info()

    def get_vlm_prompt_block(self) -> str:
        """Get text block for VLM prompt (inventory + coordinates + robot state)."""
        return self.map_provider.get_vlm_prompt_block()

    def get_vlm_image(self) -> np.ndarray:
        """Get top-down image with shelf labels and robot position."""
        return self.map_provider.get_vlm_image()


# ============================================================
# 7b. Simulated environment (test the loop without a real robot)
# ============================================================
class SimEnv:
    """Minimal stand-in: drives the active subtask to its done-signal after a
    fixed number of steps, exposing only the obs keys _is_done reads. Lets you
    run plan -> sequential execute end-to-end with mocks."""

    def __init__(self, steps_to_done: int = 15):
        self.steps_to_done = steps_to_done
        self.step = 0
        self.subtask: Optional[Subtask] = None
        self.pos = np.array([2.0, 3.0])
        self.yaw = 0.0
        self.nav_goal = np.array([2.0, 3.0])
        self.gripper_closed = False

    # scheduler hooks
    def set_subtask(self, subtask: Subtask):
        self.subtask = subtask
        self.step = 0

    def set_nav_goal(self, goal: np.ndarray):
        self.nav_goal = goal

    def apply_action(self, action: np.ndarray):
        pass

    def get_obs(self) -> Dict:
        self.step += 1
        t = self.subtask.type if self.subtask else ""
        progressed = self.step >= self.steps_to_done
        if t == "navigate_to" and progressed:
            self.pos = self.nav_goal.copy()
        if t == "turn_to" and progressed and self.subtask:
            self.yaw = np.radians(float(self.subtask.target))
        if t in ("pick_to_basket", "pick_from_floor") and progressed:
            self.gripper_closed = True
        if t == "restock_basket_to_shelf" and progressed:
            self.gripper_closed = False
        return {
            "head_rgb": np.zeros((3, 224, 224), np.float32),
            "left_wrist_rgb": np.zeros((3, 224, 224), np.float32),
            # Fetch has one gripper — right wrist is a copy of left wrist
            "right_wrist_rgb": np.zeros((3, 224, 224), np.float32),
            "head_rgb_raw": np.zeros((224, 224, 3), np.uint8),
            "left_wrist_rgb_raw": np.zeros((128, 128, 3), np.uint8),
            "right_wrist_rgb_raw": np.zeros((128, 128, 3), np.uint8),
            "state": {"default": np.zeros((15,), np.float32)},
            "robot_position": self.pos.copy(),
            "robot_yaw": self.yaw,
            "nav_goal": self.nav_goal.copy(),
            "nav_arrived": bool(np.linalg.norm(self.pos - self.nav_goal) < 0.3),
            "gripper_closed": self.gripper_closed,
            "gripper_open": not self.gripper_closed,
        }
# --- section: main ---


# ============================================================
# 8. Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Hierarchical VLM+VLA supermarket deployment")
    p.add_argument("--command", required=True, help="high-level human command")
    p.add_argument("--map-file", default="", help="store map JSON (not needed for real_env mode)")
    p.add_argument("--vlm-provider", default="qwen", choices=["qwen", "gemini", "openai", "openai_compatible"])
    p.add_argument("--vlm-api-key", default="")
    p.add_argument("--vlm-model", default="")
    p.add_argument("--vlm-base-url", default="",
                   help="OpenAI-compatible base URL (for openai_compatible provider, e.g. https://api.packy.com/v1)")
    p.add_argument("--vla-ckpt", default="", help="Fetch LoRA checkpoint (empty -> mock)")
    p.add_argument("--no-vlm-judge", action="store_true",
                   help="disable closed-loop VLM completion check (sensor-only, no recovery)")
    p.add_argument("--mode", default="simulate", choices=["simulate", "real_env"],
                   help="simulate=mock, real_env=RoboBenchMart simulation")
    p.add_argument("--env-name", default="DarkstoreContinuousBaseEnv",
                   help="RoboBenchMart environment name (for real_env mode)")
    p.add_argument("--robot-uids", default="ds_fetch_basket",
                   help="Robot type: ds_fetch_basket, panda_wristcam, etc.")
    p.add_argument("--nav-backend", default="mock",
                   choices=["mock", "navdp", "motionplanner"],
                   help="Navigation backend")
    p.add_argument("--navdp-ckpt", default="", help="NavDP checkpoint path")
    p.add_argument("--scene-dir", default="",
                   help="RoboBenchMart scene config directory (contains input_config.yaml)")
    p.add_argument("--output", default="")
    return p.parse_args()


def main():
    args = parse_args()
    vlm = VLMPlanner(args.vlm_provider, args.vlm_api_key, args.vlm_model, args.vlm_base_url)
    vla = VLAExecutor(ckpt_path=args.vla_ckpt)
    turn = TurnExecutor()

    if args.mode == "simulate":
        # Mock mode: uses SimEnv + static map JSON
        store_map = StoreMap.from_file(args.map_file) if args.map_file else StoreMap({
            "grid_ascii": "WWWWWWWWWW\nW  W  W  W\nW  W  W  W\n   W  W   \nW  W  W  W\nWWWWWWWWWW",
            "legend": {"W": "wall/shelf", " ": "corridor"},
            "waypoints": {"wp_0": [2.0, 1.0], "wp_1": [5.0, 1.0], "wp_2": [5.0, 5.0]},
            "shelf_approach": {"zone_0_shelf_0": {"pos": [2.0, 1.5], "face_deg": 0}},
        })
        sim = SimEnv(steps_to_done=15)
        nav = NavExecutor(backend="mock")
        obs_provider = sim.get_obs
        scheduler = Scheduler(vlm, store_map, vla, nav, turn,
                              use_vlm_judge=not args.no_vlm_judge)

    elif args.mode == "real_env":
        # Real RoboBenchMart simulation environment
        import gymnasium as gym
        import sys
        sys.path.append("/home/lh/VLA/RoboBenchMart-main")

        env = gym.make(
            args.env_name,
            num_envs=1,
            obs_mode="rgbd",
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            robot_uids=args.robot_uids,
            config_dir_path=args.scene_dir,
            enable_shadow=False,
            parallel_in_single_scene=False,
        )
        obs, info = env.reset(options={"reconfigure": True})

        # Create RealEnvObsProvider (auto-extracts map from environment)
        real_env = RealEnvObsProvider(env)

        # Build StoreMap from RealEnvObsProvider's map data
        map_info = real_env.get_map_info()
        store_map = StoreMap({
            "grid_ascii": f"Scene {map_info.scene_size[0]:.1f}x{map_info.scene_size[1]:.1f}m",
            "legend": {"S": "shelf", " ": "corridor"},
            "waypoints": {k: v for k, v in map_info.waypoints.items()},
            "shelf_approach": {
                name: {"pos": [s.approach_pos[0], s.approach_pos[1]],
                        "face_deg": s.approach_yaw_deg}
                for name, s in map_info.shelves.items()
            },
        })

        # Create NavExecutor with chosen backend
        nav = NavExecutor(
            backend=args.nav_backend,
            navdp_ckpt=args.navdp_ckpt,
            env=env,
        )

        obs_provider = real_env
        scheduler = Scheduler(vlm, store_map, vla, nav, turn,
                              use_vlm_judge=not args.no_vlm_judge)

        # Override VLM planner to use StoreMapProvider's dynamic map
        # (top-down image + inventory text, updated with robot position)
        _orig_plan = vlm.plan
        def plan_with_map(command, map_block, image=None):
            map_info = real_env.get_map_info()
            return _orig_plan(command, map_info.full_prompt_block, map_info.topdown_image)
        vlm.plan = plan_with_map

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    result = scheduler.run(args.command, obs_provider)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"saved -> {args.output}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
