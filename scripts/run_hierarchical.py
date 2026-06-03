"""
VLM Offline Planner + VLA Executor for GalaxeaVLA
===================================================
Architecture:
  VLM API (Qwen/Gemini) → Task Decomposition → Subtask Scheduler → VLA/NavDP Execution

Atomic Tasks (VLA handles):
  - pick: pick up an object
  - place: place an object at a target location
  - open: open a container/drawer
  - close: close a container/drawer

Navigation Tasks (NavDP handles):
  - navigate: move to a target location with obstacle avoidance

Composite Tasks (VLM decomposes):
  - restock: warehouse_pick → navigate_shelf → place_shelf
  - tidy_up: pick_item → navigate_target → place_item

Usage:
  python scripts/run_hierarchical.py \
    --command "restock the chips to shelf B3" \
    --vlm_provider qwen \
    --vlm_api_key YOUR_KEY \
    --vla_ckpt /path/to/checkpoint \
    --mode simulate
"""

import json
import time
import enum
import logging
import argparse
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("HierarchicalPlanner")


# ============================================================
# 1. Task Definitions
# ============================================================

class TaskType(enum.Enum):
    PICK = "pick"
    PLACE = "place"
    OPEN = "open"
    CLOSE = "close"
    NAVIGATE_TO = "navigate_to"
    TURN_TO = "turn_to"


class ExecutorType(enum.Enum):
    VLA = "vla"
    NAVDP = "navdp"
    MPC = "mpc"


TASK_EXECUTOR_MAP = {
    TaskType.PICK: ExecutorType.VLA,
    TaskType.PLACE: ExecutorType.VLA,
    TaskType.OPEN: ExecutorType.VLA,
    TaskType.CLOSE: ExecutorType.VLA,
    TaskType.NAVIGATE_TO: ExecutorType.NAVDP,
    TaskType.TURN_TO: ExecutorType.MPC,
}

COMPOSITE_TASKS = {
    "restock": {
        "description": "Move goods from warehouse to retail shelf",
        "template": [
            {"type": "navigate_to", "target": "warehouse_{item_location}"},
            {"type": "turn_to", "target": "warehouse_shelf_facing"},
            {"type": "pick", "target": "{item}"},
            {"type": "navigate_to", "target": "shelf_{shelf_id}"},
            {"type": "turn_to", "target": "shelf_{shelf_id}_facing"},
            {"type": "place", "target": "shelf_{shelf_id}"},
        ],
    },
    "tidy_up": {
        "description": "Pick up scattered items and place them in designated locations",
        "template": [
            {"type": "navigate_to", "target": "{item_location}"},
            {"type": "turn_to", "target": "{item_location}_facing"},
            {"type": "pick", "target": "{item}"},
            {"type": "navigate_to", "target": "{target_location}"},
            {"type": "turn_to", "target": "{target_location}_facing"},
            {"type": "place", "target": "{target_location}"},
        ],
    },
}

VALID_TASK_TYPES = [t.value for t in TaskType]



@dataclass
class Subtask:
    task_type: TaskType
    instruction: str
    executor: ExecutorType
    target: str = ""
    max_steps: int = 200
    timeout_seconds: float = 60.0
    status: str = "pending"

    def to_dict(self) -> Dict:
        return {
            "task_type": self.task_type.value,
            "instruction": self.instruction,
            "executor": self.executor.value,
            "target": self.target,
            "max_steps": self.max_steps,
            "status": self.status,
        }


# ============================================================
# 2. VLM Planner (API-based, no fine-tuning)
# ============================================================

VLM_SYSTEM_PROMPT = """You are a robot task planner for a mobile manipulation robot (R1Lite).
Your job is to decompose a high-level human command into a sequence of atomic subtasks.

Available atomic actions:
- navigate_to <location>: Move the robot base to a target location using visual navigation (NavDP). The robot will navigate along a safe path but may not face the target. Example: "navigate to shelf A3"
- turn_to <direction>: Turn the robot body to face a specific direction or object. This is needed BEFORE manipulation tasks. Example: "turn to face the shelf"
- pick <object>: Pick up a specified object. Example: "pick the red cup"
- place <location>: Place the held object at a location. Example: "place on the table"
- open <container>: Open a container. Example: "open the drawer"
- close <container>: Close a container. Example: "close the drawer"

Rules:
1. Each subtask must be EXACTLY one of the 6 atomic actions above.
2. The instruction for each subtask must be a clear, specific English sentence.
3. Navigation (navigate_to) moves the robot to a location but does NOT guarantee facing it.
4. After navigate_to, you MUST add a turn_to subtask BEFORE any manipulation (pick/place/open/close).
5. Typical sequence at each location: navigate_to → turn_to → manipulation.
6. For restock: navigate_to warehouse → turn_to shelf → pick → navigate_to retail → turn_to shelf → place.
7. Output ONLY valid JSON, no other text.

Output format (JSON array):
[
  {
    "type": "navigate_to",
    "instruction": "navigate to the warehouse shelf A",
    "target": "warehouse_shelf_A"
  },
  {
    "type": "turn_to",
    "instruction": "turn to face the warehouse shelf A",
    "target": "warehouse_shelf_A_facing"
  },
  {
    "type": "pick",
    "instruction": "pick up the chips box",
    "target": "chips_box"
  },
  {
    "type": "navigate_to",
    "instruction": "navigate to the retail shelf B3",
    "target": "retail_shelf_B3"
  },
  {
    "type": "turn_to",
    "instruction": "turn to face the retail shelf B3",
    "target": "retail_shelf_B3_facing"
  },
  {
    "type": "place",
    "instruction": "place on the retail shelf B3",
    "target": "retail_shelf_B3"
  }
]"""

VLM_USER_PROMPT_TEMPLATE = """Human command: {command}
Current scene description: {scene_description}
Available locations: {locations}

Decompose this command into atomic subtasks. Output ONLY the JSON array."""


class VLMPlanner:
    def __init__(self, provider: str = "qwen", api_key: str = "", model: str = ""):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = model or self._default_model()
        self.client = None
        self._init_client()

    def _default_model(self) -> str:
        if self.provider == "qwen":
            return "qwen-vl-max"
        elif self.provider == "gemini":
            return "gemini-2.0-flash"
        elif self.provider == "openai":
            return "gpt-4o"
        return "qwen-vl-max"

    def _init_client(self):
        if self.provider == "qwen":
            try:
                from dashscope import MultiModalConversation
                self.client = MultiModalConversation
            except ImportError:
                logger.warning("dashscope not installed, using HTTP API fallback")
                self.client = None
        elif self.provider in ("gemini", "openai"):
            try:
                import openai
                base_url = {
                    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
                    "openai": None,
                }[self.provider]
                self.client = openai.OpenAI(api_key=self.api_key, base_url=base_url)
            except ImportError:
                raise ImportError("openai package required: pip install openai")

    def plan(
        self,
        command: str,
        image: Optional[np.ndarray] = None,
        scene_description: str = "retail store",
        locations: str = "warehouse_shelf_A, retail_shelf_B3, checkout_counter",
    ) -> List[Subtask]:
        user_prompt = VLM_USER_PROMPT_TEMPLATE.format(
            command=command,
            scene_description=scene_description,
            locations=locations,
        )

        raw_response = self._call_api(user_prompt, image)
        subtasks = self._parse_response(raw_response)
        return subtasks

    def _call_api(self, prompt: str, image: Optional[np.ndarray] = None) -> str:
        if self.provider == "qwen" and self.client is not None:
            return self._call_qwen(prompt, image)
        elif self.provider in ("gemini", "openai") and self.client is not None:
            return self._call_openai_compatible(prompt, image)
        else:
            return self._call_http(prompt, image)

    def _call_qwen(self, prompt: str, image: Optional[np.ndarray] = None) -> str:
        messages = [{"role": "system", "content": [{"text": VLM_SYSTEM_PROMPT}]}]
        user_content = [{"text": prompt}]
        if image is not None:
            import base64
            _, buffer = cv2.imencode(".jpg", image)
            b64 = base64.b64encode(buffer).decode("utf-8")
            user_content.insert(0, {"image": f"data:image/jpeg;base64,{b64}"})
        messages.append({"role": "user", "content": user_content})
        response = self.client.call(
            model=self.model,
            messages=messages,
            api_key=self.api_key,
        )
        return response.output.choices[0].message.content[0]["text"]

    def _call_openai_compatible(self, prompt: str, image: Optional[np.ndarray] = None) -> str:
        messages = [{"role": "system", "content": VLM_SYSTEM_PROMPT}]
        user_content = [{"type": "text", "text": prompt}]
        if image is not None:
            import base64
            _, buffer = cv2.imencode(".jpg", image)
            b64 = base64.b64encode(buffer).decode("utf-8")
            user_content.insert(0, {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        messages.append({"role": "user", "content": user_content})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
        )
        return response.choices[0].message.content

    def _call_http(self, prompt: str, image: Optional[np.ndarray] = None) -> str:
        import requests
        if self.provider == "qwen":
            url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "input": {
                    "messages": [
                        {"role": "system", "content": [{"text": VLM_SYSTEM_PROMPT}]},
                        {"role": "user", "content": [{"text": prompt}]},
                    ]
                },
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            return resp.json()["output"]["choices"][0]["message"]["content"][0]["text"]
        raise ValueError(f"Unsupported provider: {self.provider}")

    def _parse_response(self, raw: str) -> List[Subtask]:
        json_str = raw.strip()
        json_match = re.search(r"\[.*\]", json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group()
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"VLM output is not valid JSON: {raw[:200]}")
            raise ValueError(f"VLM output parse error: {e}")

        subtasks = []
        for item in parsed:
            task_type_str = item.get("type", "").lower()
            if task_type_str not in VALID_TASK_TYPES:
                logger.warning(f"Unknown task type: {task_type_str}, skipping")
                continue
            task_type = TaskType(task_type_str)
            executor = TASK_EXECUTOR_MAP[task_type]
            instruction = item.get("instruction", f"{task_type_str} {item.get('target', '')}")
            subtasks.append(Subtask(
                task_type=task_type,
                instruction=instruction,
                executor=executor,
                target=item.get("target", ""),
                max_steps=200 if executor == ExecutorType.VLA else 500,
                timeout_seconds=60.0 if executor == ExecutorType.VLA else 120.0,
            ))
        return subtasks


# ============================================================
# 3. Fallback Rule-Based Planner (when VLM is unavailable)
# ============================================================

class RuleBasedPlanner:
    PATTERNS = {
        "restock": [
            {"type": "navigate_to", "instruction": "navigate to the warehouse pickup point", "target": "warehouse"},
            {"type": "turn_to", "instruction": "turn to face the warehouse shelf", "target": "warehouse_facing"},
            {"type": "pick", "instruction": "pick up the {item}", "target": "{item}"},
            {"type": "navigate_to", "instruction": "navigate to the {shelf}", "target": "{shelf}"},
            {"type": "turn_to", "instruction": "turn to face the {shelf}", "target": "{shelf}_facing"},
            {"type": "place", "instruction": "place on the {shelf}", "target": "{shelf}"},
        ],
        "tidy_up": [
            {"type": "navigate_to", "instruction": "navigate to the {item_location}", "target": "{item_location}"},
            {"type": "turn_to", "instruction": "turn to face the {item_location}", "target": "{item_location}_facing"},
            {"type": "pick", "instruction": "pick up the {item}", "target": "{item}"},
            {"type": "navigate_to", "instruction": "navigate to the {target_location}", "target": "{target_location}"},
            {"type": "turn_to", "instruction": "turn to face the {target_location}", "target": "{target_location}_facing"},
            {"type": "place", "instruction": "place at the {target_location}", "target": "{target_location}"},
        ],
        "pick": [
            {"type": "pick", "instruction": "pick up the {item}", "target": "{item}"},
        ],
        "place": [
            {"type": "place", "instruction": "place at the {location}", "target": "{location}"},
        ],
    }

    def plan(
        self,
        command: str,
        **kwargs,
    ) -> List[Subtask]:
        cmd_lower = command.lower()
        matched_key = None
        for key in self.PATTERNS:
            if key in cmd_lower:
                matched_key = key
                break

        if matched_key is None:
            if any(w in cmd_lower for w in ["pick", "grab", "take"]):
                matched_key = "pick"
            elif any(w in cmd_lower for w in ["place", "put", "drop"]):
                matched_key = "place"
            else:
                matched_key = "restock"

        template = self.PATTERNS[matched_key]
        item = self._extract_item(cmd_lower)
        subtasks = []
        for step in template:
            task_type = TaskType(step["type"])
            executor = TASK_EXECUTOR_MAP[task_type]
            instruction = step["instruction"].format(
                item=item, shelf="retail shelf B3",
                item_location="item location", target_location="target location",
            )
            subtasks.append(Subtask(
                task_type=task_type,
                instruction=instruction,
                executor=executor,
                target=step.get("target", "").format(
                    item=item, shelf="retail_shelf_B3",
                    item_location="item_location", target_location="target_location",
                ),
            ))
        return subtasks

    def _extract_item(self, command: str) -> str:
        for prefix in ["pick up the ", "grab the ", "take the ", "restock the ", "restock "]:
            if prefix in command:
                idx = command.index(prefix) + len(prefix)
                rest = command[idx:].strip()
                if " to " in rest:
                    return rest.split(" to ")[0].strip()
                return rest
        return "object"


# ============================================================
# 4. Executor Interface
# ============================================================

class BaseExecutor(ABC):
    @abstractmethod
    def execute(self, subtask: Subtask, obs: Dict) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def is_done(self, subtask: Subtask, obs: Dict) -> bool:
        raise NotImplementedError


class VLAExecutor(BaseExecutor):
    """
    VLA executor: loads GalaxeaZero model, runs inference per step.
    In real deployment, this wraps the ROS2 VLA node.
    In simulation, this uses the model directly.
    """

    def __init__(self, ckpt_path: str = "", device: str = "cuda"):
        self.ckpt_path = ckpt_path
        self.device = device
        self.policy = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        if not self.ckpt_path:
            logger.warning("No VLA checkpoint path provided, using mock executor")
            self.policy = None
            self.processor = None
            return

        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        from galaxea_fm.utils.load_pretrained_resumed import load_checkpoint_for_eval

        cfg_path = "configs/task/real/g0plus_r1lite_lora_finetune.yaml"
        cfg = OmegaConf.load(cfg_path)
        OmegaConf.resolve(cfg)

        policy = instantiate(cfg.model.model_arch)
        policy, dataset_stats = load_checkpoint_for_eval(self.ckpt_path, policy, device="cpu")
        self.policy = policy.cuda().eval()

        self.processor = instantiate(cfg.data.processor)
        self.processor.set_normalizer_from_stats(dataset_stats)
        self.processor.eval()

        logger.info(f"VLA model loaded from {self.ckpt_path}")

    def execute(self, subtask: Subtask, obs: Dict) -> Dict:
        if self.policy is None:
            return self._mock_execute(subtask, obs)

        import torch
        from galaxea_fm.utils.pytorch_utils import dict_apply

        sample = {
            "images": {
                "head_rgb": obs["head_rgb"],
                "left_wrist_rgb": obs["left_wrist_rgb"],
                "right_wrist_rgb": obs["right_wrist_rgb"],
            },
            "state": obs["state"],
            "task": subtask.instruction,
            "coarse_task": "",
            "state_is_pad": torch.tensor([False]),
            "image_is_pad": torch.tensor([False]),
            "action_is_pad": torch.tensor([False] * 32),
            "idx": torch.tensor(0),
        }

        sample = self.processor.preprocess(sample)
        batch = dict_apply(
            sample,
            lambda x: x.unsqueeze(0).to(self.device) if isinstance(x, torch.Tensor) else x,
        )

        with torch.no_grad():
            batch = self.policy.predict_action(batch)

        batch = dict_apply(batch, lambda x: x.cpu() if isinstance(x, torch.Tensor) else x)
        batch = self.processor.postprocess(batch)

        action = dict_apply(batch["action"], lambda x: x.numpy())
        return {"action": action, "status": "running"}

    def _mock_execute(self, subtask: Subtask, obs: Dict) -> Dict:
        action_dim = 32
        action = np.random.randn(1, 16, action_dim) * 0.01
        logger.info(f"[VLA MOCK] Executing: {subtask.instruction}")
        return {"action": action, "status": "running"}

    def is_done(self, subtask: Subtask, obs: Dict) -> bool:
        if subtask.task_type == TaskType.PICK:
            gripper_state = obs.get("gripper_closed", False)
            return gripper_state
        elif subtask.task_type == TaskType.PLACE:
            gripper_state = obs.get("gripper_open", True)
            return gripper_state
        return False


class NavDPExecutor(BaseExecutor):
    """
    NavDP executor: visual navigation diffusion policy.
    
    NavDP paper: https://arxiv.org/abs/2505.08712
    
    Input:
      - RGB-D images (single frame)
      - Navigation goal (one of 4 types):
        1. Point goal: relative (dx, dy) on 2D navigable plane
        2. Image goal: RGB observation from target location
        3. Trajectory goal: preferred trajectory projected to first-person view
        4. No goal: free roaming
    
    Output:
      - M dense SE2 waypoints: [(x, y, yaw), ...] for robot base to follow
      - Critic score: safety score for trajectory selection
    
    Key properties:
      - NavDP outputs waypoints with (x, y, yaw), so it CAN control turning
      - But NavDP is a LOCAL planner: it generates short-horizon waypoints
      - For long-range navigation, VLM must decompose into segments
      - NavDP handles obstacle avoidance within its local horizon
      - After NavDP reaches the target position, a turn_to step ensures
        the robot faces the correct direction for manipulation
    
    In real deployment, this wraps the NavDP ROS2 node or GitHub model.
    GitHub: https://github.com/InternRobotics/NavDP
    """

    def __init__(self, device: str = "cuda", navdp_ckpt: str = ""):
        self.device = device
        self.navdp_ckpt = navdp_ckpt
        self.policy = None
        self.goal_type = "point"
        self._load_model()

    def _load_model(self):
        if self.navdp_ckpt:
            logger.info(f"NavDP: loading model from {self.navdp_ckpt}")
            try:
                from navdp.models import NavDPModel
                self.policy = NavDPModel.from_pretrained(self.navdp_ckpt)
                self.policy = self.policy.to(self.device).eval()
                logger.info("NavDP model loaded successfully")
            except (ImportError, Exception) as e:
                logger.warning(f"NavDP model load failed: {e}, using mock")
                self.policy = None
        else:
            logger.info("NavDP: no checkpoint provided, using mock executor")
            self.policy = None

    def execute(self, subtask: Subtask, obs: Dict) -> Dict:
        if self.policy is None:
            return self._mock_execute(subtask, obs)

        import torch
        rgb = obs.get("head_rgb")
        depth = obs.get("head_depth")
        goal = obs.get("nav_goal", np.array([1.0, 0.0]))

        with torch.no_grad():
            trajectories, scores = self.policy.predict(
                rgb=rgb, depth=depth, goal=goal, goal_type=self.goal_type,
            )
        best_idx = scores.argmax()
        best_traj = trajectories[best_idx]

        waypoints = best_traj.cpu().numpy()
        action = self._waypoints_to_action(waypoints)
        return {"action": action, "status": "running", "waypoints": waypoints}

    def _waypoints_to_action(self, waypoints: np.ndarray) -> np.ndarray:
        """
        Convert NavDP SE2 waypoints [(x, y, yaw), ...] to R1Lite action format.
        
        R1Lite action space (32-DoF):
          [0:6]   left_arm
          [6:7]   left_gripper
          [7:13]  right_arm
          [13:14] right_gripper
          [14:20] torso.velocities (6-DoF)
          [20:26] chassis.velocities (6-DoF)
        
        NavDP waypoints → chassis velocity commands:
          vx = Δx / Δt
          vy = Δy / Δt
          ω = Δyaw / Δt
        """
        T = waypoints.shape[0]
        action = np.zeros((1, T, 32))
        if T >= 2:
            dx = np.diff(waypoints[:, 0])
            dy = np.diff(waypoints[:, 1])
            dyaw = np.diff(waypoints[:, 2])
            dt = 0.1
            vx = dx / dt
            vy = dy / dt
            omega = dyaw / dt
            action[0, :-1, 20] = vx[:T - 1]
            action[0, :-1, 21] = vy[:T - 1]
            action[0, :-1, 22] = omega[:T - 1]
        return action

    def _mock_execute(self, subtask: Subtask, obs: Dict) -> Dict:
        action = np.zeros((1, 16, 32))
        chassis_vel = np.random.randn(1, 16, 6) * 0.05
        torso_vel = np.random.randn(1, 16, 6) * 0.02
        action[:, :, 14:20] = torso_vel
        action[:, :, 20:26] = chassis_vel
        logger.info(f"[NavDP MOCK] Navigating to: {subtask.target}")
        return {"action": action, "status": "running"}

    def is_done(self, subtask: Subtask, obs: Dict) -> bool:
        current_pos = obs.get("robot_position", np.zeros(3))
        target_pos = obs.get("target_position", np.ones(3))
        dist = np.linalg.norm(current_pos[:2] - target_pos[:2])
        return dist < 0.3


class MPCExecutor(BaseExecutor):
    """
    MPC executor for turn_to tasks.
    Uses a simple proportional controller to turn the robot to face a target.
    
    In real deployment, this could use:
    - R1 Pro's built-in MPC controller
    - A simple PD controller on yaw angle
    - NavDP with image goal (target-facing image)
    
    Input: target direction/yaw
    Output: chassis angular velocity
    """

    def __init__(self, kp: float = 1.5, kd: float = 0.3):
        self.kp = kp
        self.kd = kd
        self.prev_error = 0.0

    def execute(self, subtask: Subtask, obs: Dict) -> Dict:
        current_yaw = obs.get("robot_yaw", 0.0)
        target_yaw = obs.get("target_yaw", 0.0)
        error = self._normalize_angle(target_yaw - current_yaw)
        d_error = error - self.prev_error
        self.prev_error = error
        omega = self.kp * error + self.kd * d_error
        omega = np.clip(omega, -1.0, 1.0)

        action = np.zeros((1, 16, 32))
        action[:, :, 22] = omega
        logger.info(f"[MPC] Turning: yaw_error={np.degrees(error):.1f}°, omega={omega:.2f}")
        return {"action": action, "status": "running"}

    def is_done(self, subtask: Subtask, obs: Dict) -> bool:
        current_yaw = obs.get("robot_yaw", 0.0)
        target_yaw = obs.get("target_yaw", 0.0)
        error = abs(self._normalize_angle(target_yaw - current_yaw))
        return error < np.radians(5)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


# ============================================================
# 5. Hierarchical Scheduler
# ============================================================

class HierarchicalScheduler:
    """
    Orchestrates VLM planning → subtask scheduling → executor switching.

    Flow:
      1. Receive human command
      2. VLM decomposes into subtask sequence
      3. For each subtask:
         a. Determine executor type (VLA or NavDP)
         b. Switch to appropriate executor
         c. Run execution loop until done or timeout
         d. Report status, move to next subtask
      4. Report overall result
    """

    def __init__(
        self,
        vlm_planner: VLMPlanner,
        vla_executor: VLAExecutor,
        navdp_executor: NavDPExecutor,
        mpc_executor: MPCExecutor,
        use_rule_based_fallback: bool = True,
    ):
        self.vlm_planner = vlm_planner
        self.vla_executor = vla_executor
        self.navdp_executor = navdp_executor
        self.mpc_executor = mpc_executor
        self.use_rule_based_fallback = use_rule_based_fallback
        self.rule_planner = RuleBasedPlanner() if use_rule_based_fallback else None

        self.current_executor: Optional[ExecutorType] = None
        self.subtask_history: List[Dict] = []
        self.step_count = 0

    def run(
        self,
        command: str,
        image: Optional[np.ndarray] = None,
        scene_description: str = "retail store",
        locations: str = "warehouse_shelf_A, retail_shelf_B3, checkout_counter",
        obs_provider: Optional[Any] = None,
    ) -> Dict:
        logger.info(f"=" * 60)
        logger.info(f"Received command: {command}")
        logger.info(f"=" * 60)

        subtasks = self._plan(command, image, scene_description, locations)
        if not subtasks:
            return {"status": "failed", "reason": "planning produced no subtasks"}

        logger.info(f"Plan: {len(subtasks)} subtasks")
        for i, st in enumerate(subtasks):
            logger.info(f"  [{i+1}] {st.executor.value:5s} | {st.task_type.value:8s} | {st.instruction}")

        results = []
        for i, subtask in enumerate(subtasks):
            logger.info(f"\n--- Subtask {i+1}/{len(subtasks)}: {subtask.instruction} ---")
            result = self._execute_subtask(subtask, obs_provider)
            results.append(result)
            subtask.status = result["status"]

            if result["status"] == "failed":
                logger.warning(f"Subtask failed: {subtask.instruction}")
                break

            logger.info(f"Subtask completed: {subtask.instruction} ({result['steps']} steps, {result['time']:.1f}s)")

        overall_status = "success" if all(r["status"] == "completed" for r in results) else "partial"
        if results and results[-1]["status"] == "failed":
            overall_status = "failed"

        summary = {
            "command": command,
            "status": overall_status,
            "total_subtasks": len(subtasks),
            "completed_subtasks": sum(1 for r in results if r["status"] == "completed"),
            "subtask_results": results,
            "plan": [st.to_dict() for st in subtasks],
        }
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Overall: {overall_status} ({summary['completed_subtasks']}/{summary['total_subtasks']} subtasks)")
        logger.info(f"{'=' * 60}")
        return summary

    def _plan(
        self,
        command: str,
        image: Optional[np.ndarray],
        scene_description: str,
        locations: str,
    ) -> List[Subtask]:
        try:
            subtasks = self.vlm_planner.plan(
                command=command,
                image=image,
                scene_description=scene_description,
                locations=locations,
            )
            if subtasks:
                logger.info("VLM planning succeeded")
                return subtasks
        except Exception as e:
            logger.warning(f"VLM planning failed: {e}")

        if self.use_rule_based_fallback and self.rule_planner:
            logger.info("Falling back to rule-based planner")
            return self.rule_planner.plan(command)

        return []

    def _execute_subtask(self, subtask: Subtask, obs_provider: Optional[Any] = None) -> Dict:
        executor = self._get_executor(subtask)
        new_executor_type = TASK_EXECUTOR_MAP[subtask.task_type]

        if self.current_executor != new_executor_type:
            logger.info(f"Switching executor: {self.current_executor} → {new_executor_type.value}")
            self.current_executor = new_executor_type

        if obs_provider is not None and hasattr(obs_provider, "__self__"):
            env = obs_provider.__self__
            if hasattr(env, "set_subtask_type"):
                env.set_subtask_type(subtask.task_type.value)
            if hasattr(env, "set_target") and subtask.target:
                env.set_target(subtask.target)

        start_time = time.time()
        step = 0
        status = "running"

        while step < subtask.max_steps:
            obs = self._get_obs(obs_provider)
            result = executor.execute(subtask, obs)
            step += 1
            self.step_count += 1

            if executor.is_done(subtask, obs):
                status = "completed"
                break

            elapsed = time.time() - start_time
            if elapsed > subtask.timeout_seconds:
                status = "timeout"
                break

            if step % 50 == 0:
                logger.info(f"  Step {step}/{subtask.max_steps}, elapsed {elapsed:.1f}s")

        if status == "running":
            status = "timeout"

        elapsed = time.time() - start_time
        return {
            "instruction": subtask.instruction,
            "task_type": subtask.task_type.value,
            "executor": subtask.executor.value,
            "status": status,
            "steps": step,
            "time": elapsed,
        }

    def _get_executor(self, subtask: Subtask) -> BaseExecutor:
        if subtask.executor == ExecutorType.VLA:
            return self.vla_executor
        elif subtask.executor == ExecutorType.NAVDP:
            return self.navdp_executor
        elif subtask.executor == ExecutorType.MPC:
            return self.mpc_executor
        raise ValueError(f"Unknown executor type: {subtask.executor}")

    def _get_obs(self, obs_provider: Optional[Any] = None) -> Dict:
        if obs_provider is not None:
            return obs_provider()
        return {
            "head_rgb": np.zeros((1, 3, 720, 1280), dtype=np.float32),
            "left_wrist_rgb": np.zeros((1, 3, 720, 1280), dtype=np.float32),
            "right_wrist_rgb": np.zeros((1, 3, 720, 1280), dtype=np.float32),
            "state": {"default": np.zeros((1, 32), dtype=np.float32)},
            "robot_position": np.array([0.0, 0.0, 0.0]),
            "target_position": np.array([1.0, 1.0, 0.0]),
            "gripper_closed": False,
            "gripper_open": True,
        }


# ============================================================
# 6. Simulation Environment (for testing without real robot)
# ============================================================

class SimulatedEnvironment:
    """
    Minimal simulation for testing the hierarchical pipeline.
    Simulates subtask completion after a random number of steps.
    """

    def __init__(self, steps_to_complete: int = 30):
        self.steps_to_complete = steps_to_complete
        self.step_counter = 0
        self.robot_position = np.array([0.0, 0.0, 0.0])
        self.robot_yaw = 0.0
        self.target_position = np.array([1.0, 1.0, 0.0])
        self.target_yaw = 0.0
        self.gripper_closed = False
        self.current_subtask_type = None

    def get_obs(self) -> Dict:
        self.step_counter += 1
        progress = self.step_counter / self.steps_to_complete

        if self.current_subtask_type == "navigate_to" and progress > 0.7:
            self.robot_position = self.target_position.copy()
        elif self.current_subtask_type == "turn_to" and progress > 0.5:
            self.robot_yaw = self.target_yaw
        elif self.current_subtask_type == "pick" and progress > 0.5:
            self.gripper_closed = True
        elif self.current_subtask_type == "place" and progress > 0.5:
            self.gripper_closed = False

        return {
            "head_rgb": np.random.rand(1, 3, 720, 1280).astype(np.float32),
            "left_wrist_rgb": np.random.rand(1, 3, 720, 1280).astype(np.float32),
            "right_wrist_rgb": np.random.rand(1, 3, 720, 1280).astype(np.float32),
            "state": {"default": np.random.rand(1, 32).astype(np.float32)},
            "robot_position": self.robot_position.copy(),
            "robot_yaw": self.robot_yaw,
            "target_position": self.target_position.copy(),
            "target_yaw": self.target_yaw,
            "gripper_closed": self.gripper_closed,
            "gripper_open": not self.gripper_closed,
        }

    def set_target(self, target: str):
        target_map = {
            "warehouse": (np.array([2.0, 0.0, 0.0]), 0.0),
            "retail_shelf_B3": (np.array([5.0, 3.0, 0.0]), np.pi / 2),
            "checkout_counter": (np.array([1.0, 5.0, 0.0]), np.pi),
        }
        for key, (pos, yaw) in target_map.items():
            if key in target.lower():
                self.target_position = pos
                self.target_yaw = yaw
                break
        self.step_counter = 0
        self.gripper_closed = False

    def set_subtask_type(self, task_type: str):
        self.current_subtask_type = task_type
        self.step_counter = 0


# ============================================================
# 7. Main Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Hierarchical VLM+VLA Task Execution")
    parser.add_argument("--command", type=str, default="restock the chips to shelf B3",
                        help="Natural language command from human")
    parser.add_argument("--vlm_provider", type=str, default="qwen",
                        choices=["qwen", "gemini", "openai", "rule"],
                        help="VLM API provider (use 'rule' for rule-based only)")
    parser.add_argument("--vlm_api_key", type=str, default="",
                        help="API key for VLM service")
    parser.add_argument("--vla_ckpt", type=str, default="",
                        help="Path to VLA checkpoint (leave empty for mock)")
    parser.add_argument("--scene", type=str, default="retail store",
                        help="Scene description for VLM context")
    parser.add_argument("--locations", type=str,
                        default="warehouse_shelf_A, retail_shelf_B3, checkout_counter",
                        help="Available locations for navigation")
    parser.add_argument("--mode", type=str, default="simulate",
                        choices=["simulate", "real"],
                        help="Execution mode")
    parser.add_argument("--output", type=str, default="",
                        help="Output JSON file path for results")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.vlm_provider == "rule":
        vlm_planner = None
    else:
        vlm_planner = VLMPlanner(
            provider=args.vlm_provider,
            api_key=args.vlm_api_key,
        )

    vla_executor = VLAExecutor(ckpt_path=args.vla_ckpt)
    navdp_executor = NavDPExecutor()
    mpc_executor = MPCExecutor()

    scheduler = HierarchicalScheduler(
        vlm_planner=vlm_planner,
        vla_executor=vla_executor,
        navdp_executor=navdp_executor,
        mpc_executor=mpc_executor,
        use_rule_based_fallback=True,
    )

    if args.mode == "simulate":
        sim = SimulatedEnvironment(steps_to_complete=30)
        obs_provider = sim.get_obs
    else:
        obs_provider = None

    result = scheduler.run(
        command=args.command,
        scene_description=args.scene,
        locations=args.locations,
        obs_provider=obs_provider,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
