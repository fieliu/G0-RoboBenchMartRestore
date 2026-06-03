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
    """What the VLM reports when a sensor trigger fires mid-subtask. The verdict
    drives the scheduler state machine (see Scheduler._step_subtask):
      IN_PROGRESS  world matches expectation, not at the terminal state yet
                   -> change NOTHING, let the VLA keep running this subtask
      SUCCESS      terminal state reached (item in basket / on shelf)
                   -> advance to the NEXT subtask (instruction changes)
      RETRY        failed but world is intact (grasp slipped, item still there)
                   -> reset the executor and re-run the SAME subtask
      REPLAN       failed AND world state breaks the remaining plan
                   (item on the floor, not where the next subtask assumes)
                   -> ask the VLM for a fresh plan from the current state
      ABORT        unrecoverable (collision, out of reach, scene changed)
                   -> stop; open-loop fallback treats this as task failure
    """
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    RETRY = "retry"
    REPLAN = "replan"
    ABORT = "abort"


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
Navigation (executed by a local planner that avoids obstacles within a segment):
  navigate_to              Drive in a STRAIGHT LINE to ONE waypoint. target = "x,y".
  turn_to                  Rotate in place to an absolute heading. target = degrees
                           (0=east, 90=north, 180=west, 270=south).

=== RULES ===
1. type MUST be one of: pick_to_basket, restock_basket_to_shelf, pick_from_floor,
   navigate_to, turn_to.
2. NAVIGATION IS MULTI-SEGMENT. Each navigate_to covers ONE straight segment
   between adjacent waypoints. To travel far, emit a CHAIN of navigate_to:
   current -> waypoint1 -> waypoint2 -> shelf_approach_point.
   Pick the waypoint sequence from the listed CORRIDOR WAYPOINTS so consecutive
   points are roughly in a straight line (no diagonal cutting through shelves).
3. navigate_to targets MUST be exact "x,y" taken from the listed waypoints or
   shelf approach points. Never invent coordinates.
4. The local planner turns as needed WHILE driving a segment. Do NOT insert
   turn_to between two navigate_to commands.
5. Use turn_to ONLY after arriving at a shelf approach point, to face the shelf
   for manipulation (target = that approach point's face degrees).
6. Before any manipulation the robot must be at the shelf's approach point AND
   facing it.
7. Output ONLY a JSON array. No prose, no markdown fences.

Each subtask object has keys: "type", "instruction", "target"."""

USER_TMPL = """{map_block}

=== HUMAN COMMAND ===
{command}

Decompose the command into the full subtask sequence from the robot's current
position. Remember: navigation is a CHAIN of straight navigate_to segments
between adjacent waypoints. Output ONLY the JSON array."""

# Judge prompt: called when a cheap sensor trigger fires mid-subtask. The VLM
# looks at the head camera BEFORE/AFTER frames and decides the verdict that
# drives the scheduler state machine.
JUDGE_SYSTEM_PROMPT = """You verify a retail robot's subtask from two head-camera frames
(BEFORE and AFTER a short action burst). Report ONE verdict — be conservative.

verdict (use these EXACT strings):
  in_progress  The robot is still doing the subtask correctly; the terminal
               goal is NOT yet reached. (Most common — default to this.)
  success      The subtask's terminal goal IS achieved
               (item is in the basket / placed on the shelf).
  retry        It failed but the scene is intact and the SAME subtask can be
               retried (e.g. grasp slipped, but the item is still in place).
  replan       It failed AND the world changed so the remaining plan no longer
               holds (e.g. the item fell on the floor).
  abort        Unrecoverable (collision, item gone, robot stuck, out of reach).

Output ONLY JSON: {"verdict": "...", "reason": "<short>"}"""

JUDGE_USER_TMPL = """High-level goal: {coarse_task}
Current subtask: {subtask}
The two images are the head camera BEFORE and AFTER a short action burst.
Has this subtask reached its terminal goal, or what went wrong?
Output ONLY the JSON verdict."""
# --- section: planner ---


# ============================================================
# 4. VLM planner — ONE call, returns the whole subtask sequence
# ============================================================
class VLMPlanner:
    def __init__(self, provider: str = "qwen", api_key: str = "", model: str = ""):
        self.provider = provider.lower()
        self.api_key = api_key or os.environ.get("VLM_API_KEY", "")
        self.model = model or {"qwen": "qwen-vl-max", "gemini": "gemini-2.0-flash",
                               "openai": "gpt-4o"}.get(self.provider, "qwen-vl-max")
        self._init_client()

    def _init_client(self):
        self.client = None
        if self.provider in ("gemini", "openai"):
            import openai
            base_url = {"gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
                        "openai": None}[self.provider]
            self.client = openai.OpenAI(api_key=self.api_key, base_url=base_url)
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
              before: Optional[np.ndarray], after: Optional[np.ndarray]) -> "JudgeVerdict":
        """Verify a subtask from head-camera BEFORE/AFTER frames. Returns a
        JudgeVerdict. On any error or unparseable reply, defaults to IN_PROGRESS
        (conservative: change nothing rather than wrongly advance/abort)."""
        user = JUDGE_USER_TMPL.format(coarse_task=coarse_task, subtask=subtask)
        imgs = [im for im in (before, after) if im is not None]
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
        if self.provider in ("gemini", "openai") and self.client is not None:
            return self._call_openai(prompt, images, system)
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
    """Point-goal navigation for ONE straight segment. The local planner
    (NavDP / iPlanner / Nav2) drives to the goal (x, y), turning and avoiding
    obstacles within the segment, then reports arrival. We hand it one goal
    coordinate per navigate_to subtask and poll arrival."""

    def __init__(self, device: str = "cuda", ckpt: str = ""):
        self.device = device
        self.ckpt = ckpt
        logger.info("Nav: point-goal executor ready (one straight segment per call)")

    def set_goal(self, obs_provider, target: str) -> np.ndarray:
        x, y = [float(v) for v in target.split(",")]
        env = getattr(obs_provider, "__self__", None)
        if env is not None and hasattr(env, "set_nav_goal"):
            env.set_nav_goal(np.array([x, y]))
        return np.array([x, y])

    def act(self, subtask: Subtask, obs: Dict) -> np.ndarray:
        # Real deployment: the nav node drives the base. Here: no-op step.
        return np.zeros((16, 15), dtype=np.float32)


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

        # The full human command is the high-level goal for the VLA executor.
        # G0 was trained on "[High]: {coarse_task}, [Low]: {task}", so feeding
        # the command here lets each subtask inherit the overall intent.
        self.vla.set_coarse_task(command)

        logger.info("=" * 64)
        logger.info(f"COMMAND: {command}")
        plan = self.vlm.plan(command, map_block, obs_provider().get("head_rgb_raw"))
        if not plan:
            return {"command": command, "status": "failed", "reason": "empty plan"}
        self._log_plan(plan)

        # State machine over the subtask sequence. A VLA subtask's VLM judge can
        # return RETRY (re-run same subtask), REPLAN (regenerate the remaining
        # plan from current state), or ABORT (give up). Nav/turn never judge.
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
                i += 1
                continue

            if verdict is JudgeVerdict.RETRY:
                if subtask.retries < self.max_retries:
                    subtask.retries += 1
                    logger.warning(f"  RETRY {subtask.retries}/{self.max_retries}: {subtask.instruction}")
                    continue                              # same i: re-run this subtask
                logger.error(f"  retries exhausted -> failing: {subtask.instruction}")
                subtask.status = "failed"
                return self._summary(command, "failed", plan, completed)

            if verdict is JudgeVerdict.REPLAN:
                if replans < self.max_replans:
                    replans += 1
                    logger.warning(f"  REPLAN {replans}/{self.max_replans} from current state")
                    pose = self._pose(obs_provider)
                    fresh = self.vlm.plan(command, self.map.render(pose["xy"], pose["yaw_deg"]),
                                          obs_provider().get("head_rgb_raw"))
                    if fresh:
                        plan = plan[:i] + fresh           # keep done prefix, swap the rest
                        self._log_plan(plan)
                        continue                          # same i: start of the new tail
                logger.error("  replan budget exhausted (or empty plan) -> failing")
                subtask.status = "failed"
                return self._summary(command, "failed", plan, completed)

            # ABORT (includes max_steps timeout)
            logger.error(f"  ABORT: {subtask.instruction}")
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

        env = getattr(obs_provider, "__self__", None)
        if env is not None and hasattr(env, "set_subtask"):
            env.set_subtask(subtask)

        # VLA subtasks get the VLM judge; nav/turn use the sensor signal directly
        # (arrival / heading are unambiguous, no vision needed).
        judging = is_vla and self.use_vlm_judge and self.judge_fn is not None
        step, last_judge = 0, 0
        obs = obs_provider()
        before = obs.get("head_rgb_raw") if judging else None

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
                        v = self.judge_fn(self.vla.coarse_task, subtask.instruction,
                                          before, obs.get("head_rgb_raw"))
                        logger.info(f"  trigger@{step} -> judge={v.value}")
                        if v is JudgeVerdict.IN_PROGRESS:
                            before = obs.get("head_rgb_raw")   # refresh baseline, keep going
                        else:
                            return v                      # SUCCESS / RETRY / REPLAN / ABORT
                if step >= subtask.max_steps:
                    break
        logger.warning(f"  hit max_steps ({subtask.max_steps}) -> ABORT")
        return JudgeVerdict.ABORT
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
            return bool(obs.get("gripper_closed", False))
        if t == "restock_basket_to_shelf":
            return bool(obs.get("gripper_open", False))
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
# --- section: simenv ---


# ============================================================
# 7. Simulated environment (test the loop without a real robot)
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
            "right_wrist_rgb": np.zeros((3, 224, 224), np.float32),
            "head_rgb_raw": np.zeros((224, 224, 3), np.uint8),
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
    p = argparse.ArgumentParser(description="Hierarchical VLM+VLA supermarket deployment (open-loop)")
    p.add_argument("--command", required=True, help="high-level human command")
    p.add_argument("--map-file", required=True, help="store map JSON")
    p.add_argument("--vlm-provider", default="qwen", choices=["qwen", "gemini", "openai"])
    p.add_argument("--vlm-api-key", default="")
    p.add_argument("--vlm-model", default="")
    p.add_argument("--vla-ckpt", default="", help="Fetch LoRA checkpoint (empty -> mock)")
    p.add_argument("--no-vlm-judge", action="store_true",
                   help="disable closed-loop VLM completion check (sensor-only, no recovery)")
    p.add_argument("--mode", default="simulate", choices=["simulate", "real"])
    p.add_argument("--output", default="")
    return p.parse_args()


def main():
    args = parse_args()
    store_map = StoreMap.from_file(args.map_file)
    vlm = VLMPlanner(args.vlm_provider, args.vlm_api_key, args.vlm_model)
    vla = VLAExecutor(ckpt_path=args.vla_ckpt)
    nav = NavExecutor()
    turn = TurnExecutor()
    scheduler = Scheduler(vlm, store_map, vla, nav, turn,
                          use_vlm_judge=not args.no_vlm_judge)

    if args.mode == "simulate":
        obs_provider = SimEnv(steps_to_done=15).get_obs
    else:
        raise NotImplementedError("real mode: wire obs_provider to your ROS2 bridge")

    result = scheduler.run(args.command, obs_provider)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"saved -> {args.output}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
