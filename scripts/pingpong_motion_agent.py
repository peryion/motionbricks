from copy import deepcopy

import mujoco
import numpy as np
import torch as t

from motionbricks.helper.mujoco_helper import get_mujoco_converter
from motionbricks.geometry.quaternions import matrix_to_quaternion
from motionbricks.motionlib.core.utils.rotations import (
    matrix_to_cont6d,
    quaternion_to_matrix,
)
from motionbricks.motion_backbone.inference.motion_inference import motion_inference


def angle_to_Z_rotation_matrix(angle):
    """Create rotation matrix around Z-axis for Z-up coordinate system"""
    cos, sin = t.cos(angle), t.sin(angle)
    one, zero = t.ones_like(angle), t.zeros_like(angle)
    # Z-axis rotation matrix:
    # [cos(θ) -sin(θ)  0]
    # [sin(θ)  cos(θ)  0]
    # [  0       0     1]
    mat = t.stack((cos, -sin, zero, sin, cos, zero, zero, zero, one), -1)
    mat = mat.reshape(angle.shape + (3, 3))
    return mat

class PingPongMotionAgent(t.nn.Module):
    """MotionBricks agent with target-pose conditioning.

    This class keeps only the generation path used by the ping-pong demo. It is
    independent from the generic navigation agent, while still using the same
    model IO convention.
    """

    NUM_FRAMES_PER_TOKEN = 4
    DEFAULT_PRED_OFFSETS = 4

    def __init__(
        self,
        inferencer: motion_inference,
        device: str = "cuda",
        skeleton_xml: str = "assets/skeletons/g1/g1.xml",
        initial_qpos = None,
        target_root_realignment: bool = True,
        source_root_realignment: bool = True,
        filter_qpos: bool = True,
        force_canonicalization: bool = True,
        skip_ending_target_cond: bool = True,
    ):
        super().__init__()
        if initial_qpos is None:
            raise ValueError("PingPongMotionAgent requires initial_qpos (got None).")
        self._inferencer = inferencer.eval().to(device)
        self._motion_rep = deepcopy(inferencer.motion_rep).to(device)
        self._converter = get_mujoco_converter(self._motion_rep, skeleton_xml).to(device)
        self._device = device
        self._fps = self._motion_rep.fps
        self._initial_qpos = t.tensor(initial_qpos).to(device)
        self._masked_num_tokens = int(inferencer._root_model.backbone_net.MASKED_NUM_TOKENS)
        self._target_root_realignment = target_root_realignment
        self._source_root_realignment = source_root_realignment
        self.PRED_OFFSETS = self.DEFAULT_PRED_OFFSETS
        self.FILTER_QPOS = filter_qpos
        self.FORCE_CANONICALIZATION = force_canonicalization
        self.SKIP_ENDING_TARGET_COND = skip_ending_target_cond
        self.frames = {"model_features": None, "mujoco_qpos": None}
        self.latest_model_target_roots = None
        self._initialize_frames()

    def reset(self) -> None:
        self._initialize_frames()

    def _initialize_frames(self) -> None:
        self._current_frame_idx = 0
        num_frames = 64
        self.frames["mujoco_qpos"] = self._initial_qpos.view(1, 1, -1).repeat(1, num_frames, 1)
        self.frames["model_features"] = t.empty(1, num_frames, 0, device=self._device)

    def is_done(self) -> bool:
        # True only AFTER the last frame has been returned by get_next_frame
        # (idx is allowed to grow to shape[1] before is_done flips, so the
        # state machine sees a transition exactly one tick after the last
        # frame plays; otherwise hit's final frame is skipped — see hit→recovery
        # join discontinuity).
        return self._current_frame_idx >= self.frames["mujoco_qpos"].shape[1]

    def generate_new_frames(self, input: dict, controller_dt: float = 0.25, force_generation: bool = False):
        """Run model inference and overwrite the rolling frame buffer.

        The ping-pong demo always drives this via ``force_generate_and_trim``,
        so the body unconditionally regenerates. ``controller_dt`` and
        ``force_generation`` are kept for caller compatibility.

        Required input keys:
            context_mujoco_qpos: [B, NUM_FRAMES_PER_TOKEN, qpos_dim]
            target_mujoco_qpos:  [B, NUM_FRAMES_PER_TOKEN, qpos_dim]
            movement_direction:  [B, 3] in mujoco world frame
            facing_direction:    [B, 3] in mujoco world frame
        """
        del controller_dt, force_generation
        input = {key: value.to(self._device) for key, value in input.items() if value is not None}

        input["context_global_joint_positions"], input["context_global_joint_rotations"] = (
            self._process_input_to_joint_transforms(input)
        )
        (
            input["target_global_joint_positions"],
            input["target_global_joint_rotations"],
            input["target_global_root_positions"],
        ) = self._generate_target_joint_transforms(input)

        model_features, mujoco_qpos, num_pred_frames = self._generate_inbetween_frames(input)
        self.frames["model_features"] = model_features[:, :num_pred_frames.item(), :]
        self.frames["mujoco_qpos"] = mujoco_qpos[:, :num_pred_frames.item(), :]
        return self.frames["mujoco_qpos"], num_pred_frames

    def _process_input_to_joint_transforms(self, input: dict):
        """ @brief: process the input to joint transforms
        """
        if "context_mujoco_qpos" in input:
            if self.FORCE_CANONICALIZATION:
                self._canonicalize_mujoco_qpos(input)
            else:
                input["raw_context_mujoco_qpos"] = input["context_mujoco_qpos"].clone()
            return self._converter.convert_mujoco_qpos_to_motion_transforms(input["context_mujoco_qpos"])
        if "context_global_joint_positions" in input and "context_global_joint_rotations" in input:
            if self.FORCE_CANONICALIZATION:
                raise NotImplementedError("Canonicalization currently expects context_mujoco_qpos.")
            return input["context_global_joint_positions"], input["context_global_joint_rotations"]
        if "context_motion_features" in input:
            if self.FORCE_CANONICALIZATION:
                raise NotImplementedError("Canonicalization currently expects context_mujoco_qpos.")
            results = self._motion_rep.inverse(
                input["context_motion_features"], is_normalized=False, return_quat=False, return_all=False)
            return results["posed_joints"], results["global_joint_rots"]
        raise ValueError("Invalid input: missing context_mujoco_qpos or context joint transforms.")

    def _canonicalize_target_qpos(self, qpos: t.Tensor, input: dict) -> t.Tensor:
        if not self.FORCE_CANONICALIZATION or "first_frame_heading_angle" not in input:
            return qpos
        qpos = qpos.clone()
        first_pos = input["first_frame_position"].to(qpos.device, qpos.dtype)
        first_heading = input["first_frame_heading_angle"].to(qpos.device, qpos.dtype)
        inv_heading = angle_to_Z_rotation_matrix(first_heading).to(qpos.device, qpos.dtype).transpose(-2, -1)
        qpos[:, :, :3] = t.matmul(
            inv_heading[:, None], (qpos[:, :, :3] - first_pos[:, None])[..., None])[..., 0]
        root_rot = quaternion_to_matrix(qpos[:, :, 3:7])
        qpos[:, :, 3:7] = matrix_to_quaternion(t.matmul(inv_heading[:, None], root_rot))
        return qpos

    def _generate_target_joint_transforms(self, input: dict):
        target_qpos = input["target_mujoco_qpos"]
        target_qpos = self._canonicalize_target_qpos(target_qpos, input)
        joint_pos, joint_rot = self._converter.convert_mujoco_qpos_to_motion_transforms(target_qpos)
        root_motion = (
            joint_pos[:, :, 0]
            * t.tensor([1.0, 0.0, 1.0], device=joint_pos.device, dtype=joint_pos.dtype)
        )
        joint_pos = joint_pos - root_motion[:, :, None, :]

        roots_mujoco = t.matmul(
            self._converter.motion_to_mujoco_matrix.to(root_motion.device, root_motion.dtype)[None, None],
            root_motion[..., None],
        )[..., 0]
        if self.FORCE_CANONICALIZATION and "first_frame_heading_angle" in input:
            rot = angle_to_Z_rotation_matrix(
                input["first_frame_heading_angle"]).to(roots_mujoco.device, roots_mujoco.dtype)
            roots_mujoco = t.matmul(rot[:, None], roots_mujoco[..., None])[..., 0]
            first_pos = input["first_frame_position"].to(roots_mujoco.device, roots_mujoco.dtype)
            roots_mujoco = roots_mujoco + first_pos[:, None]
        self.latest_model_target_roots = roots_mujoco[0].detach().cpu().numpy()
        return joint_pos, joint_rot, root_motion

    def _generate_inbetween_frames(self, input: dict):
        batch_size, MASKED_NUM_TOKENS = 1, self._masked_num_tokens
        fps = self._inferencer.local_motion_rep.fps
        root_joint_idx = 0

        # prepare the values for the context frames
        context_global_root_pos = input['context_global_joint_positions'][:, :, root_joint_idx, :]
        context_rotation_angle = t.atan2(input['context_global_joint_rotations'][:, :, root_joint_idx, 0, 2],
                                         input['context_global_joint_rotations'][:, :, root_joint_idx, 2, 2])
        context_global_root_values = t.cat([context_global_root_pos, t.cos(context_rotation_angle)[..., None],
                                            t.sin(context_rotation_angle)[..., None]], dim=-1)  # [B, numF, 5]
        context_local_root_values = \
            t.zeros([batch_size, self.NUM_FRAMES_PER_TOKEN, 4]).to(self._device)  # [B, numF, 4]
        context_local_root_values[:, :self.NUM_FRAMES_PER_TOKEN - 1, 0] = \
            (((context_rotation_angle[:, 1:] - context_rotation_angle[:, :-1] + t.pi) % (2 * t.pi)) - t.pi) * fps
        context_local_root_values[:, :self.NUM_FRAMES_PER_TOKEN - 1, 1: 3] = \
            (context_global_root_pos[:, 1:, [0, 2]] - context_global_root_pos[:, :-1, [0, 2]]) * fps
        context_local_root_values[:, :self.NUM_FRAMES_PER_TOKEN - 1, 3] = \
            context_global_root_values[:, :self.NUM_FRAMES_PER_TOKEN - 1, 1]

        context_global_joint_positions = input['context_global_joint_positions'].clone()
        joint_positions = context_global_joint_positions[:, :, 1:, :]
        joint_positions[..., 0] = \
            context_global_joint_positions[:, :, 1:, 0] - context_global_joint_positions[:, :, :1, 0]
        joint_positions[..., 2] = \
            context_global_joint_positions[:, :, 1:, 2] - context_global_joint_positions[:, :, :1, 2]

        joint_rotation_ortho6d = matrix_to_cont6d(input['context_global_joint_rotations'])
        context_local_poses = t.cat([joint_positions.view([batch_size, self.NUM_FRAMES_PER_TOKEN, -1]),
                                     joint_rotation_ortho6d.view([batch_size, self.NUM_FRAMES_PER_TOKEN, -1])], dim=-1)

        # prepare the values for the target frames
        target_global_root_pos = input['target_global_root_positions'] + \
            input['target_global_joint_positions'][:, :, root_joint_idx, :]
        target_rotation_angle = t.atan2(input['target_global_joint_rotations'][:, :, root_joint_idx, 0, 2],
                                        input['target_global_joint_rotations'][:, :, root_joint_idx, 2, 2])
        if 'target_root_headings' in input:
            target_rotation_angle = input['target_root_headings']  # avoid double counting AND avoid ill defined angles
        target_rotation_angle = target_rotation_angle.float()

        target_global_root_values = t.cat([target_global_root_pos, t.cos(target_rotation_angle)[..., None],
                                           t.sin(target_rotation_angle)[..., None]], dim=-1)  # [B, num_frames, 5]
        target_local_root_values = t.zeros_like(context_local_root_values).to(self._device)  # [b=1, num_frames, 4]
        target_local_root_values[:, :self.NUM_FRAMES_PER_TOKEN - 1, 0] = \
            (((target_rotation_angle[:, 1:] - target_rotation_angle[:, :-1] + t.pi) % (2 * t.pi)) - t.pi) * fps
        target_local_root_values[:, :self.NUM_FRAMES_PER_TOKEN - 1, 1: 3] = \
            (target_global_root_pos[:, 1:, [0, 2]] - target_global_root_pos[:, :-1, [0, 2]]) * fps
        target_local_root_values[:, -1, 0: 3] = target_local_root_values[:, -2, 0: 3]  # add the last velocity
        target_local_root_values[:, :, 3] = target_global_root_values[:, :, 1]

        joint_positions = input['target_global_joint_positions'][:, :, 1:, :]
        joint_rotation_ortho6d = matrix_to_cont6d(input['target_global_joint_rotations'])
        target_local_poses = t.cat([joint_positions.view([batch_size, self.NUM_FRAMES_PER_TOKEN, -1]),
                                    joint_rotation_ortho6d.view([batch_size, self.NUM_FRAMES_PER_TOKEN, -1])], dim=-1)

        # merge the constraints
        local_root_values = t.cat([context_local_root_values, target_local_root_values], dim=1)
        global_root_values = t.cat([context_global_root_values, target_global_root_values], dim=1)
        local_poses = t.cat([context_local_poses, target_local_poses], dim=1)

        has_global_root_values = t.ones_like(global_root_values[:, :, 0], dtype=t.bool)
        has_local_root_values = t.ones_like(local_root_values[:, :, 0], dtype=t.bool)
        has_local_poses = t.ones_like(local_poses[:, :, 0], dtype=t.bool)
        has_local_root_values[:, self.NUM_FRAMES_PER_TOKEN - 1] = False  # the last velocity is incorrect

        if 'target_global_root_mask' in input:
            has_global_root_values[:, -self.NUM_FRAMES_PER_TOKEN:] = input['target_global_root_mask'].bool()
        if 'target_local_root_mask' in input:
            has_local_root_values[:, -self.NUM_FRAMES_PER_TOKEN:] = input['target_local_root_mask'].bool()
        if 'target_pose_mask' in input:
            has_local_poses[:, -self.NUM_FRAMES_PER_TOKEN:] = input['target_pose_mask'].bool()

        if not self._target_root_realignment:
            # if root is not realigned, disable the following info since they might be misleading
            has_local_root_values[:, -self.NUM_FRAMES_PER_TOKEN:] = False
            has_global_root_values[:, -self.NUM_FRAMES_PER_TOKEN + 1:] = False
            has_local_poses[:, -self.NUM_FRAMES_PER_TOKEN + 1:] = False

        num_tokens = t.full([batch_size], MASKED_NUM_TOKENS).int().to(self._device)

        # pred the motions
        config = {'num_inference_step': 1, 'smooth_root_traj': False, 'allow_pred_out_of_reach_num_tokens': False,
                  'pose_token_sampling_use_argmax': True, 'skip_ending_target_cond': self.SKIP_ENDING_TARGET_COND}
        info = {}
        pred_global_motions, num_pred_tokens = self._inferencer.predict(
            global_root_values, has_global_root_values, local_root_values, has_local_root_values,
            local_poses, has_local_poses, num_tokens, config=config, info=info,
            allowed_pred_num_tokens=input.get('allowed_pred_num_tokens', None)
        )

        self.frames['model_features'] = pred_global_motions
        self.frames['num_pred_frames'] = self.NUM_FRAMES_PER_TOKEN * num_pred_tokens

        self.frames['mujoco_qpos'] = \
            self._converter.convert_motion_features_to_mujoco_qpos(self.frames['model_features'], self._motion_rep, False)
        root_rot = self.frames['mujoco_qpos'][:, :, 3: 7].clone()
        self.frames['mujoco_qpos'][:, :, 3: 7] = root_rot[:, :, [3, 0, 1, 2]]
        if self.FORCE_CANONICALIZATION:
            input['mujoco_qpos'] = self.frames['mujoco_qpos']
            self.frames['mujoco_qpos'] = self._uncanonicalize_mujoco_qpos(input)
        self._current_frame_idx = self.NUM_FRAMES_PER_TOKEN - self.PRED_OFFSETS

        if self.FILTER_QPOS:
            # blend the generated first frames with the context frames for smooth transitions
            # can remove since it does not cause visual difference in the motion
            self.frames['raw_mujoco_qpos'] = self.frames['mujoco_qpos'].clone()
            ctx = input['raw_context_mujoco_qpos']
            num_ctx = ctx.shape[1]
            blend = t.linspace(0.3, 0.7, num_ctx)[None, :, None].to(ctx.device)
            self.frames['mujoco_qpos'][:, :num_ctx, :3] = \
                ctx[:, :, :3] * (1 - blend) + self.frames['mujoco_qpos'][:, :num_ctx, :3] * blend
            self.frames['mujoco_qpos'][:, :num_ctx, 7:] = \
                ctx[:, :, 7:] * (1 - blend) + self.frames['mujoco_qpos'][:, :num_ctx, 7:] * blend

        return self.frames['model_features'], self.frames['mujoco_qpos'], self.frames['num_pred_frames']

    def get_next_frame(self):
        # The READ index is clamped so IDLE keeps repeating the last frame;
        # the STORED index is allowed to grow to shape[1] so is_done() can flip
        # exactly one tick after the last frame has been returned.
        n = self.frames["mujoco_qpos"].shape[1]
        current_frame_idx = min(self._current_frame_idx, n - 1)
        next_qpos = self.frames["mujoco_qpos"][0, current_frame_idx]
        self._current_frame_idx = min(self._current_frame_idx + 1, n)
        if type(next_qpos) == t.Tensor:
            next_qpos = next_qpos.detach().cpu().numpy()
        return next_qpos

    def get_context_mujoco_qpos(self):
        indices = [max(0, min(self._current_frame_idx - self.NUM_FRAMES_PER_TOKEN + i + self.PRED_OFFSETS,
                              self.frames['mujoco_qpos'].shape[1] - 1))
                   for i in range(self.NUM_FRAMES_PER_TOKEN)]
        return self.frames['mujoco_qpos'][:, indices, :].to(self._device)

    def _canonicalize_mujoco_qpos(self, input: dict):
        # Pin everything to float32 here so downstream matmuls don't hit dtype
        # mismatches (callers sometimes hand in float64 qpos from numpy).
        mujoco_qpos = input['context_mujoco_qpos'].float()
        input['context_mujoco_qpos'] = mujoco_qpos
        input['raw_context_mujoco_qpos'] = mujoco_qpos.clone()

        # first frame information
        first_frame_position = mujoco_qpos[:, 0, :3].clone() * t.tensor([[1.0, 1.0, 0.0]]).to(mujoco_qpos.device)
        first_frame_rot = quaternion_to_matrix(mujoco_qpos[:, 0, 3: 7].clone())  # the rotation of first frame
        first_frame_heading_angle = t.atan2(first_frame_rot[:, 1, 0], first_frame_rot[:, 0, 0])
        first_frame_heading_angle[first_frame_heading_angle.isnan()] = 0.0
        first_frame_rot_heading = angle_to_Z_rotation_matrix(first_frame_heading_angle)
        inverse_first_frame_rot_heading = first_frame_rot_heading.transpose(-2, -1)

        # get the canonicalized root info
        canonicalized_root_position = \
            t.matmul(inverse_first_frame_rot_heading[:, None, :, :],
                     (mujoco_qpos[:, :, :3].clone() - first_frame_position)[..., None])[..., 0]

        canonicalized_rot_matrix = t.matmul(inverse_first_frame_rot_heading[:, None, :, :],
                                            quaternion_to_matrix(mujoco_qpos[:, :, 3: 7]))

        mujoco_qpos[:, :, 3: 7] = matrix_to_quaternion(canonicalized_rot_matrix)
        mujoco_qpos[:, :, :3] = canonicalized_root_position

        # canonicalize the movement & facing direction
        input['movement_direction'] = t.matmul(inverse_first_frame_rot_heading,
                                               input['movement_direction'][..., None].float())[..., 0]
        input['facing_direction'] = t.matmul(inverse_first_frame_rot_heading,
                                             input['facing_direction'][:, :, None].float())[..., 0]
        input['first_frame_heading_angle'] = first_frame_heading_angle
        input['first_frame_position'] = first_frame_position
        input['context_mujoco_qpos'] = mujoco_qpos

        # also if specific target headings are provided, canonicalize them
        if 'specific_target_headings' in input:
            input['specific_target_headings'] = \
                input['specific_target_headings'] - first_frame_heading_angle.view([-1, 1])
            input['specific_target_positions'] = \
                t.matmul(inverse_first_frame_rot_heading[:, None, :, :],
                         (input['specific_target_positions'] - first_frame_position[:, None, :])[..., None])[..., 0]

    def _uncanonicalize_mujoco_qpos(self, input: dict):
        mujoco_qpos = input['mujoco_qpos']
        first_frame_heading_angle = input['first_frame_heading_angle']
        first_frame_position = input['first_frame_position']

        # the first frame
        first_frame_rot_heading = angle_to_Z_rotation_matrix(first_frame_heading_angle)

        # get the uncanonicalized root information
        current_first_frame_rotation = quaternion_to_matrix(mujoco_qpos[:, :1, 3: 7])
        current_first_frame_heading_angle = t.atan2(current_first_frame_rotation[:, :, 1, 0],
                                                    current_first_frame_rotation[:, :, 0, 0])
        current_first_frame_rot_heading = angle_to_Z_rotation_matrix(current_first_frame_heading_angle)
        rot_matrix = quaternion_to_matrix(mujoco_qpos[:, :, 3: 7])
        rot_matrix = t.matmul(first_frame_rot_heading[:, None, :, :],
                              t.matmul(current_first_frame_rot_heading.transpose(-2, -1), rot_matrix))
        root_positions = t.matmul(first_frame_rot_heading[:, None, :, :],
                                  t.matmul(current_first_frame_rot_heading.transpose(-2, -1),
                                           mujoco_qpos[:, :, :3, None]))[..., 0]
        root_positions = root_positions - \
            root_positions[:, :1, :] * t.tensor([[[1.0, 1.0, 0.0]]]).to(mujoco_qpos.device) + first_frame_position

        mujoco_qpos[:, :, 3: 7] = matrix_to_quaternion(rot_matrix)
        mujoco_qpos[:, :, :3] = root_positions
        return mujoco_qpos

def force_generate_and_trim(agent: PingPongMotionAgent, control: dict, controller_dt: float) -> None:
    with t.no_grad():
        agent.generate_new_frames(control, controller_dt, force_generation=True)
    n = agent.NUM_FRAMES_PER_TOKEN
    agent.frames["mujoco_qpos"] = agent.frames["mujoco_qpos"][:, n:]
    agent.frames["model_features"] = agent.frames["model_features"][:, n:]
    agent._current_frame_idx = 0


def _context_tensor_from_qpos(context_qpos: np.ndarray, num_frames: int) -> t.Tensor:
    context_qpos = np.asarray(context_qpos, dtype=np.float32)
    if context_qpos.shape[0] >= num_frames:
        context_qpos = context_qpos[-num_frames:]
    else:
        pad = np.repeat(context_qpos[:1], num_frames - context_qpos.shape[0], axis=0)
        context_qpos = np.concatenate([pad, context_qpos], axis=0)
    return t.from_numpy(context_qpos.copy()).view(1, num_frames, context_qpos.shape[-1])


def generate_hit_segment_from_context(demo, context_qpos: np.ndarray, controller_dt: float) -> np.ndarray:
    context_qpos = np.asarray(context_qpos, dtype=np.float32)
    qpos = context_qpos[-1]
    demo.mj_data.qpos[:] = qpos
    mujoco.mj_forward(demo.mj_model, demo.mj_data)
    control = demo.controller.sample_hit_control(qpos.copy())
    control["context_mujoco_qpos"] = _context_tensor_from_qpos(context_qpos, demo.agent.NUM_FRAMES_PER_TOKEN)
    force_generate_and_trim(demo.agent, control, controller_dt)
    return demo.agent.frames["mujoco_qpos"][0].detach().cpu().numpy().astype(np.float32)


def generate_recovery_segment_from_context(demo, context_qpos: np.ndarray, controller_dt: float) -> np.ndarray:
    context_qpos = np.asarray(context_qpos, dtype=np.float32)
    root_target = context_qpos[-1, :3].copy()
    control = demo.controller.recovery_control(root_target)
    control["context_mujoco_qpos"] = _context_tensor_from_qpos(context_qpos, demo.agent.NUM_FRAMES_PER_TOKEN)
    force_generate_and_trim(demo.agent, control, controller_dt)
    return demo.agent.frames["mujoco_qpos"][0].detach().cpu().numpy().astype(np.float32)
