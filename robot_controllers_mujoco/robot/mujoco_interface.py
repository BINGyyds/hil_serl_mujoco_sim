import logging
from typing import Dict, Optional, Tuple, Union

import mujoco
import mujoco.viewer
import numpy as np
from mujoco import MjvCamera

from robot_controllers_mujoco.robot.config import ProjectConfig

logger = logging.getLogger(__name__)


class MujocoInterface:
    """封装 Panda MuJoCo 仿真的底层交互接口。"""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        try:
            self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(str(config.MJCF_PATH))
            if config.TIMESTEP is not None:
                self.model.opt.timestep = config.TIMESTEP

            self.data: mujoco.MjData = mujoco.MjData(self.model)
            logger.info("MuJoCo 模型加载成功: %s", config.MJCF_PATH)
            logger.info("仿真时间步长 (dt) 设置为: %s", self.model.opt.timestep)
        except Exception as exc:
            logger.exception("加载 MuJoCo 模型失败: %s", config.MJCF_PATH)
            raise exc

        self.viewer: Optional[mujoco.viewer.Handle] = None
        self.renderer: Optional[mujoco.Renderer] = None

        self._get_ids_from_config(config)

        self.camera = MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)

    def _get_ids_from_config(self, config: ProjectConfig) -> None:
        """根据配置缓存 joint、actuator 和 site 的 ID。"""
        self.arm_joint_names = config.MUJOCO_ARM_JOINT_NAMES
        self.arm_joint_ids = np.array(
            [self.model.joint(name).id for name in self.arm_joint_names], dtype=np.int32
        )
        self.arm_qpos_addrs = self.model.jnt_qposadr[self.arm_joint_ids]
        self.arm_dof_addrs = self.model.jnt_dofadr[self.arm_joint_ids]

        self.arm_actuator_names = config.MUJOCO_ARM_ACTUATOR_NAMES
        self.arm_actuator_ids = np.array(
            [self.model.actuator(name).id for name in self.arm_actuator_names],
            dtype=np.int32,
        )

        self.base_site_id = (
            self.model.site(config.MUJOCO_BASE_SITE_NAME).id
            if config.MUJOCO_BASE_SITE_NAME
            else None
        )
        self.end_effector_site_id = self.model.site(
            config.MUJOCO_END_EFFECTOR_SITE_NAME
        ).id

    def reset_to_keyframe(self, key_name: str) -> None:
        """通过 MJCF 中定义的 keyframe 重置仿真状态。"""
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        if key_id < 0:
            raise ValueError(
                f"MuJoCo 模型中不存在名为 '{key_name}' 的 keyframe。"
            )

        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)
        logger.info("机械臂已通过 keyframe '%s' 重置。", key_name)

    def reset_to_home_keyframe(self) -> None:
        """将机械臂重置到 MJCF 中定义的 home keyframe。"""
        self.reset_to_keyframe("home")

    def get_arm_joint_angles(self) -> np.ndarray:
        """读取机械臂当前关节角度。"""
        return self.data.qpos[self.arm_qpos_addrs].copy()

    def get_arm_joint_velocities(self) -> np.ndarray:
        """读取机械臂当前关节速度。"""
        return self.data.qvel[self.arm_dof_addrs].copy()

    def get_joint_torques(self) -> np.ndarray:
        """读取 Panda 7 个关节执行器当前输出。"""
        return self.data.actuator_force[self.arm_actuator_ids].copy()

    def get_end_effector_position(self) -> np.ndarray:
        """获取末端执行器 site 的当前世界坐标。"""
        return self.data.site_xpos[self.end_effector_site_id].copy()

    def set_control(self, ctrl: np.ndarray) -> None:
        """设置 Panda 7 个关节的控制目标。"""
        self.data.ctrl[:] = ctrl

    def step(self) -> None:
        """执行一步物理仿真。"""
        mujoco.mj_step(self.model, self.data)

    def launch_viewer(self) -> None:
        """启动被动式查看器。"""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def sync_viewer(self) -> None:
        """与被动查看器同步。"""
        if self.viewer and self.viewer.is_running():
            self.viewer.sync()

    def close_viewer(self) -> None:
        """关闭查看器窗口。"""
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def setup_renderer(self, width: int = 1280, height: int = 720) -> mujoco.Renderer:
        """初始化离线渲染器，并返回可用的 renderer 实例。"""
        if self.renderer is None:
            max_width = int(self.model.vis.global_.offwidth)
            max_height = int(self.model.vis.global_.offheight)
            render_width = min(width, max_width)
            render_height = min(height, max_height)
            if (render_width, render_height) != (width, height):
                logger.warning(
                    "请求的离屏渲染分辨率 %sx%s 超出 MJCF framebuffer %sx%s，已自动调整为 %sx%s。",
                    width,
                    height,
                    max_width,
                    max_height,
                    render_width,
                    render_height,
                )
            self.renderer = mujoco.Renderer(self.model, render_height, render_width)
            logger.info("离线渲染器已初始化，分辨率: %sx%s", render_width, render_height)
        return self.renderer

    def set_camera_view(
        self,
        azimuth: float,
        elevation: float,
        distance: float,
        lookat: Optional[np.ndarray] = None,
    ) -> None:
        """更新离线渲染相机视角。"""
        self.camera.azimuth = azimuth
        self.camera.elevation = elevation
        self.camera.distance = distance
        if lookat is not None:
            self.camera.lookat[:] = lookat

    def render_frame(self, camera: Union[str, mujoco.MjvCamera]) -> np.ndarray:
        """从指定相机渲染一帧 RGB 图像。"""
        renderer = self.setup_renderer()
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()

    def render_all_custom_frames(self, camera_names: Tuple[str, ...]) -> Dict[str, np.ndarray]:
        """从多个命名相机渲染图像。"""
        frames: Dict[str, np.ndarray] = {}

        for cam_name in camera_names:
            try:
                frames[cam_name] = self.render_frame(camera=cam_name)
            except mujoco.FatalError:
                logger.warning("相机 '%s' 渲染失败，已跳过。", cam_name)
        return frames
