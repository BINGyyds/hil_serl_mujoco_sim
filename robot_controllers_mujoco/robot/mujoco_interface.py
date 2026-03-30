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
        """根据配置缓存 actuator、site、body 和 sensor 的索引信息。"""
        self.arm_joint_names = config.MUJOCO_ARM_JOINT_NAMES
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
        self.attachment_body_id = self.model.body("attachment").id

        self.arm_joint_pos_sensor_slices = tuple(
            self._get_sensor_slice(name, expected_dim=1)
            for name in config.MUJOCO_ARM_JOINT_POS_SENSOR_NAMES
        )
        self.arm_joint_vel_sensor_slices = tuple(
            self._get_sensor_slice(name, expected_dim=1)
            for name in config.MUJOCO_ARM_JOINT_VEL_SENSOR_NAMES
        )
        self.arm_joint_torque_sensor_slices = tuple(
            self._get_sensor_slice(name, expected_dim=1)
            for name in config.MUJOCO_ARM_JOINT_TORQUE_SENSOR_NAMES
        )
        self.end_effector_pos_sensor_slice = self._get_sensor_slice(
            config.MUJOCO_END_EFFECTOR_POS_SENSOR_NAME, expected_dim=3
        )
        self.attachment_pos_sensor_slice = self._get_sensor_slice(
            config.MUJOCO_ATTACHMENT_POS_SENSOR_NAME, expected_dim=3
        )
        self.attachment_quat_sensor_slice = self._get_sensor_slice(
            config.MUJOCO_ATTACHMENT_QUAT_SENSOR_NAME, expected_dim=4
        )

    def _get_sensor_slice(self, sensor_name: str, expected_dim: int) -> slice:
        """返回指定 sensor 在 sensordata 中的切片，并检查维度。"""
        sensor_id = self.model.sensor(sensor_name).id
        sensor_adr = int(self.model.sensor_adr[sensor_id])
        sensor_dim = int(self.model.sensor_dim[sensor_id])

        if sensor_dim != expected_dim:
            raise ValueError(
                f"Sensor '{sensor_name}' 维度为 {sensor_dim}，预期为 {expected_dim}。"
            )

        return slice(sensor_adr, sensor_adr + sensor_dim)

    def _read_sensor(self, sensor_slice: slice) -> np.ndarray:
        """读取一个 sensor 切片对应的数据。"""
        return np.asarray(self.data.sensordata[sensor_slice], dtype=np.float64).copy()

    def _read_scalar_sensor(self, sensor_slice: slice) -> float:
        """读取一个标量 sensor。"""
        return float(self.data.sensordata[sensor_slice][0])

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
        return np.asarray(
            [self._read_scalar_sensor(sensor_slice) for sensor_slice in self.arm_joint_pos_sensor_slices],
            dtype=np.float64,
        )

    def get_arm_joint_velocities(self) -> np.ndarray:
        """读取机械臂当前关节速度。"""
        return np.asarray(
            [self._read_scalar_sensor(sensor_slice) for sensor_slice in self.arm_joint_vel_sensor_slices],
            dtype=np.float64,
        )

    def get_joint_torques(self) -> np.ndarray:
        """读取 Panda 7 个关节执行器当前输出。"""
        return np.asarray(
            [self._read_scalar_sensor(sensor_slice) for sensor_slice in self.arm_joint_torque_sensor_slices],
            dtype=np.float64,
        )

    def get_end_effector_position(self) -> np.ndarray:
        """获取末端执行器 site 的当前世界坐标。"""
        return self._read_sensor(self.end_effector_pos_sensor_slice)

    def get_attachment_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取末端 attachment body 的当前世界位置和四元数姿态。

        Returns:
            position: 世界坐标系下的位置，形状为 (3,)。
            quaternion: 世界坐标系下的四元数，顺序为 (w, x, y, z)，形状为 (4,)。
        """
        position = self._read_sensor(self.attachment_pos_sensor_slice)
        quaternion = self._read_sensor(self.attachment_quat_sensor_slice)
        return position, quaternion

    def get_attachment_transform(self) -> np.ndarray:
        """获取末端 attachment body 的当前世界系 4x4 齐次变换矩阵。"""
        position, quaternion = self.get_attachment_pose()
        w, x, y, z = quaternion

        rotation = np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = position
        return transform

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
