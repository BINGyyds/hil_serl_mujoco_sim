import logging
from typing import Dict, Optional, Tuple, Union

import mujoco
import mujoco.viewer
import numpy as np
from mujoco import MjvCamera

from robot_controllers_mujoco.common.config import ProjectConfig
from robot_controllers_mujoco.simulation.site_ik_solver import (
    SiteIKOptions,
    solve_site_pose_ik,
)

logger = logging.getLogger(__name__)


class MujocoInterface:
    """封装 Panda MuJoCo 仿真的底层交互接口。"""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        model_config = config.model
        sim_config = config.sim
        try:
            if model_config.mjcf_path is None:
                raise ValueError("MJCF 路径尚未配置。")

            self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(
                str(model_config.mjcf_path)
            )
            if sim_config.timestep is not None:
                self.model.opt.timestep = sim_config.timestep

            self.data: mujoco.MjData = mujoco.MjData(self.model)
            self.ik_data: mujoco.MjData = mujoco.MjData(self.model)
            logger.info("MuJoCo 模型加载成功: %s", model_config.mjcf_path)
            logger.info("仿真时间步长 (dt) 设置为: %s", self.model.opt.timestep)
        except Exception as exc:
            logger.exception("加载 MuJoCo 模型失败: %s", model_config.mjcf_path)
            raise exc

        self.viewer: Optional[mujoco.viewer.Handle] = None
        self.renderer: Optional[mujoco.Renderer] = None
        self.renderers: Dict[Tuple[int, int], mujoco.Renderer] = {}

        self._get_ids_from_config(config)
        self.end_effector_wrench_bias_local = np.zeros(6, dtype=np.float64)

        self.camera = MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)

    def _get_ids_from_config(self, config: ProjectConfig) -> None:
        """根据配置缓存 actuator、site、body 和 sensor 的索引信息。"""
        model_config = config.model

        self.arm_joint_names = model_config.arm_joint_names
        self.arm_actuator_names = model_config.arm_actuator_names
        if len(self.arm_joint_names) != len(self.arm_actuator_names):
            raise ValueError("关节名与 actuator 名数量不一致，无法建立速度命令映射。")

        self.arm_joint_ids = np.array(
            [self.model.joint(name).id for name in self.arm_joint_names],
            dtype=np.int32,
        )
        self.arm_dof_addrs = np.asarray(
            self.model.jnt_dofadr[self.arm_joint_ids],
            dtype=np.int32,
        )
        self.arm_qpos_addrs = np.asarray(
            self.model.jnt_qposadr[self.arm_joint_ids],
            dtype=np.int32,
        )
        limited = self.model.jnt_limited[self.arm_joint_ids].astype(bool)
        self.arm_joint_lower_limits = np.where(
            limited,
            self.model.jnt_range[self.arm_joint_ids, 0],
            -np.inf,
        )
        self.arm_joint_upper_limits = np.where(
            limited,
            self.model.jnt_range[self.arm_joint_ids, 1],
            np.inf,
        )
        self.arm_actuator_ids = np.array(
            [self.model.actuator(name).id for name in self.arm_actuator_names],
            dtype=np.int32,
        )
        self.arm_joint_name_to_actuator_id = {
            joint_name: self.model.actuator(actuator_name).id
            for joint_name, actuator_name in zip(
                self.arm_joint_names,
                self.arm_actuator_names,
            )
        }

        self.base_site_id = (
            self.model.site(model_config.base_site_name).id
            if model_config.base_site_name
            else None
        )
        self.end_effector_site_id = self.model.site(
            model_config.end_effector_site_name
        ).id
        self.attachment_body_id = self.model.body("attachment").id

        self.arm_joint_pos_sensor_slices = tuple(
            self._get_sensor_slice(name, expected_dim=1)
            for name in model_config.arm_joint_pos_sensor_names
        )
        self.arm_joint_vel_sensor_slices = tuple(
            self._get_sensor_slice(name, expected_dim=1)
            for name in model_config.arm_joint_vel_sensor_names
        )
        self.arm_joint_torque_sensor_slices = tuple(
            self._get_sensor_slice(name, expected_dim=1)
            for name in model_config.arm_joint_torque_sensor_names
        )
        if model_config.end_effector_pos_sensor_name:
            self.end_effector_pos_sensor_slice = self._get_sensor_slice(
                model_config.end_effector_pos_sensor_name,
                expected_dim=3,
            )
        else:
            self.end_effector_pos_sensor_slice = None
        if model_config.end_effector_force_sensor_name:
            self.end_effector_force_sensor_slice = self._get_sensor_slice(
                model_config.end_effector_force_sensor_name,
                expected_dim=3,
            )
        else:
            self.end_effector_force_sensor_slice = None
        if model_config.end_effector_torque_sensor_name:
            self.end_effector_torque_sensor_slice = self._get_sensor_slice(
                model_config.end_effector_torque_sensor_name,
                expected_dim=3,
            )
        else:
            self.end_effector_torque_sensor_slice = None
        self.attachment_pos_sensor_slice = self._get_sensor_slice(
            model_config.attachment_pos_sensor_name, expected_dim=3
        )
        self.attachment_quat_sensor_slice = self._get_sensor_slice(
            model_config.attachment_quat_sensor_name, expected_dim=4
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

    def forward(self) -> None:
        """执行一次空 forward，刷新 site/body/sensor 等派生状态。"""
        mujoco.mj_forward(self.model, self.data)

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

    def get_arm_joint_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """读取机械臂当前关节角度和关节速度。"""
        return self.get_arm_joint_angles(), self.get_arm_joint_velocities()

    def get_joint_torques(self) -> np.ndarray:
        """读取 Panda 7 个关节执行器当前输出。"""
        return np.asarray(
            [self._read_scalar_sensor(sensor_slice) for sensor_slice in self.arm_joint_torque_sensor_slices],
            dtype=np.float64,
        )

    def has_end_effector_wrench_sensors(self) -> bool:
        """返回末端 6D force/torque sensor 是否可用。"""
        return (
            self.end_effector_force_sensor_slice is not None
            and self.end_effector_torque_sensor_slice is not None
        )

    def get_end_effector_wrench_local_raw(self) -> np.ndarray:
        """读取末端 site 本体坐标系下的原始 6D wrench。"""
        if not self.has_end_effector_wrench_sensors():
            raise RuntimeError("当前模型未配置末端 force/torque sensors。")

        force_local = self._read_sensor(self.end_effector_force_sensor_slice)
        torque_local = self._read_sensor(self.end_effector_torque_sensor_slice)
        return np.concatenate([force_local, torque_local])

    def tare_end_effector_wrench(self) -> None:
        """将当前末端本体系 6D wrench 设为零偏。"""
        if not self.has_end_effector_wrench_sensors():
            self.end_effector_wrench_bias_local[:] = 0.0
            return
        self.end_effector_wrench_bias_local = self.get_end_effector_wrench_local_raw()

    def get_end_effector_wrench_local(self) -> np.ndarray:
        """读取减去 tare 偏置后的末端本体系 6D wrench。"""
        wrench_local_raw = self.get_end_effector_wrench_local_raw()
        return wrench_local_raw - self.end_effector_wrench_bias_local

    def get_end_effector_position(self) -> np.ndarray:
        """获取末端执行器 site 的当前世界坐标。"""
        return np.asarray(
            self.data.site_xpos[self.end_effector_site_id],
            dtype=np.float64,
        ).copy()

    def get_site_transform(
        self,
        site_name: Optional[str] = None,
    ) -> np.ndarray:
        """获取任意 site 的当前世界系 4x4 齐次变换矩阵。"""
        site_id = self._get_site_id(site_name)
        return self._get_site_transform_from_data(self.data, site_id)

    def _get_site_id(self, site_name: Optional[str] = None) -> int:
        if site_name is None:
            return int(self.end_effector_site_id)
        return int(self.model.site(site_name).id)

    def _get_site_transform_from_data(
        self,
        data: mujoco.MjData,
        site_id: int,
    ) -> np.ndarray:
        position = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
        rotation = np.asarray(
            data.site_xmat[site_id],
            dtype=np.float64,
        ).reshape(3, 3).copy()

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = position
        return transform

    def get_end_effector_transform(self) -> np.ndarray:
        """获取配置中末端 site 的当前世界系 4x4 齐次变换矩阵。"""
        return self.get_site_transform()

    def get_end_effector_wrench_world(self) -> np.ndarray:
        """读取减去 tare 偏置后的末端世界系 6D wrench。"""
        wrench_local = self.get_end_effector_wrench_local()
        rotation = self.get_end_effector_transform()[:3, :3]
        force_world = rotation @ wrench_local[:3]
        torque_world = rotation @ wrench_local[3:]
        return np.concatenate([force_world, torque_world])

    def get_site_world_jacobian(
        self,
        site_name: Optional[str] = None,
    ) -> np.ndarray:
        """返回指定 site 在世界系下的 6x7 Jacobian。"""
        site_id = self._get_site_id(site_name)
        return self._get_site_world_jacobian_from_data(self.data, site_id)

    def _get_site_world_jacobian_from_data(
        self,
        data: mujoco.MjData,
        site_id: int,
    ) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, data, jacp, jacr, site_id)
        return np.vstack(
            [
                jacp[:, self.arm_dof_addrs],
                jacr[:, self.arm_dof_addrs],
            ]
        )

    def get_site_local_jacobian(
        self,
        site_name: Optional[str] = None,
    ) -> np.ndarray:
        """返回指定 site 在自身 LOCAL 坐标系下的 6x7 Jacobian。"""
        site_id = self._get_site_id(site_name)
        transform = self._get_site_transform_from_data(self.data, site_id)
        jacobian_world = self._get_site_world_jacobian_from_data(self.data, site_id)
        world_to_local = np.block(
            [
                [transform[:3, :3].T, np.zeros((3, 3), dtype=np.float64)],
                [np.zeros((3, 3), dtype=np.float64), transform[:3, :3].T],
            ]
        )
        return world_to_local @ jacobian_world

    def get_end_effector_local_jacobian(self) -> np.ndarray:
        """返回配置中末端 site 在自身 LOCAL 坐标系下的 6x7 Jacobian。"""
        return self.get_site_local_jacobian()

    def inverse_kinematics_site(
        self,
        target_transform: np.ndarray,
        q_init: np.ndarray,
        site_name: Optional[str] = None,
        tolerance: float = 1e-4,
        max_iterations: int = 200,
        damping: float = 1e-4,
        step_scale: float = 0.6,
        max_step_norm: float = 0.2,
    ) -> np.ndarray:
        """用共享的有界非线性最小二乘后端求解 site IK。

        ``damping``、``step_scale`` 和 ``max_step_norm`` 是旧 DLS 接口参数，
        这里保留签名以兼容轨迹规划调用，但主求解逻辑由
        :mod:`robot_controllers_mujoco.simulation.site_ik_solver` 负责。
        """
        site_id = self._get_site_id(site_name)
        finite_limits = np.isfinite(self.arm_joint_lower_limits) & np.isfinite(
            self.arm_joint_upper_limits
        )
        q_center = np.asarray(q_init, dtype=np.float64).reshape(
            (len(self.arm_joint_ids),)
        ).copy()
        q_center[finite_limits] = 0.5 * (
            self.arm_joint_lower_limits[finite_limits]
            + self.arm_joint_upper_limits[finite_limits]
        )

        self.ik_data.qpos[:] = self.data.qpos
        self.ik_data.qvel[:] = self.data.qvel
        result = solve_site_pose_ik(
            model=self.model,
            data=self.ik_data,
            target_transform=target_transform,
            q_seed=q_init,
            qpos_addrs=self.arm_qpos_addrs,
            dof_addrs=self.arm_dof_addrs,
            q_low=self.arm_joint_lower_limits,
            q_high=self.arm_joint_upper_limits,
            site_id=site_id,
            options=SiteIKOptions(
                max_iterations=max_iterations,
                tolerance=tolerance,
                position_weight=1.0,
                rotation_weight=1.0,
                seed_weight=0.0,
                center_weight=0.0,
            ),
            extra_seeds=[q_center],
        )
        return result.q.copy()

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
        control = np.asarray(ctrl, dtype=np.float64)
        if control.shape != (len(self.arm_actuator_ids),):
            raise ValueError(
                f"控制输入维度错误，期望 {(len(self.arm_actuator_ids),)}，实际 {control.shape}"
            )
        self.data.ctrl[self.arm_actuator_ids] = control

    def set_named_joint_velocity_command(
        self,
        joint_velocity_by_name: Dict[str, float],
    ) -> None:
        """按关节名写入速度控制命令。"""
        unknown_joint_names = sorted(
            set(joint_velocity_by_name) - set(self.arm_joint_name_to_actuator_id)
        )
        if unknown_joint_names:
            raise ValueError(
                f"未知关节名，无法下发速度命令: {unknown_joint_names}"
            )

        for joint_name, joint_velocity in joint_velocity_by_name.items():
            actuator_id = self.arm_joint_name_to_actuator_id[joint_name]
            self.data.ctrl[actuator_id] = float(joint_velocity)

    def step(self) -> None:
        """执行一步物理仿真。"""
        if self.config.sim.enable_gravity_compensation:
            mujoco.mj_step1(self.model, self.data)
            self.data.qfrc_applied[:] = 0.0
            self.data.qfrc_applied[self.arm_dof_addrs] = self.data.qfrc_bias[
                self.arm_dof_addrs
            ]
            mujoco.mj_step2(self.model, self.data)
            return

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

    def has_camera(self, camera_name: str) -> bool:
        """返回模型中是否存在指定命名相机。"""
        return (
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                camera_name,
            )
            >= 0
        )

    def setup_renderer(self, width: int = 1280, height: int = 720) -> mujoco.Renderer:
        """初始化离线渲染器，并返回可用的 renderer 实例。"""
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

        renderer_key = (render_width, render_height)
        renderer = self.renderers.get(renderer_key)
        if renderer is None:
            renderer = mujoco.Renderer(self.model, render_height, render_width)
            self.renderers[renderer_key] = renderer
            if self.renderer is None:
                self.renderer = renderer
            logger.info("离线渲染器已初始化，分辨率: %sx%s", render_width, render_height)
        return renderer

    def close_renderers(self) -> None:
        """关闭所有离屏渲染器。"""
        for renderer in self.renderers.values():
            try:
                renderer.close()
            except Exception:
                logger.exception("关闭离屏渲染器失败。")
        self.renderers = {}
        self.renderer = None

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

    def render_frame(
        self,
        camera: Union[str, mujoco.MjvCamera],
        width: int = 1280,
        height: int = 720,
    ) -> np.ndarray:
        """从指定相机渲染一帧 RGB 图像。"""
        if isinstance(camera, str) and not self.has_camera(camera):
            raise ValueError(f"MuJoCo 模型中不存在名为 '{camera}' 的相机。")
        renderer = self.setup_renderer(width=width, height=height)
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()

    def render_all_custom_frames(
        self,
        camera_names: Tuple[str, ...],
        width: int = 1280,
        height: int = 720,
    ) -> Dict[str, np.ndarray]:
        """从多个命名相机渲染图像。"""
        frames: Dict[str, np.ndarray] = {}

        for cam_name in camera_names:
            try:
                frames[cam_name] = self.render_frame(
                    camera=cam_name,
                    width=width,
                    height=height,
                )
            except mujoco.FatalError:
                logger.warning("相机 '%s' 渲染失败，已跳过。", cam_name)
        return frames
