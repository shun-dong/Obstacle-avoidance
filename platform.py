import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import math
import random
import numpy as np
from autocruise import AutoCruise

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

class CarController:
    def __init__(self):
        # 小车运动参数
        self.x = 0.0
        self.y = 0.0
        self.velocity = np.array([0.0, 0.0])  # 速度矢量
        self.max_speed = 3                 # 最大速度
        self.acceleration = 2               # 基础加速度
        self.trajectory = []

        # 避障参数
        self.obstacle_force_gain = 16.0        # 避障力增益
        self.recovery_distance = 2.0       # 避障生效距离
        self.safe_distance = 1.0              # 安全距离
        self.car_radius = 0.5                 # 小车半径

        # 雷达参数
        self.radar_max_distance = 50.0
        self.radar_angle_res = 1.0
        self.radar_noise_std = 0.05
        self.scan_points = []

        # 初始化界面
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.setup_plot()

        # 控制状态
        self.control_vector = np.array([0.0, 0.0])  # 控制输入方向
        self.active_keys = set()

        # 障碍物系统
        self.obstacles = []
        self.init_obstacles()
        self.draw_obstacles()

        # 速度恢复机制
        self.pre_avoidance_velocity = np.array([0.0, 0.0])
        self.avoidance_active = False

        # 图形对象
        self.car_circle = self.ax.add_patch(plt.Circle((0, 0), self.car_radius, color='red'))
        self.trajectory_line, = self.ax.plot([], [], 'b-', lw=1)
        self.info_text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes, va='top')
        self.scan_scatter = self.ax.scatter([], [], c='lime', s=8, marker='.', alpha=0.6)
        self.safe_zone = self.ax.add_patch(plt.Circle((0,0), self.safe_distance, fill=False,
                                                      linestyle='--', color='yellow', alpha=0.6))
        self.border_rect = plt.Rectangle((-10, -10), 20, 20, linewidth=1,
                                         edgecolor='red', linestyle='--', facecolor='none')
        self.ax.add_patch(self.border_rect)

        self.timer = self.fig.canvas.new_timer(interval=30)
        self.timer.add_callback(self.update_frame)
        self.timer.start()

    def setup_plot(self):
        """初始化绘图区域"""
        self.ax.set_title("全向自动驾驶小车 - 矢量控制版\nWASD: 移动  空格: 急停  Q/E: 旋转")
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')

    def on_key_press(self, event):
        """键盘按下事件"""
        self.active_keys.add(event.key)
        self.update_control_vector()

    def on_key_release(self, event):
        """键盘释放事件"""
        if event.key in self.active_keys:
            self.active_keys.remove(event.key)
        self.update_control_vector()

    def update_control_vector(self):
        """更新控制方向矢量"""
        vec = np.array([0.0, 0.0])
        if 'up' in self.active_keys: vec += [0, 1]
        if 'down' in self.active_keys: vec += [0, -1]
        if 'left' in self.active_keys: vec += [-1, 0]
        if 'right' in self.active_keys: vec += [1, 0]

        # 归一化控制方向
        if np.linalg.norm(vec) > 0:
            self.control_vector = vec / np.linalg.norm(vec)
        else:
            self.control_vector = vec

    def init_obstacles(self):
        """初始化6个动态障碍物"""
        for _ in range(6):
            angle = random.uniform(0, 2*math.pi)
            radius = random.uniform(4, 8)
            speed = random.uniform(0.3, 0.3)
            cx = radius * math.cos(angle)
            cy = radius * math.sin(angle)

            obstacle = {
                'circle': plt.Circle((cx, cy), 0.8, color='darkorange', alpha=0.8),
                'angle': angle,
                'radius': radius,
                'speed': speed
            }
            self.obstacles.append(obstacle)

    def draw_obstacles(self):
        """绘制障碍物"""
        for o in self.obstacles:
            self.ax.add_patch(o['circle'])

    def ray_intersect_circle(self, origin, theta, circle):
        """射线与圆求交"""
        dx, dy = math.cos(theta), math.sin(theta)
        cx, cy = circle.center
        ox, oy = origin[0]-cx, origin[1]-cy

        a = dx**2 + dy**2
        b = 2*(ox*dx + oy*dy)
        c = ox**2 + oy**2 - circle.radius**2

        delta = b**2 - 4*a*c
        if delta < 0: return None

        t1 = (-b - math.sqrt(delta))/(2*a)
        t2 = (-b + math.sqrt(delta))/(2*a)
        t = min([t for t in [t1, t2] if t >= 0], default=None)
        return (origin[0]+t*dx, origin[1]+t*dy) if t else None

    def scan_environment(self):
        """雷达扫描环境（不再检测边界）"""
        self.scan_points = []
        origin = (self.x, self.y)

        for angle in np.arange(0, 360, self.radar_angle_res):
            theta = math.radians(angle)
            closest = None
            min_dist = self.radar_max_distance

            # 仅检测障碍物
            for obstacle in self.obstacles:
                point = self.ray_intersect_circle(origin, theta, obstacle['circle'])
                if point:
                    dist = math.hypot(point[0]-origin[0], point[1]-origin[1])
                    if dist < min_dist:
                        min_dist = dist
                        noise = random.gauss(0, self.radar_noise_std)
                        closest = (point[0]+noise, point[1]+noise)

            if closest:
                self.scan_points.append(closest)


    def update_movement(self, dt):
        """更新运动状态（添加边界速度限制）"""
        # 控制加速度
        control_acc = self.control_vector * self.acceleration

        # 综合加速度
        total_acc = control_acc
        acc_mag = np.linalg.norm(total_acc)
        if acc_mag > 5.0:
            total_acc = total_acc * 5.0 / acc_mag

        # 更新速度
        self.velocity += total_acc * dt
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity * self.max_speed / speed

        # 边界速度调整（缓慢抵消法向速度）
        border_safe_distance = self.safe_distance
        # 左边界
        distance_to_left = self.x + 10
        if distance_to_left < border_safe_distance and self.velocity[0] < 0:
            factor = max(distance_to_left / border_safe_distance, 0.0)
            self.velocity[0] *= factor
        # 右边界
        distance_to_right = 10 - self.x
        if distance_to_right < border_safe_distance and self.velocity[0] > 0:
            factor = max(distance_to_right / border_safe_distance, 0.0)
            self.velocity[0] *= factor
        # 下边界
        distance_to_bottom = self.y + 10
        if distance_to_bottom < border_safe_distance and self.velocity[1] < 0:
            factor = max(distance_to_bottom / border_safe_distance, 0.0)
            self.velocity[1] *= factor
        # 上边界
        distance_to_top = 10 - self.y
        if distance_to_top < border_safe_distance and self.velocity[1] > 0:
            factor = max(distance_to_top / border_safe_distance, 0.0)
            self.velocity[1] *= factor

        # 更新位置
        self.x += self.velocity[0] * dt
        self.y += self.velocity[1] * dt

        # 边框碰撞限制
        self.x = np.clip(self.x, -10 + self.car_radius, 10 - self.car_radius)
        self.y = np.clip(self.y, -10 + self.car_radius, 10 - self.car_radius)

        self.trajectory.append((self.x, self.y))

    def check_collision(self):
        speed = np.linalg.norm(self.velocity)

        for obstacle in self.obstacles:
            dx = self.x - obstacle['circle'].center[0]
            dy = self.y - obstacle['circle'].center[1]
            if math.hypot(dx, dy) < (self.car_radius + obstacle['circle'].radius):
                return True
        return False

    def update_frame(self):
        """主更新循环"""
        # 更新障碍物位置
        for obstacle in self.obstacles:
            obstacle['angle'] += obstacle['speed'] * 0.03
            cx = obstacle['radius'] * math.cos(obstacle['angle'])
            cy = obstacle['radius'] * math.sin(obstacle['angle'])
            obstacle['circle'].center = (cx, cy)

        # 环境扫描与运动更新
        self.scan_environment()
        self.update_movement(dt=0.03)

        # 碰撞处理
        if self.check_collision():
            self.velocity = np.array([0.0, 0.0])
            self.info_text.set_text("! 碰撞 ! 速度已重置")
        else:
            self.info_text.set_text(f"速度: {np.linalg.norm(self.velocity):.1f}m/s\n"
                                    f"控制方向: ({self.control_vector[0]:.1f}, {self.control_vector[1]:.1f})")

        # 更新图形
        self.car_circle.center = (self.x, self.y)
        self.scan_scatter.set_offsets(self.scan_points)
        self.trajectory_line.set_data(*zip(*self.trajectory))
        self.safe_zone.center = (self.x, self.y)
        self.auto_scale_view()
        self.fig.canvas.draw_idle()

    def auto_scale_view(self):
        """自动视图缩放"""
        all_x = [self.x] + [p[0] for p in self.trajectory[-50:]] + [o['circle'].center[0] for o in self.obstacles]
        all_y = [self.y] + [p[1] for p in self.trajectory[-50:]] + [o['circle'].center[1] for o in self.obstacles]
        margin = 3.0
        self.ax.set_xlim(min(all_x)-margin, max(all_x)+margin)
        self.ax.set_ylim(min(all_y)-margin, max(all_y)+margin)

    def show(self):
        plt.show()

# 运行系统
controller = CarController()
controller.show()