"""
二维伊辛格模型实现
"""
import numpy as np
from tqdm import tqdm
import time


class IsingLattice:
    """二维伊辛格模型格子"""
    
    def __init__(self, L, J=1.0, H=0.0, seed=None):
        """
        初始化伊辛格格子
        
        参数:
            L: 格子的线性尺寸（格子总大小为L×L）
            J: 耦合常数
            H: 外部磁场
            seed: 随机数种子
        """
        if seed is not None:
            np.random.seed(seed)
            
        self.L = L
        self.J = J
        self.H = H
        self.N = L * L  # 总格点数
        
        # 随机初始化格子状态（每个格点的自旋为+1或-1）
        self.lattice = np.random.choice([-1, 1], size=(L, L))
        
        # 初始能量和磁化强度
        self._energy = self.calculate_total_energy()
        self._magnetization = self.calculate_magnetization()
        
    def reset(self, random_state=True):
        """
        重置格子状态
        
        参数:
            random_state: 如果为True，随机初始化；否则所有自旋设为+1
        """
        if random_state:
            self.lattice = np.random.choice([-1, 1], size=(self.L, self.L))
        else:
            self.lattice = np.ones((self.L, self.L))
            
        # 重新计算能量和磁化强度
        self._energy = self.calculate_total_energy()
        self._magnetization = self.calculate_magnetization()
    
    def get_neighbors(self, i, j):
        """
        获取格点(i,j)的所有邻居（考虑周期性边界条件）
        
        参数:
            i, j: 格点坐标
            
        返回:
            邻居格点的坐标列表
        """
        neighbors = [
            ((i + 1) % self.L, j),
            ((i - 1) % self.L, j),
            (i, (j + 1) % self.L),
            (i, (j - 1) % self.L)
        ]
        return neighbors
    
    def energy_difference(self, i, j):
        """
        计算翻转格点(i,j)的自旋导致的能量变化
        
        参数:
            i, j: 格点坐标
            
        返回:
            能量变化ΔE
        """
        s = self.lattice[i, j]
        
        # 计算与邻居的相互作用
        neighbors_sum = 0
        for ni, nj in self.get_neighbors(i, j):
            neighbors_sum += self.lattice[ni, nj]
        
        # 能量变化 ΔE = 2*J*s*sum(s_neighbors) + 2*H*s
        dE = 2 * self.J * s * neighbors_sum + 2 * self.H * s
        
        return dE
    
    def calculate_total_energy(self):
        """
        计算整个系统的总能量
        
        返回:
            总能量
        """
        energy = 0.0
        
        # 计算相邻自旋的相互作用
        for i in range(self.L):
            for j in range(self.L):
                s = self.lattice[i, j]
                
                # 只计算每个键一次（向右和向下的相互作用）
                right_neighbor = self.lattice[i, (j + 1) % self.L]
                down_neighbor = self.lattice[(i + 1) % self.L, j]
                
                energy -= self.J * s * (right_neighbor + down_neighbor)
                
                # 磁场贡献
                energy -= self.H * s
        
        return energy
    
    def calculate_magnetization(self):
        """
        计算系统的总磁化强度
        
        返回:
            总磁化强度
        """
        return np.sum(self.lattice)
    
    def metropolis_step(self, temperature):
        """
        执行一次Metropolis蒙特卡洛步骤
        
        参数:
            temperature: 温度T
            
        返回:
            是否接受了翻转
        """
        # 随机选择一个格点
        i, j = np.random.randint(0, self.L, size=2)
        
        # 计算能量变化
        dE = self.energy_difference(i, j)
        
        # Metropolis判据
        if dE <= 0 or np.random.random() < np.exp(-dE / temperature):
            self.lattice[i, j] *= -1  # 翻转自旋
            
            # 更新能量和磁化强度
            self._energy += dE
            self._magnetization -= 2 * self.lattice[i, j]  # 翻转后的值
            
            return True
        
        return False
    
    def monte_carlo_sweep(self, temperature):
        """
        执行一次蒙特卡洛扫描（N次metropolis_step，N为格点总数）
        
        参数:
            temperature: 温度T
            
        返回:
            接受翻转的次数
        """
        accepted = 0
        for _ in range(self.N):
            if self.metropolis_step(temperature):
                accepted += 1
        
        return accepted
    
    def run_simulation(self, temperature, thermalization_steps, measurement_steps, 
                       measurement_interval=10, task_id=None, verbose=True):
        """
        运行完整的蒙特卡洛模拟
        
        参数:
            temperature: 温度T
            thermalization_steps: 热化步数
            measurement_steps: 测量步数
            measurement_interval: 测量间隔
            task_id: 任务ID（用于详细输出）
            verbose: 是否显示进度条
            
        返回:
            (能量列表, 磁化强度列表)
        """
        task_prefix = f"[任务 {task_id}] " if task_id is not None else ""
        
        # 热化阶段
        if verbose:
            print(f"{task_prefix}热化阶段 (T={temperature:.4f})...")
        
        iterator = range(thermalization_steps)
        if verbose:
            iterator = tqdm(iterator, desc=f"{task_prefix}热化", 
                          unit="步", ncols=100, position=0, leave=True)
            
        therm_start = time.time()
        accept_rate_hist = []
        
        for step in iterator:
            accepted = self.monte_carlo_sweep(temperature)
            accept_rate = accepted / self.N
            accept_rate_hist.append(accept_rate)
            
            # 每100步更新一次额外信息
            if verbose and step % 100 == 0:
                avg_rate = np.mean(accept_rate_hist[-100:]) if accept_rate_hist else 0
                energy_per_site = self._energy / self.N
                mag_per_site = abs(self._magnetization) / self.N
                
                iterator.set_postfix({
                    "接受率": f"{avg_rate:.3f}",
                    "E/N": f"{energy_per_site:.4f}",
                    "M/N": f"{mag_per_site:.4f}"
                })
        
        therm_time = time.time() - therm_start
        if verbose:
            print(f"{task_prefix}热化完成，耗时 {therm_time:.2f} 秒，" +
                 f"平均接受率: {np.mean(accept_rate_hist):.4f}")
            
        # 测量阶段
        energies = []
        magnetizations = []
        
        if verbose:
            print(f"{task_prefix}测量阶段 (T={temperature:.4f})...")
        
        iterator = range(measurement_steps)
        if verbose:
            iterator = tqdm(iterator, desc=f"{task_prefix}测量", 
                          unit="步", ncols=100, position=0, leave=True)
            
        measure_start = time.time()
        accept_rate_hist = []
        
        for step in iterator:
            accepted = self.monte_carlo_sweep(temperature)
            accept_rate = accepted / self.N
            accept_rate_hist.append(accept_rate)
            
            # 按指定间隔进行测量
            if step % measurement_interval == 0:
                energies.append(self._energy)
                magnetizations.append(self._magnetization)
            
            # 每100步更新一次额外信息
            if verbose and step % 100 == 0:
                avg_rate = np.mean(accept_rate_hist[-100:]) if accept_rate_hist else 0
                energy_per_site = self._energy / self.N
                mag_per_site = abs(self._magnetization) / self.N
                
                iterator.set_postfix({
                    "接受率": f"{avg_rate:.3f}",
                    "E/N": f"{energy_per_site:.4f}",
                    "M/N": f"{mag_per_site:.4f}",
                    "测量数": f"{len(energies)}"
                })
                
        measure_time = time.time() - measure_start
        if verbose:
            print(f"{task_prefix}测量完成，耗时 {measure_time:.2f} 秒，" +
                 f"平均接受率: {np.mean(accept_rate_hist):.4f}，共采集 {len(energies)} 个数据点")
        
        return np.array(energies), np.array(magnetizations) 