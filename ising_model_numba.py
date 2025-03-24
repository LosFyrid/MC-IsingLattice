"""
二维伊辛格模型实现 - Numba加速版本
"""
import numpy as np
from tqdm import tqdm
import time
import numba as nb


# 使用Numba JIT编译的函数用于计算邻居
@nb.njit
def get_neighbors(i, j, L):
    """获取格点(i,j)的所有邻居（考虑周期性边界条件）"""
    neighbors = np.array([
        [(i + 1) % L, j],
        [(i - 1) % L, j],
        [i, (j + 1) % L],
        [i, (j - 1) % L]
    ], dtype=np.int64)
    return neighbors


# 计算能量差
@nb.njit
def calculate_energy_difference(lattice, i, j, J, H, L):
    """计算翻转格点(i,j)的自旋导致的能量变化"""
    s = lattice[i, j]
    
    # 计算与邻居的相互作用
    neighbors_sum = 0
    neighbors = get_neighbors(i, j, L)
    for k in range(4):
        ni, nj = neighbors[k]
        neighbors_sum += lattice[ni, nj]
    
    # 能量变化 ΔE = 2*J*s*sum(s_neighbors) + 2*H*s
    dE = 2 * J * s * neighbors_sum + 2 * H * s
    
    return dE


# 计算总能量
@nb.njit
def calculate_total_energy(lattice, J, H, L):
    """计算整个系统的总能量"""
    energy = 0.0
    
    # 计算相邻自旋的相互作用
    for i in range(L):
        for j in range(L):
            s = lattice[i, j]
            
            # 只计算每个键一次（向右和向下的相互作用）
            right_neighbor = lattice[i, (j + 1) % L]
            down_neighbor = lattice[(i + 1) % L, j]
            
            energy -= J * s * (right_neighbor + down_neighbor)
            
            # 磁场贡献
            energy -= H * s
    
    return energy


# 计算总磁化强度
@nb.njit
def calculate_magnetization(lattice):
    """计算系统的总磁化强度"""
    return np.sum(lattice)


# 执行单步Metropolis
@nb.njit
def metropolis_step(lattice, temperature, energy, magnetization, J, H, L, random_state):
    """执行一次Metropolis蒙特卡洛步骤"""
    # 随机选择一个格点
    i = random_state[0] % L
    j = random_state[1] % L
    
    # 更新随机状态（简单的线性同余生成器）
    random_state[0] = (1664525 * random_state[0] + 1013904223) % (2**31-1)
    random_state[1] = (1664525 * random_state[1] + 1013904223) % (2**31-1)
    
    # 计算能量变化
    dE = calculate_energy_difference(lattice, i, j, J, H, L)
    
    # Metropolis判据
    rand = (random_state[2] % 1000000) / 1000000.0
    random_state[2] = (1664525 * random_state[2] + 1013904223) % (2**31-1)
    
    if dE <= 0 or rand < np.exp(-dE / temperature):
        # 翻转自旋
        lattice[i, j] *= -1
        
        # 更新能量和磁化强度
        energy += dE
        magnetization -= 2 * lattice[i, j]  # 翻转后的值
        
        return True, energy, magnetization
    
    return False, energy, magnetization


# 执行一次Monte Carlo扫描
@nb.njit
def monte_carlo_sweep(lattice, temperature, energy, magnetization, J, H, L, random_state):
    """执行一次蒙特卡洛扫描（N次metropolis_step，N为格点总数）"""
    N = L * L
    accepted = 0
    
    for _ in range(N):
        accepted_step, energy, magnetization = metropolis_step(
            lattice, temperature, energy, magnetization, J, H, L, random_state)
        
        if accepted_step:
            accepted += 1
    
    return accepted, energy, magnetization


# 批量Monte Carlo步骤，用于热化阶段
@nb.njit
def batch_monte_carlo_sweeps(lattice, temperature, energy, magnetization, J, H, L, 
                            n_sweeps, random_state):
    """执行多次蒙特卡洛扫描"""
    accept_rates = np.zeros(n_sweeps, dtype=np.float64)
    
    for i in range(n_sweeps):
        accepted, energy, magnetization = monte_carlo_sweep(
            lattice, temperature, energy, magnetization, J, H, L, random_state)
        
        accept_rates[i] = accepted / (L * L)
        
    return accept_rates, energy, magnetization


# 批量蒙特卡洛步骤，用于测量阶段，并记录能量和磁化强度
@nb.njit
def batch_monte_carlo_measurements(lattice, temperature, energy, magnetization, J, H, L, 
                                 n_sweeps, measurement_interval, random_state):
    """执行多次蒙特卡洛扫描并记录测量结果"""
    N = L * L
    n_measurements = n_sweeps // measurement_interval + 1
    
    energies = np.zeros(n_measurements, dtype=np.float64)
    magnetizations = np.zeros(n_measurements, dtype=np.float64)
    accept_rates = np.zeros(n_sweeps, dtype=np.float64)
    
    measurement_idx = 0
    energies[0] = energy
    magnetizations[0] = magnetization
    measurement_idx += 1
    
    for i in range(n_sweeps):
        accepted, energy, magnetization = monte_carlo_sweep(
            lattice, temperature, energy, magnetization, J, H, L, random_state)
        
        accept_rates[i] = accepted / N
        
        if (i + 1) % measurement_interval == 0 and measurement_idx < n_measurements:
            energies[measurement_idx] = energy
            magnetizations[measurement_idx] = magnetization
            measurement_idx += 1
    
    return accept_rates, energies[:measurement_idx], magnetizations[:measurement_idx], energy, magnetization


class IsingLatticeNumba:
    """二维伊辛格模型格子 - Numba加速版本"""
    
    def __init__(self, L, J=1.0, H=0.0, seed=None):
        """
        初始化伊辛格格子
        
        参数:
            L: 格子的线性尺寸（格子总大小为L×L）
            J: 耦合常数
            H: 外部磁场
            seed: 随机数种子
        """
        if seed is None:
            seed = np.random.randint(0, 2**31-1)
            
        self.L = L
        self.J = J
        self.H = H
        self.N = L * L  # 总格点数
        
        # 初始化随机数状态
        self.random_state = np.array([seed, (seed + 1) % (2**31-1), (seed + 2) % (2**31-1)], dtype=np.int64)
        
        # 随机初始化格子状态（每个格点的自旋为+1或-1）
        self.lattice = np.random.choice([-1, 1], size=(L, L))
        
        # 初始能量和磁化强度
        self._energy = calculate_total_energy(self.lattice, self.J, self.H, self.L)
        self._magnetization = calculate_magnetization(self.lattice)
    
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
        self._energy = calculate_total_energy(self.lattice, self.J, self.H, self.L)
        self._magnetization = calculate_magnetization(self.lattice)
    
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
            print(f"{task_prefix}热化阶段开始 (T={temperature:.4f}, J={self.J:.2f}, H={self.H:.2f})...")
            print(f"{task_prefix}格子大小: {self.L}x{self.L} = {self.N}个格点")
            print(f"{task_prefix}热化步数: {thermalization_steps}，每步包含{self.N}次Metropolis尝试")
        
        # 复制格子和能量/磁化强度，以便传给Numba函数
        lattice_copy = self.lattice.copy()
        energy = self._energy
        magnetization = self._magnetization
        random_state = self.random_state.copy()
        
        therm_start = time.time()
        
        # 将热化阶段分成多个批次，以便显示进度
        batch_size = 100
        n_batches = max(1, thermalization_steps // batch_size)
        last_batch_size = thermalization_steps % batch_size if thermalization_steps % batch_size != 0 else batch_size
        
        all_accept_rates = []
        
        if verbose:
            iterator = tqdm(range(n_batches), desc=f"{task_prefix}热化", 
                           unit="批", ncols=100, position=0, leave=True,
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        else:
            iterator = range(n_batches)
            
        for batch in iterator:
            sweeps_in_batch = last_batch_size if batch == n_batches - 1 else batch_size
            
            accept_rates, energy, magnetization = batch_monte_carlo_sweeps(
                lattice_copy, temperature, energy, magnetization, 
                self.J, self.H, self.L, sweeps_in_batch, random_state
            )
            
            all_accept_rates.extend(accept_rates)
            
            # 更新进度条
            if verbose:
                avg_rate = np.mean(accept_rates)
                energy_per_site = energy / self.N
                mag_per_site = abs(magnetization) / self.N
                total_steps = (batch + 1) * batch_size * self.N
                
                iterator.set_postfix({
                    "接受率": f"{avg_rate:.3f}",
                    "E/N": f"{energy_per_site:.4f}",
                    "M/N": f"{mag_per_site:.4f}",
                    "总尝试": f"{total_steps:,}"
                })
        
        therm_time = time.time() - therm_start
        
        # 更新实例变量
        self.lattice = lattice_copy
        self._energy = energy
        self._magnetization = magnetization
        self.random_state = random_state
        
        if verbose:
            avg_accept = np.mean(all_accept_rates)
            energy_per_site = energy / self.N
            mag_per_site = abs(magnetization) / self.N
            print(f"{task_prefix}热化完成，耗时 {therm_time:.2f} 秒，平均接受率: {avg_accept:.4f}")
            print(f"{task_prefix}当前状态: E/N = {energy_per_site:.6f}, |M|/N = {mag_per_site:.6f}")
        
        # 测量阶段
        if verbose:
            print(f"\n{task_prefix}测量阶段开始 (T={temperature:.4f})...")
            print(f"{task_prefix}测量步数: {measurement_steps}，测量间隔: {measurement_interval}")
            print(f"{task_prefix}预计测量点数: ~{measurement_steps//measurement_interval + 1}")
        
        measure_start = time.time()
        
        # 同样分批执行测量阶段
        batch_size = 100
        n_batches = max(1, measurement_steps // batch_size)
        last_batch_size = measurement_steps % batch_size if measurement_steps % batch_size != 0 else batch_size
        
        all_accept_rates = []
        all_energies = []
        all_magnetizations = []
        
        if verbose:
            iterator = tqdm(range(n_batches), desc=f"{task_prefix}测量", 
                           unit="批", ncols=100, position=0, leave=True,
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        else:
            iterator = range(n_batches)
            
        for batch in iterator:
            sweeps_in_batch = last_batch_size if batch == n_batches - 1 else batch_size
            
            accept_rates, batch_energies, batch_magnetizations, energy, magnetization = batch_monte_carlo_measurements(
                self.lattice, temperature, energy, magnetization, 
                self.J, self.H, self.L, sweeps_in_batch, measurement_interval, random_state
            )
            
            all_accept_rates.extend(accept_rates)
            all_energies.extend(batch_energies)
            all_magnetizations.extend(batch_magnetizations)
            
            # 更新进度条
            if verbose:
                avg_rate = np.mean(accept_rates)
                energy_per_site = energy / self.N
                mag_per_site = abs(magnetization) / self.N
                total_steps = (batch + 1) * batch_size * self.N
                
                iterator.set_postfix({
                    "接受率": f"{avg_rate:.3f}",
                    "E/N": f"{energy_per_site:.4f}",
                    "M/N": f"{mag_per_site:.4f}",
                    "测量点": f"{len(all_energies)}",
                    "总尝试": f"{total_steps:,}"
                })
        
        # 更新实例变量
        self._energy = energy
        self._magnetization = magnetization
        self.random_state = random_state
        
        measure_time = time.time() - measure_start
        total_time = therm_time + measure_time
        
        if verbose:
            avg_accept = np.mean(all_accept_rates)
            energy_per_site = np.mean(all_energies) / self.N
            mag_abs = np.mean(np.abs(all_magnetizations))
            mag_per_site = mag_abs / self.N
            
            print(f"\n{task_prefix}测量完成，耗时 {measure_time:.2f} 秒 (总耗时: {total_time:.2f} 秒)")
            print(f"{task_prefix}平均接受率: {avg_accept:.4f}，共采集 {len(all_energies)} 个数据点")
            print(f"{task_prefix}平均物理量: <E>/N = {energy_per_site:.6f}, <|M|>/N = {mag_per_site:.6f}")
        
        return np.array(all_energies), np.array(all_magnetizations) 