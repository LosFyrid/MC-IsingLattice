"""
二维伊辛格模型的模拟模块
"""
import numpy as np
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os

from ising_model import IsingLattice
import analysis


def run_single_temperature(params):
    """
    在单个温度下运行模拟
    
    参数:
        params: 参数元组 (L, T, therm_steps, measure_steps, measure_interval, task_id, verbose)
        
    返回:
        (L, T, 平均能量, 比热容, 平均磁化强度, 磁化率)
    """
    L, T, therm_steps, measure_steps, measure_interval, task_id, verbose = params
    N = L * L
    
    # 初始化格子
    lattice = IsingLattice(L, J=1.0, H=0.0)
    
    # 运行模拟
    if verbose:
        print(f"\n[任务 {task_id}] 开始模拟: L = {L}, T = {T:.4f}, 格点数 = {N}")
        print(f"[任务 {task_id}] 热化步数: {therm_steps}, 测量步数: {measure_steps}, 测量间隔: {measure_interval}")
    
    start_time = time.time()
    energies, magnetizations = lattice.run_simulation(
        temperature=T,
        thermalization_steps=therm_steps,
        measurement_steps=measure_steps,
        measurement_interval=measure_interval,
        task_id=task_id,
        verbose=verbose
    )
    end_time = time.time()
    elapsed = end_time - start_time
    
    # 计算物理量
    avg_energy = analysis.calculate_average_energy(energies, N)
    specific_heat = analysis.calculate_specific_heat(energies, T, N)
    avg_magnetization = analysis.calculate_average_magnetization(magnetizations, N)
    susceptibility = analysis.calculate_susceptibility(magnetizations, T, N)
    
    if verbose:
        print(f"\n[任务 {task_id}] 完成: L = {L}, T = {T:.4f}, 耗时 {elapsed:.2f} 秒")
        print(f"[任务 {task_id}] 物理量: 能量 = {avg_energy:.6f}, 磁化强度 = {avg_magnetization:.6f}")
        print(f"[任务 {task_id}] 物理量: 比热容 = {specific_heat:.6f}, 磁化率 = {susceptibility:.6f}")
        print("-" * 50)
    
    return L, T, avg_energy, specific_heat, avg_magnetization, susceptibility


def run_simulation_sweep(L_values, T_range, therm_steps=10**5, measure_steps=3*10**5, 
                        measure_interval=10, n_processes=None, cpu_limit=80, verbose=True):
    """
    运行一系列温度和系统大小的模拟
    
    参数:
        L_values: 系统大小列表
        T_range: 温度列表
        therm_steps: 热化步数
        measure_steps: 测量步数
        measure_interval: 测量间隔
        n_processes: 并行进程数 (None表示自动根据CPU限制设置)
        cpu_limit: CPU使用率上限百分比 (默认80%)
        verbose: 是否显示进度信息
        
    返回:
        结果字典
    """
    # 设置并行进程数
    available_cpus = cpu_count()
    
    if n_processes is None:
        # 根据CPU限制计算使用的CPU数量
        n_processes = max(1, int(available_cpus * cpu_limit / 100))
        # 确保至少保留一个核心空闲
        n_processes = min(n_processes, available_cpus - 1)
    
    # 准备并行任务参数
    tasks = []
    task_id = 0
    for L in L_values:
        for T in T_range:
            task_id += 1
            tasks.append((L, T, therm_steps, measure_steps, measure_interval, task_id, verbose))
    
    # 运行并行模拟
    if verbose:
        print("=" * 70)
        print(f"开始并行模拟：{len(tasks)}个任务")
        print(f"将使用{n_processes}个CPU核心 (总共{available_cpus}个可用, 限制{cpu_limit}%)")
        print(f"系统大小: {L_values}")
        print(f"温度范围: {min(T_range):.3f} 到 {max(T_range):.3f}, 共{len(T_range)}个点")
        print("=" * 70)
    
    start_time = time.time()
    
    # 保存任务状态到文件
    if not os.path.exists('output'):
        os.makedirs('output')
    with open('output/tasks_info.txt', 'w') as f:
        f.write(f"总任务数: {len(tasks)}\n")
        f.write(f"系统大小: {L_values}\n")
        f.write(f"温度范围: {min(T_range):.3f} 到 {max(T_range):.3f}, 共{len(T_range)}个点\n")
        f.write("\n任务列表:\n")
        for i, (L, T, _, _, _, task_id, _) in enumerate(tasks):
            f.write(f"任务 {task_id}: L = {L}, T = {T:.4f}\n")
    
    results = []
    with Pool(n_processes) as pool:
        # 使用tqdm显示进度条
        iterator = pool.imap(run_single_temperature, tasks)
        results = list(tqdm(iterator, total=len(tasks), desc="总进度", 
                          unit="任务", ncols=100, position=0))
    
    end_time = time.time()
    total_time = end_time - start_time
    
    if verbose:
        print("\n" + "=" * 70)
        print(f"模拟完成：总耗时 {total_time:.2f} 秒 ({total_time/3600:.2f} 小时)")
        print(f"平均每个任务耗时: {total_time/len(tasks):.2f} 秒")
        print("=" * 70)
    
    # 整理结果
    temperatures = T_range
    
    energy_data = {L: [] for L in L_values}
    specific_heat_data = {L: [] for L in L_values}
    magnetization_data = {L: [] for L in L_values}
    susceptibility_data = {L: [] for L in L_values}
    
    for r in results:
        L, T, e, c, m, chi = r
        
        # 按温度顺序插入数据
        t_idx = np.where(temperatures == T)[0][0]
        
        # 确保列表长度符合预期
        while len(energy_data[L]) < t_idx:
            energy_data[L].append(None)
            specific_heat_data[L].append(None)
            magnetization_data[L].append(None)
            susceptibility_data[L].append(None)
        
        # 添加数据
        if len(energy_data[L]) == t_idx:
            energy_data[L].append(e)
            specific_heat_data[L].append(c)
            magnetization_data[L].append(m)
            susceptibility_data[L].append(chi)
        else:
            energy_data[L][t_idx] = e
            specific_heat_data[L][t_idx] = c
            magnetization_data[L][t_idx] = m
            susceptibility_data[L][t_idx] = chi
    
    # 进行最终处理以确保数据完整
    for L in L_values:
        # 确保所有列表长度相等
        assert len(energy_data[L]) == len(temperatures), f"L={L}的能量数据长度不匹配"
        assert len(specific_heat_data[L]) == len(temperatures), f"L={L}的比热容数据长度不匹配"
        assert len(magnetization_data[L]) == len(temperatures), f"L={L}的磁化强度数据长度不匹配"
        assert len(susceptibility_data[L]) == len(temperatures), f"L={L}的磁化率数据长度不匹配"
        
        # 转换为numpy数组
        energy_data[L] = np.array(energy_data[L])
        specific_heat_data[L] = np.array(specific_heat_data[L])
        magnetization_data[L] = np.array(magnetization_data[L])
        susceptibility_data[L] = np.array(susceptibility_data[L])
    
    # 找出临界温度
    critical_temperatures = {}
    for L in L_values:
        # 从磁化率估计
        critical_temperatures[L] = analysis.find_critical_temperature(
            temperatures, susceptibility_data[L])
    
    # 返回结果字典
    return {
        'temperatures': temperatures,
        'energy': energy_data,
        'specific_heat': specific_heat_data,
        'magnetization': magnetization_data,
        'susceptibility': susceptibility_data,
        'critical_temperatures': critical_temperatures
    } 