"""
二维伊辛格模型的模拟模块 - Numba加速版本
"""
import numpy as np
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os

from ising_model_numba import IsingLatticeNumba
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
    lattice = IsingLatticeNumba(L, J=1.0, H=0.0)
    
    # 运行模拟
    if verbose:
        print(f"\n[Task {task_id}] Starting simulation: L = {L}, T = {T:.4f}, sites = {N}")
        print(f"[Task {task_id}] Thermalization steps: {therm_steps}, measurement steps: {measure_steps}, interval: {measure_interval}")
    
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
        print(f"\n[Task {task_id}] Completed: L = {L}, T = {T:.4f}, time: {elapsed:.2f} seconds")
        print(f"[Task {task_id}] Results: Energy = {avg_energy:.6f}, Magnetization = {avg_magnetization:.6f}")
        print(f"[Task {task_id}] Results: Specific Heat = {specific_heat:.6f}, Susceptibility = {susceptibility:.6f}")
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
        total_sites = sum([L*L for L in L_values]) * len(T_range)
        total_mc_steps = (therm_steps + measure_steps) * total_sites
        print("=" * 70)
        print(f"Starting parallel simulation (Numba accelerated): {len(tasks)} tasks")
        print(f"Using {n_processes} CPU cores (out of {available_cpus} available, limit {cpu_limit}%)")
        print(f"System sizes: {L_values}")
        print(f"Temperature range: {min(T_range):.3f} to {max(T_range):.3f}, {len(T_range)} points")
        print(f"Thermalization steps: {therm_steps}, Measurement steps: {measure_steps}, Interval: {measure_interval}")
        print(f"Total sites: {total_sites}, Total Monte Carlo attempts: {total_mc_steps:,}")
        print("-" * 70)
        print("First Numba function run will have a short compilation delay, subsequent runs will be faster")
        print("=" * 70)
    
    start_time = time.time()
    
    # 保存任务状态到文件
    if not os.path.exists('output'):
        os.makedirs('output')
    task_info_file = 'output/numba_tasks_info.txt'
    with open(task_info_file, 'w') as f:
        f.write(f"2D Ising Model Numba Accelerated Simulation Task Information\n")
        f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total tasks: {len(tasks)}\n")
        f.write(f"System sizes: {L_values}\n")
        f.write(f"Temperature range: {min(T_range):.3f} to {max(T_range):.3f}, {len(T_range)} points\n")
        f.write(f"Thermalization steps: {therm_steps}, Measurement steps: {measure_steps}, Interval: {measure_interval}\n")
        f.write(f"CPU cores: {n_processes}/{available_cpus}\n\n")
        f.write("Task list:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Task ID':^8}|{'Size':^10}|{'Temp':^10}|{'Status':^10}\n")
        f.write("-" * 50 + "\n")
        for i, (L, T, _, _, _, task_id, _) in enumerate(tasks):
            f.write(f"{task_id:^8}|{L:^10}|{T:^10.4f}|{'Waiting':^10}\n")
    
    if verbose:
        print(f"Task information saved to {task_info_file}")
    
    # 任务完成计数器（用于更新进度）
    completed_tasks = 0
    
    # 更新任务状态的函数
    def task_completed(result):
        nonlocal completed_tasks
        L, T, _, _, _, _ = result
        completed_tasks += 1
        
        # 更新任务信息文件
        with open(task_info_file, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            if len(line) > 20 and f"|{L:^10}|{T:^10.4f}|" in line:
                lines[i] = line.replace("Waiting", "Completed")
                break
        
        with open(task_info_file, 'w') as f:
            f.writelines(lines)
        
        if verbose and completed_tasks % max(1, len(tasks)//20) == 0:
            percent = completed_tasks / len(tasks) * 100
            elapsed = time.time() - start_time
            est_total = elapsed / (completed_tasks / len(tasks))
            remaining = est_total - elapsed
            print(f"Completed: {completed_tasks}/{len(tasks)} tasks ({percent:.1f}%), "
                  f"Estimated remaining time: {remaining/60:.1f} minutes")
    
    results = []
    with Pool(n_processes) as pool:
        # 使用tqdm显示进度条
        iterator = pool.imap(run_single_temperature, tasks)
        for result in tqdm(iterator, total=len(tasks), desc="Overall progress", 
                         unit="tasks", ncols=100, position=0,
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'):
            results.append(result)
            task_completed(result)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    if verbose:
        print("\n" + "=" * 70)
        print(f"Simulation complete: Total time {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        print(f"Average time per task: {total_time/len(tasks):.2f} seconds")
        print(f"Estimated Monte Carlo attempts per second: {total_mc_steps/total_time:.2e}")
        
        # 添加Numba加速对比提示
        print("-" * 70)
        print("Numba acceleration tips:")
        print(" * For large systems (L ≥ 20), Numba version is typically 5-10 times faster than the original")
        print(" * The first run of Numba includes compilation time, subsequent runs will be faster")
        print(" * Actual speedup depends on your CPU and system size")
        print("=" * 70)
    
    # 在任务信息文件中添加完成信息
    with open(task_info_file, 'a') as f:
        f.write("\n" + "-" * 50 + "\n")
        f.write(f"Completion time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)\n")
        f.write(f"Average time per task: {total_time/len(tasks):.2f} seconds\n")
    
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
    
    # 返回结果字典 - 只包含原始测量数据，不包含计算的临界温度
    return {
        'temperatures': temperatures,
        'energy': energy_data,
        'specific_heat': specific_heat_data,
        'magnetization': magnetization_data,
        'susceptibility': susceptibility_data
    } 