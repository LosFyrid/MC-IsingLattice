"""
2D Ising Model Main Program - Numba Accelerated Version
"""
import numpy as np
import os
import argparse
import json
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from ising_model_numba import IsingLatticeNumba
from simulation_numba import run_simulation_sweep
from analysis import finite_size_scaling, find_critical_temperature, verify_susceptibility_scaling
import plotting


def main():
    """Main program entry point"""
    
    parser = argparse.ArgumentParser(description='2D Ising Model Simulation (Numba Accelerated)')
    parser.add_argument('--L', type=int, nargs='+', default=[10, 20, 30],
                       help='List of system sizes (default: 10 20 30)')
    parser.add_argument('--T-min', type=float, default=0.015,
                       help='Minimum temperature (default: 0.015)')
    parser.add_argument('--T-max', type=float, default=4.5,
                       help='Maximum temperature (default: 4.5)')
    parser.add_argument('--T-step', type=float, default=0.015,
                       help='Temperature step (default: 0.015)')
    parser.add_argument('--therm-steps', type=int, default=10**5,
                       help='Thermalization steps (default: 10^5)')
    parser.add_argument('--measure-steps', type=int, default=3*10**5,
                       help='Measurement steps (default: 3*10^5)')
    parser.add_argument('--measure-interval', type=int, default=10,
                       help='Measurement interval (default: 10)')
    parser.add_argument('--processes', type=int, default=None,
                       help='Number of parallel processes (default: automatically set based on CPU limit)')
    parser.add_argument('--cpu-limit', type=int, default=80,
                       help='CPU usage limit percentage (default: 80%%)')
    parser.add_argument('--load', type=str, default=None,
                       help='Load previous results file (default: none)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Do not generate plots')
    parser.add_argument('--save-prefix', type=str, default="numba_",
                       help='Result file name prefix (default: "numba_")')
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = plotting.setup_output_dir()
    
    # 确定是加载已有结果还是运行新模拟
    if args.load and os.path.exists(args.load):
        print(f"Loading previous results: {args.load}")
        with open(args.load, 'r') as f:
            results = json.load(f)
            
        # 转换为正确的数据类型
        temperatures = np.array(results['temperatures'])
        energy_data = {int(L): np.array(data) for L, data in results['energy'].items()}
        specific_heat_data = {int(L): np.array(data) for L, data in results['specific_heat'].items()}
        magnetization_data = {int(L): np.array(data) for L, data in results['magnetization'].items()}
        susceptibility_data = {int(L): np.array(data) for L, data in results['susceptibility'].items()}
        
    else:
        # 准备温度范围
        temperatures = np.arange(args.T_min, args.T_max + args.T_step/2, args.T_step)
        
        print("=" * 70)
        print("2D Ising Model Simulation (Numba Accelerated)")
        print("=" * 70)
        print(f"System sizes: {args.L}")
        print(f"Temperature range: {args.T_min} to {args.T_max}, step {args.T_step}, {len(temperatures)} points")
        print(f"Thermalization steps: {args.therm_steps}")
        print(f"Measurement steps: {args.measure_steps}")
        print(f"Measurement interval: {args.measure_interval}")
        print(f"CPU usage limit: {args.cpu_limit}%")
        
        # 计算总尝试次数
        total_sites = sum([L*L for L in args.L]) * len(temperatures)
        total_mc_steps = (args.therm_steps + args.measure_steps) * total_sites
        
        print(f"Total lattice sites: {total_sites}, Total Monte Carlo attempts: {total_mc_steps:,}")
        print("-" * 70)
        print("Numba acceleration notes:")
        print(" * First run will have compilation delay, subsequent runs will be faster")
        print(" * Acceleration is more significant for larger systems (L ≥ 20)")
        print(" * There's some overhead in inter-process communication, but overall performance is much better")
        print("=" * 70)
        
        # 运行模拟
        start_time = time.time()
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = run_simulation_sweep(
            L_values=args.L,
            T_range=temperatures,
            therm_steps=args.therm_steps,
            measure_steps=args.measure_steps,
            measure_interval=args.measure_interval,
            n_processes=args.processes,
            cpu_limit=args.cpu_limit,
            verbose=True
        )
        end_time = time.time()
        total_time = end_time - start_time
        mc_steps_per_sec = total_mc_steps / total_time
        
        print("\n" + "=" * 70)
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total simulation time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        print(f"Monte Carlo attempts per second: {mc_steps_per_sec:.2e}")
        print(f"Processing speed per site: {mc_steps_per_sec/total_sites:.2f} attempts/second")
        print("=" * 70)
        
        # 提取结果
        temperatures = results['temperatures']
        energy_data = results['energy']
        specific_heat_data = results['specific_heat']
        magnetization_data = results['magnetization']
        susceptibility_data = results['susceptibility']
        
        # 保存结果到JSON文件，注意不再存储临界温度
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{args.save_prefix}results_{timestamp}.json" if args.save_prefix else f"results_{timestamp}.json"
        save_path = os.path.join(output_dir, filename)
        
        with open(save_path, 'w') as f:
            # 转换为可序列化格式
            json_results = {
                'temperatures': temperatures.tolist(),
                'energy': {str(L): data.tolist() for L, data in energy_data.items()},
                'specific_heat': {str(L): data.tolist() for L, data in specific_heat_data.items()},
                'magnetization': {str(L): data.tolist() for L, data in magnetization_data.items()},
                'susceptibility': {str(L): data.tolist() for L, data in susceptibility_data.items()},
                'parameters': {
                    'L_values': args.L,
                    'T_min': args.T_min,
                    'T_max': args.T_max,
                    'T_step': args.T_step,
                    'therm_steps': args.therm_steps,
                    'measure_steps': args.measure_steps,
                    'measure_interval': args.measure_interval,
                    'cpu_limit': args.cpu_limit,
                    'total_time_seconds': total_time,
                    'version': 'numba'
                }
            }
            json.dump(json_results, f, indent=2)
        
        try:
            # 创建一个符号链接到最新结果
            latest_link = os.path.join(output_dir, "numba_results_latest.json")
            if os.path.exists(latest_link):
                os.remove(latest_link)
            os.symlink(os.path.basename(save_path), latest_link)
            print(f"Results saved to {save_path}")
            print(f"Symbolic link created at {latest_link}")
        except Exception as e:
            # 在Windows上symlink可能需要管理员权限
            print(f"Results saved to {save_path}")
            print(f"Warning: Could not create symbolic link: {e}")
            print(f"You can manually copy the file if needed.")
        
        print(f"File size: {os.path.getsize(save_path)/1024:.2f} KB")
        print("\nSaved data contents:")
        print(f" - Temperature points: {len(temperatures)}")
        print(f" - System sizes: {len(args.L)} types ({args.L})")
        print(f" - Includes: energy, magnetization, specific heat, susceptibility")
        print(f" - Includes simulation parameters")
        print(f" - JSON format for further analysis or visualization")
    
    # 使用当前的分析方法计算临界温度（无论是加载的还是新模拟的数据）
    print("\n" + "=" * 70)
    print("Critical Temperature Analysis")
    print("-" * 70)
    
    # 从磁化率估计临界温度
    print("Method 1: Estimating critical temperatures from susceptibility peaks")
    chi_critical_temperatures = {}
    for L in energy_data.keys():
        chi_critical_temperatures[L] = find_critical_temperature(
            temperatures, susceptibility_data[L], method='fit_poly')
    
    print("Critical temperature estimates from susceptibility peaks:")
    for L, Tc in chi_critical_temperatures.items():
        print(f"L = {L:2d}: Tc = {Tc:.6f}")
    
    # 从比热容估计临界温度
    print("\nMethod 2: Estimating critical temperatures from specific heat peaks")
    c_critical_temperatures = {}
    for L in energy_data.keys():
        c_critical_temperatures[L] = find_critical_temperature(
            temperatures, specific_heat_data[L], method='fit_poly')
    
    print("Critical temperature estimates from specific heat peaks:")
    for L, Tc in c_critical_temperatures.items():
        print(f"L = {L:2d}: Tc = {Tc:.6f}")
    
    # 有限尺寸标度分析 - 使用磁化率数据
    print("\n" + "-" * 70)
    print("Finite-size scaling analysis using susceptibility data:")
    chi_system_sizes = list(chi_critical_temperatures.keys())
    chi_Tc_values = [chi_critical_temperatures[L] for L in chi_system_sizes]
    chi_Tc_inf, chi_v = finite_size_scaling(chi_Tc_values, chi_system_sizes)
    
    theoretical_Tc = 2.269
    chi_error = abs(chi_Tc_inf - theoretical_Tc)/theoretical_Tc*100
    
    print(f"From susceptibility: Estimated Tc(∞) = {chi_Tc_inf:.6f}")
    if chi_v:
        print(f"Critical exponent ν = {chi_v:.6f}")
        print(f"Theoretical ν = 1.0, relative error: {abs(chi_v-1.0)/1.0*100:.2f}%")
    
    print(f"Relative error compared to theoretical Tc = {theoretical_Tc}: {chi_error:.4f}%")
    
    # 有限尺寸标度分析 - 使用比热容数据
    print("\nFinite-size scaling analysis using specific heat data:")
    c_system_sizes = list(c_critical_temperatures.keys())
    c_Tc_values = [c_critical_temperatures[L] for L in c_system_sizes]
    c_Tc_inf, c_v = finite_size_scaling(c_Tc_values, c_system_sizes)
    
    c_error = abs(c_Tc_inf - theoretical_Tc)/theoretical_Tc*100
    
    print(f"From specific heat: Estimated Tc(∞) = {c_Tc_inf:.6f}")
    if c_v:
        print(f"Critical exponent ν = {c_v:.6f}")
        print(f"Theoretical ν = 1.0, relative error: {abs(c_v-1.0)/1.0*100:.2f}%")
    
    print(f"Relative error compared to theoretical Tc = {theoretical_Tc}: {c_error:.4f}%")
    
    # 打印比较结果和精度评估
    print("\n" + "-" * 70)
    print("Comparison of results:")
    print(f"Theoretical critical temperature: Tc(theory) = {theoretical_Tc}")
    print(f"From susceptibility: Tc(∞) = {chi_Tc_inf:.6f}, error: {chi_error:.4f}%")
    print(f"From specific heat: Tc(∞) = {c_Tc_inf:.6f}, error: {c_error:.4f}%")
    
    # 验证磁化率峰值高度的标度关系
    print("\n" + "-" * 70)
    print("Susceptibility Peak Scaling Analysis:")
    system_sizes = sorted(energy_data.keys())
    scaling_exponent, r_squared = verify_susceptibility_scaling(
        system_sizes, susceptibility_data, temperatures, method='fit_poly')
    
    # 添加可视化磁化率峰值标度关系的代码
    if not args.no_plots:
        # 准备数据 - 使用与verify_susceptibility_scaling相同的插值方法获取峰值
        chi_max_values = []
        for L in system_sizes:
            # 利用拟合函数找到临界温度
            Tc = find_critical_temperature(temperatures, susceptibility_data[L], method='fit_poly')
            
            # 在临界温度附近使用三次样条插值获取更精确的峰值
            # 创建插值区域
            peak_idx = np.argmin(np.abs(temperatures - Tc))
            window = max(5, len(temperatures) // 20)  # 至少5个点或数据的5%
            start_idx = max(0, peak_idx - window)
            end_idx = min(len(temperatures), peak_idx + window + 1)
            
            interp_region_t = temperatures[start_idx:end_idx]
            interp_region_chi = susceptibility_data[L][start_idx:end_idx]
            
            if len(interp_region_t) > 3:  # 三次插值至少需要4个点
                # 创建三次样条插值
                interp_func = interp1d(interp_region_t, interp_region_chi, kind='cubic')
                
                # 在临界温度附近进行精细采样
                fine_t = np.linspace(interp_region_t[0], interp_region_t[-1], 1000)
                fine_chi = interp_func(fine_t)
                
                # 在插值数据中找到最大值
                max_idx = np.argmax(fine_chi)
                chi_max = fine_chi[max_idx]
            else:
                # 如果插值点不足，直接使用最接近的值
                chi_max = susceptibility_data[L][peak_idx]
            
            chi_max_values.append(chi_max)
        
        # 调用专门的绘图函数
        print("Plotting Susceptibility Scaling Relation...")
        plotting.plot_susceptibility_scaling(
            system_sizes, chi_max_values, scaling_exponent, r_squared, output_dir=output_dir)
        print("Generated additional plot: Susceptibility scaling relation")
    
    # 根据误差评估精度
    if min(chi_error, c_error) < 1.0:
        print("Result accuracy: Excellent (best error < 1%)")
    elif min(chi_error, c_error) < 3.0:
        print("Result accuracy: Good (best error < 3%)")
    else:
        print("Result accuracy: Fair (best error > 3%)")
        print("Tip: Increase system size or sample count for better accuracy")
    
    print("=" * 70)
    
    # 绘制图表
    if not args.no_plots:
        print("\nGenerating plots...")
        print("-" * 70)
        
        # 能量vs温度
        print("Plotting Energy vs Temperature...")
        plotting.plot_energy_vs_temperature(
            temperatures, energy_data, output_dir=output_dir)
        
        # 磁化强度vs温度
        print("Plotting Magnetization vs Temperature...")
        plotting.plot_magnetization_vs_temperature(
            temperatures, magnetization_data, output_dir=output_dir)
        
        # 磁化率vs温度
        print("Plotting Susceptibility vs Temperature...")
        plotting.plot_susceptibility_vs_temperature(
            temperatures, susceptibility_data, output_dir=output_dir, critical_temps=chi_critical_temperatures)
        
        # 比热容vs温度
        print("Plotting Specific Heat vs Temperature...")
        plotting.plot_specific_heat_vs_temperature(
            temperatures, specific_heat_data, output_dir=output_dir, critical_temps=c_critical_temperatures)
        
        # 临界温度vs系统大小 - 从磁化率获得的临界温度
        print("Plotting Critical Temperature vs System Size from susceptibility data...")
        plotting.plot_critical_temperature_vs_size(
            chi_system_sizes, chi_Tc_values, method='chi', 
            Tc_inf=chi_Tc_inf, output_dir=output_dir)
        
        # 临界温度vs系统大小 - 从比热容获得的临界温度
        print("Plotting Critical Temperature vs System Size from specific heat data...")
        # 生成唯一的文件名，以区分两种方法
        specific_heat_output_dir = output_dir
        plotting.plot_critical_temperature_vs_size(
            c_system_sizes, c_Tc_values, method='C', 
            Tc_inf=c_Tc_inf, output_dir=specific_heat_output_dir)
        
        print(f"\nPlots have been saved to {output_dir} directory")
        print("Generated plots include:")
        print(" - Energy E/N vs Temperature T")
        print(" - Magnetization |M|/N vs Temperature T")
        print(" - Susceptibility χ vs Temperature T")
        print(" - Specific Heat C vs Temperature T")
        print(" - Critical Temperature Tc (from susceptibility) vs System Size 1/L")
        print(" - Critical Temperature Tc (from specific heat) vs System Size 1/L")
        print("All plots include data for different system sizes for comparison of finite-size effects")
    else:
        print("\nSkipping plot generation (--no-plots option enabled)")


if __name__ == "__main__":
    main() 