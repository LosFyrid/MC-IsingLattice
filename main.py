"""
二维伊辛格模型主程序
"""
import numpy as np
import os
import argparse
import json
import time

from ising_model import IsingLattice
from simulation import run_simulation_sweep
from analysis import finite_size_scaling
import plotting


def main():
    """主程序入口"""
    
    parser = argparse.ArgumentParser(description='二维伊辛格模型模拟')
    parser.add_argument('--L', type=int, nargs='+', default=[10, 20, 30],
                       help='系统大小列表 (默认: 10 20 30)')
    parser.add_argument('--T-min', type=float, default=0.015,
                       help='最小温度 (默认: 0.015)')
    parser.add_argument('--T-max', type=float, default=4.5,
                       help='最大温度 (默认: 4.5)')
    parser.add_argument('--T-step', type=float, default=0.015,
                       help='温度步长 (默认: 0.015)')
    parser.add_argument('--therm-steps', type=int, default=10**5,
                       help='热化步数 (默认: 10^5)')
    parser.add_argument('--measure-steps', type=int, default=3*10**5,
                       help='测量步数 (默认: 3*10^5)')
    parser.add_argument('--measure-interval', type=int, default=10,
                       help='测量间隔 (默认: 10)')
    parser.add_argument('--processes', type=int, default=None,
                       help='并行进程数 (默认: 根据CPU限制自动设置)')
    parser.add_argument('--cpu-limit', type=int, default=80,
                       help='CPU使用率上限百分比 (默认: 80%%)')
    parser.add_argument('--load', type=str, default=None,
                       help='加载之前的结果文件 (默认: 不加载)')
    parser.add_argument('--no-plots', action='store_true',
                       help='不生成图表')
    parser.add_argument('--save-prefix', type=str, default="",
                       help='结果文件名前缀 (默认: "")')
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = plotting.setup_output_dir()
    
    # 确定是加载已有结果还是运行新模拟
    if args.load and os.path.exists(args.load):
        print(f"加载先前的结果: {args.load}")
        with open(args.load, 'r') as f:
            results = json.load(f)
            
        # 转换为正确的数据类型
        temperatures = np.array(results['temperatures'])
        energy_data = {int(L): np.array(data) for L, data in results['energy'].items()}
        specific_heat_data = {int(L): np.array(data) for L, data in results['specific_heat'].items()}
        magnetization_data = {int(L): np.array(data) for L, data in results['magnetization'].items()}
        susceptibility_data = {int(L): np.array(data) for L, data in results['susceptibility'].items()}
        critical_temperatures = {int(L): float(Tc) for L, Tc in results['critical_temperatures'].items()}
        
    else:
        # 准备温度范围
        temperatures = np.arange(args.T_min, args.T_max + args.T_step/2, args.T_step)
        
        print("=" * 70)
        print("二维伊辛格模型模拟")
        print("=" * 70)
        print(f"系统大小: {args.L}")
        print(f"温度范围: {args.T_min} 到 {args.T_max}，步长 {args.T_step}")
        print(f"热化步数: {args.therm_steps}")
        print(f"测量步数: {args.measure_steps}")
        print(f"测量间隔: {args.measure_interval}")
        print(f"CPU使用率限制: {args.cpu_limit}%")
        print("=" * 70)
        
        # 运行模拟
        start_time = time.time()
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
        print(f"总模拟时间: {total_time:.2f} 秒 ({total_time/3600:.2f} 小时)")
        
        # 提取结果
        temperatures = results['temperatures']
        energy_data = results['energy']
        specific_heat_data = results['specific_heat']
        magnetization_data = results['magnetization']
        susceptibility_data = results['susceptibility']
        critical_temperatures = results['critical_temperatures']
        
        # 保存结果到JSON文件
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
                'critical_temperatures': {str(L): float(Tc) for L, Tc in critical_temperatures.items()},
                'parameters': {
                    'L_values': args.L,
                    'T_min': args.T_min,
                    'T_max': args.T_max,
                    'T_step': args.T_step,
                    'therm_steps': args.therm_steps,
                    'measure_steps': args.measure_steps,
                    'measure_interval': args.measure_interval,
                    'cpu_limit': args.cpu_limit,
                    'total_time_seconds': total_time
                }
            }
            json.dump(json_results, f, indent=2)
        
        # 创建一个符号链接到最新结果
        latest_link = os.path.join(output_dir, "results_latest.json")
        if os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(os.path.basename(save_path), latest_link)
        
        print(f"结果已保存到 {save_path}")
        print(f"符号链接创建于 {latest_link}")
    
    # 打印临界温度估计值
    print("\n估计的临界温度:")
    for L, Tc in critical_temperatures.items():
        print(f"L = {L}: Tc = {Tc:.4f}")
    
    # 有限尺寸标度分析
    system_sizes = list(critical_temperatures.keys())
    Tc_values = [critical_temperatures[L] for L in system_sizes]
    Tc_inf, v = finite_size_scaling(Tc_values, system_sizes)
    
    print(f"\n有限尺寸标度分析:")
    print(f"估计的无穷大系统临界温度 Tc(∞) = {Tc_inf:.4f}")
    if v:
        print(f"估计的临界指数 ν = {v:.4f}")
    print(f"相对于理论值 Tc = 2.269 的误差: {abs(Tc_inf - 2.269)/2.269*100:.2f}%")
    
    # 绘制图表
    if not args.no_plots:
        print("\n生成图表...")
        
        # 能量vs温度
        plotting.plot_energy_vs_temperature(
            temperatures, energy_data, output_dir=output_dir)
        
        # 磁化强度vs温度
        plotting.plot_magnetization_vs_temperature(
            temperatures, magnetization_data, output_dir=output_dir)
        
        # 磁化率vs温度
        plotting.plot_susceptibility_vs_temperature(
            temperatures, susceptibility_data, output_dir=output_dir, critical_temps=critical_temperatures)
        
        # 比热容vs温度
        plotting.plot_specific_heat_vs_temperature(
            temperatures, specific_heat_data, output_dir=output_dir, critical_temps=critical_temperatures)
        
        # 临界温度vs系统大小
        plotting.plot_critical_temperature_vs_size(
            system_sizes, Tc_values, method='chi', 
            Tc_inf=Tc_inf, output_dir=output_dir)
        
        print(f"图表已保存到 {output_dir} 目录")


if __name__ == "__main__":
    main() 