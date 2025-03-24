"""
能量和磁化强度导数分析 - 计算二维伊辛模型中物理量随温度变化的最大变化率

此脚本计算不同系统尺寸下能量和磁化强度对温度的最大变化率(导数)
"""

import numpy as np
import json
import os
import argparse
from scipy.signal import savgol_filter

def load_simulation_data(filename):
    """从JSON文件加载模拟数据"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # 转换数据为适当的格式
    temperatures = np.array(data['temperatures'])
    energy_data = {int(L): np.array(values) for L, values in data['energy'].items()}
    magnetization_data = {int(L): np.array(values) for L, values in data['magnetization'].items()}
    
    return temperatures, energy_data, magnetization_data

def calculate_derivative(temperatures, values, smoothing=True):
    """
    计算物理量对温度的数值导数
    
    参数:
        temperatures: 温度值数组
        values: 物理量值数组
        smoothing: 是否在微分前对数据进行平滑处理
        
    返回:
        temps: 导数对应的温度点
        derivative: 导数值
    """
    if smoothing:
        # 使用Savitzky-Golay滤波器进行平滑处理
        window_size = min(15, len(values) - 2)  # 必须是奇数且小于数据长度
        if window_size % 2 == 0:
            window_size -= 1
        if window_size >= 5:  # 平滑至少需要5个点
            values_smooth = savgol_filter(values, window_size, 3)
        else:
            values_smooth = values
    else:
        values_smooth = values
    
    # 使用中心有限差分计算数值导数
    derivative = np.gradient(values_smooth, temperatures)
    
    return temperatures, derivative

def find_max_derivative(temperatures, derivative, min_temp=1.0, max_temp=None, tc_window=None):
    """
    找出最大导数对应的温度
    
    参数:
        temperatures: 温度值数组
        derivative: 导数值数组
        min_temp: 要考虑的最小温度(避免低温异常)
        max_temp: 要考虑的最大温度
        tc_window: 如果提供，将在理论临界温度附近的窗口内搜索 (Tc-window, Tc+window)
        
    返回:
        max_temp: 最大导数对应的温度
        max_der: 最大导数值
    """
    # 理论临界温度
    Tc = 2.269
    
    # 如果提供tc_window，则在临界温度附近搜索
    if tc_window is not None:
        valid_idx = np.where((temperatures >= Tc - tc_window) & 
                             (temperatures <= Tc + tc_window))[0]
        print(f"  在临界温度窗口 T = [{Tc - tc_window:.3f}, {Tc + tc_window:.3f}] 内搜索最值")
    else:
        # 否则使用传统的min_temp和max_temp范围
        if max_temp is None:
            valid_idx = np.where(temperatures >= min_temp)[0]
        else:
            valid_idx = np.where((temperatures >= min_temp) & 
                                (temperatures <= max_temp))[0]
    
    valid_temps = temperatures[valid_idx]
    valid_der = derivative[valid_idx]
    
    # 找出最大值点
    max_idx = np.argmax(valid_der)
    max_temp = valid_temps[max_idx]
    max_der = valid_der[max_idx]
    
    return max_temp, max_der

def find_min_derivative(temperatures, derivative, min_temp=1.0, max_temp=None, tc_window=None):
    """
    找出最小导数对应的温度
    
    参数:
        temperatures: 温度值数组
        derivative: 导数值数组
        min_temp: 要考虑的最小温度(避免低温异常)
        max_temp: 要考虑的最大温度
        tc_window: 如果提供，将在理论临界温度附近的窗口内搜索 (Tc-window, Tc+window)
        
    返回:
        min_temp: 最小导数对应的温度
        min_der: 最小导数值
    """
    # 理论临界温度
    Tc = 2.269
    
    # 如果提供tc_window，则在临界温度附近搜索
    if tc_window is not None:
        valid_idx = np.where((temperatures >= Tc - tc_window) & 
                             (temperatures <= Tc + tc_window))[0]
        print(f"  在临界温度窗口 T = [{Tc - tc_window:.3f}, {Tc + tc_window:.3f}] 内搜索最值")
    else:
        # 否则使用传统的min_temp和max_temp范围
        if max_temp is None:
            valid_idx = np.where(temperatures >= min_temp)[0]
        else:
            valid_idx = np.where((temperatures >= min_temp) & 
                                (temperatures <= max_temp))[0]
    
    valid_temps = temperatures[valid_idx]
    valid_der = derivative[valid_idx]
    
    # 找出最小值点
    min_idx = np.argmin(valid_der)
    min_temp = valid_temps[min_idx]
    min_der = valid_der[min_idx]
    
    return min_temp, min_der

def analyze_derivatives(temperatures, energy_data, magnetization_data, output_dir=None):
    """
    分析所有系统尺寸的能量和磁化强度导数
    
    参数:
        temperatures: 温度值数组
        energy_data: 每个系统尺寸的能量值字典
        magnetization_data: 每个系统尺寸的磁化强度值字典
        output_dir: 保存输出的目录(None表示不保存)
    """
    # 结果存储
    energy_results = []
    mag_results = []
    
    print("\n" + "=" * 70)
    print("能量和磁化强度导数分析结果 (在理论临界温度Tc=2.269附近)")
    print("-" * 70)
    
    # 理论临界温度
    Tc = 2.269
    # 搜索窗口大小
    tc_window = 0.2  # 在Tc±0.5的范围内搜索
    
    system_sizes = sorted(energy_data.keys())
    
    for L in system_sizes:
        energies = energy_data[L]
        magnetizations = magnetization_data[L]
        
        print(f"系统尺寸 L = {L}:")
        
        # 计算能量导数
        temps, dE_dT = calculate_derivative(temperatures, energies, smoothing=True)
        
        # 找出能量最大导数点 (在临界温度附近)
        max_temp_E, max_der_E = find_max_derivative(temps, dE_dT, tc_window=tc_window)
        
        # 计算磁化强度导数
        temps, dM_dT = calculate_derivative(temperatures, magnetizations, smoothing=True)
        
        # 找出磁化强度最大导数点 (在临界温度附近)
        max_temp_M, max_der_M = find_max_derivative(temps, dM_dT, tc_window=tc_window)
        
        # 找出磁化强度最小导数点 (在临界温度附近)
        # 在临界温度附近，磁化强度会急剧下降，所以导数为负
        min_temp_M, min_der_M = find_min_derivative(temps, dM_dT, tc_window=tc_window)
        
        # 存储结果
        energy_results.append({
            'L': L,
            'max_derivative_temp': max_temp_E,
            'max_derivative_value': max_der_E
        })
        
        mag_results.append({
            'L': L,
            'max_derivative_temp': max_temp_M,
            'max_derivative_value': max_der_M,
            'min_derivative_temp': min_temp_M,
            'min_derivative_value': min_der_M
        })
        
        print(f"  能量导数最大值温度: T = {max_temp_E:.6f} (与Tc相差: {abs(max_temp_E-Tc):.6f})")
        print(f"  能量最大导数值: dE/dT = {max_der_E:.6f}")
        print(f"  磁化强度导数最大值温度: T = {max_temp_M:.6f} (与Tc相差: {abs(max_temp_M-Tc):.6f})")
        print(f"  磁化强度最大导数值: dM/dT = {max_der_M:.6f}")
        print(f"  磁化强度导数最小值温度: T = {min_temp_M:.6f} (与Tc相差: {abs(min_temp_M-Tc):.6f})")
        print(f"  磁化强度最小导数值: dM/dT = {min_der_M:.6f}")
        print()
    
    # 在控制台输出表格形式的结果摘要 - 能量导数
    print("\n" + "-" * 80)
    print("能量导数最大值:")
    print(f"{'系统尺寸(L)':^10}|{'最大导数温度':^15}|{'与Tc的差值':^15}|{'最大导数值':^15}")
    print("-" * 80)
    for r in energy_results:
        temp = r['max_derivative_temp']
        diff = abs(temp - Tc)
        print(f"{r['L']:^10}|{temp:^15.6f}|{diff:^15.6f}|{r['max_derivative_value']:^15.6f}")
    
    # 在控制台输出表格形式的结果摘要 - 磁化强度导数
    print("\n" + "-" * 80)
    print("磁化强度导数最大值:")
    print(f"{'系统尺寸(L)':^10}|{'最大导数温度':^15}|{'与Tc的差值':^15}|{'最大导数值':^15}")
    print("-" * 80)
    for r in mag_results:
        temp = r['max_derivative_temp']
        diff = abs(temp - Tc)
        print(f"{r['L']:^10}|{temp:^15.6f}|{diff:^15.6f}|{r['max_derivative_value']:^15.6f}")
    
    print("\n" + "-" * 80)
    print("磁化强度导数最小值(下降最快点):")
    print(f"{'系统尺寸(L)':^10}|{'最小导数温度':^15}|{'与Tc的差值':^15}|{'最小导数值':^15}")
    print("-" * 80)
    for r in mag_results:
        temp = r['min_derivative_temp']
        diff = abs(temp - Tc)
        print(f"{r['L']:^10}|{temp:^15.6f}|{diff:^15.6f}|{r['min_derivative_value']:^15.6f}")
    
    # 保存结果到CSV
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        import csv
        with open(os.path.join(output_dir, 'derivative_results.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['系统尺寸(L)', 
                           '能量最大导数温度', '与Tc差值', '能量最大导数值', 
                           '磁化强度最大导数温度', '与Tc差值', '磁化强度最大导数值',
                           '磁化强度最小导数温度', '与Tc差值', '磁化强度最小导数值'])
            
            for i, L in enumerate(system_sizes):
                e_temp = energy_results[i]['max_derivative_temp']
                e_diff = abs(e_temp - Tc)
                m_max_temp = mag_results[i]['max_derivative_temp']
                m_max_diff = abs(m_max_temp - Tc)
                m_min_temp = mag_results[i]['min_derivative_temp']
                m_min_diff = abs(m_min_temp - Tc)
                
                writer.writerow([
                    L, 
                    e_temp, e_diff, energy_results[i]['max_derivative_value'],
                    m_max_temp, m_max_diff, mag_results[i]['max_derivative_value'],
                    m_min_temp, m_min_diff, mag_results[i]['min_derivative_value']
                ])
        print(f"\n结果已保存到CSV文件: {os.path.join(output_dir, 'derivative_results.csv')}")
    
    return energy_results, mag_results

def main():
    parser = argparse.ArgumentParser(description='分析二维伊辛模型中能量和磁化强度导数')
    parser.add_argument('--input', type=str, default='output/numba_results_latest.json',
                       help='输入的JSON模拟结果文件')
    parser.add_argument('--output', type=str, default='output',
                       help='结果输出目录')
    parser.add_argument('--tc-window', type=float, default=0.5,
                       help='理论临界温度附近的搜索窗口大小 (默认: 0.5)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("二维伊辛模型物理量导数分析")
    print("=" * 70)
    print(f"从文件加载数据: {args.input}")
    print(f"理论临界温度: Tc = 2.269")
    print(f"搜索窗口: Tc ± {args.tc_window}")
    temperatures, energy_data, magnetization_data = load_simulation_data(args.input)
    
    print(f"分析{len(energy_data)}种系统尺寸的能量和磁化强度导数")
    energy_results, mag_results = analyze_derivatives(
        temperatures, energy_data, magnetization_data, 
        output_dir=args.output
    )
    
    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)

if __name__ == "__main__":
    main() 