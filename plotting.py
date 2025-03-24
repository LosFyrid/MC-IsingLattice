"""
Plotting Module for Visualization
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
import os

# 设置中文字体支持（如果有安装中文字体）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except:
    pass

# 设置图表样式
plt.style.use('seaborn-v0_8-darkgrid')
mpl.rcParams['figure.figsize'] = (10, 6)
mpl.rcParams['font.size'] = 12
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.grid'] = True


def setup_output_dir():
    """
    创建输出目录
    
    返回:
        输出目录路径
    """
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def plot_energy_vs_temperature(temperatures, energies_dict, output_dir=None):
    """
    绘制能量与温度的关系图
    
    参数:
        temperatures: 温度列表
        energies_dict: 不同系统大小对应的能量字典 {L: energies}
        output_dir: 输出目录
    """
    plt.figure(figsize=(10, 6))
    
    color_map = plt.cm.viridis
    colors = [color_map(i) for i in np.linspace(0, 0.9, len(energies_dict))]
    
    for (L, energies), color in zip(energies_dict.items(), colors):
        plt.plot(temperatures, energies, '-o', label=f'L = {L}', color=color, markersize=4)
    
    plt.xlabel('Temperature (T)')
    plt.ylabel('Average Energy (E)')
    plt.title('Energy vs Temperature')
    plt.legend()
    plt.grid(True)
    
    ax = plt.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'energy_vs_temperature.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'energy_vs_temperature.pdf'))
    
    plt.show()


def plot_magnetization_vs_temperature(temperatures, magnetizations_dict, output_dir=None):
    """
    绘制磁化强度与温度的关系图
    
    参数:
        temperatures: 温度列表
        magnetizations_dict: 不同系统大小对应的磁化强度字典 {L: magnetizations}
        output_dir: 输出目录
    """
    plt.figure(figsize=(10, 6))
    
    color_map = plt.cm.viridis
    colors = [color_map(i) for i in np.linspace(0, 0.9, len(magnetizations_dict))]
    
    for (L, magnetizations), color in zip(magnetizations_dict.items(), colors):
        plt.plot(temperatures, magnetizations, '-o', label=f'L = {L}', color=color, markersize=4)
    
    plt.xlabel('Temperature (T)')
    plt.ylabel('Magnetization (M)')
    plt.title('Magnetization vs Temperature')
    plt.legend()
    plt.grid(True)
    
    ax = plt.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # 添加垂直线表示理论临界温度
    plt.axvline(x=2.269, color='r', linestyle='--', label='Theoretical Tc = 2.269')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'magnetization_vs_temperature.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'magnetization_vs_temperature.pdf'))
    
    plt.show()


def plot_susceptibility_vs_temperature(temperatures, susceptibilities_dict, output_dir=None, critical_temps=None):
    """
    绘制磁化率与温度的关系图
    
    参数:
        temperatures: 温度列表
        susceptibilities_dict: 不同系统大小对应的磁化率字典 {L: susceptibilities}
        output_dir: 输出目录
        critical_temps: 可选的临界温度字典 {L: Tc}，如果提供则使用这些值标记峰值
    """
    plt.figure(figsize=(10, 6))
    
    color_map = plt.cm.viridis
    colors = [color_map(i) for i in np.linspace(0, 0.9, len(susceptibilities_dict))]
    
    # 如果没有提供临界温度，则使用find_critical_temperature计算
    if critical_temps is None:
        critical_temps = {}
        from analysis import find_critical_temperature
        for L, susceptibilities in susceptibilities_dict.items():
            critical_temps[L] = find_critical_temperature(temperatures, susceptibilities, method='fit_poly')
    
    # 找到最大磁化率值，用于设置y轴范围
    max_chi = 0
    for L, susceptibilities in susceptibilities_dict.items():
        # 排除低温区域(T<1)的异常值
        valid_indices = np.where(temperatures > 1.0)[0]
        if len(valid_indices) > 0:
            valid_chi = susceptibilities[valid_indices]
            max_chi = max(max_chi, np.max(valid_chi))
    
    for (L, susceptibilities), color in zip(susceptibilities_dict.items(), colors):
        plt.plot(temperatures, susceptibilities, '-o', label=f'L = {L}', color=color, markersize=4)
        
        # 标记计算出的临界温度
        if L in critical_temps:
            Tc = critical_temps[L]
            
            # 找到最接近Tc的温度点
            closest_idx = np.argmin(np.abs(temperatures - Tc))
            chi_at_tc = susceptibilities[closest_idx]
            
            # 如果需要更精确的峰值高度，可以使用插值
            from scipy.interpolate import interp1d
            try:
                # 在临界温度附近使用插值获取更精确的峰值高度
                window = max(5, len(temperatures) // 20)  # 5% of data points or at least 5 points
                start_idx = max(0, closest_idx - window)
                end_idx = min(len(temperatures), closest_idx + window + 1)
                
                if end_idx - start_idx > 3:  # 需要至少4个点进行三次插值
                    interp_t = temperatures[start_idx:end_idx]
                    interp_chi = susceptibilities[start_idx:end_idx]
                    
                    interp_func = interp1d(interp_t, interp_chi, kind='cubic')
                    
                    # 确保Tc在插值区间内
                    if interp_t[0] <= Tc <= interp_t[-1]:
                        chi_at_tc = float(interp_func(Tc))
            except:
                # 插值失败则使用最近点的值
                pass
                
            plt.plot(Tc, chi_at_tc, 'x', color='black', markersize=8)
            plt.text(Tc, chi_at_tc*1.05, f'Tc({L})={Tc:.3f}', 
                     ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Temperature (T)')
    plt.ylabel('Susceptibility (χ)')
    plt.title('Susceptibility vs Temperature')
    plt.legend()
    plt.grid(True)
    
    # 添加垂直线表示理论临界温度
    plt.axvline(x=2.269, color='r', linestyle='--', label='Theoretical Tc = 2.269')
    
    # 设置更合理的y轴范围，排除异常值
    ax = plt.gca()
    if max_chi > 0:
        ax.set_ylim(0, max_chi * 1.2)  # 给最大值留出20%的空间
    
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'susceptibility_vs_temperature.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'susceptibility_vs_temperature.pdf'))
    
    plt.show()


def plot_specific_heat_vs_temperature(temperatures, specific_heats_dict, output_dir=None, critical_temps=None):
    """
    绘制比热容与温度的关系图
    
    参数:
        temperatures: 温度列表
        specific_heats_dict: 不同系统大小对应的比热容字典 {L: specific_heats}
        output_dir: 输出目录
        critical_temps: 可选的临界温度字典 {L: Tc}，如果提供则使用这些值标记峰值
    """
    plt.figure(figsize=(10, 6))
    
    color_map = plt.cm.viridis
    colors = [color_map(i) for i in np.linspace(0, 0.9, len(specific_heats_dict))]
    
    # 如果没有提供临界温度，则使用find_critical_temperature计算
    if critical_temps is None:
        critical_temps = {}
        from analysis import find_critical_temperature
        for L, specific_heats in specific_heats_dict.items():
            critical_temps[L] = find_critical_temperature(temperatures, specific_heats, method='fit_poly')
    
    # 找到合理的比热最大值，排除低温区(T<1)的异常值
    max_c = 0
    for L, specific_heats in specific_heats_dict.items():
        valid_indices = np.where(temperatures > 1.0)[0]
        if len(valid_indices) > 0:
            valid_c = specific_heats[valid_indices]
            max_c = max(max_c, np.max(valid_c))
    
    for (L, specific_heats), color in zip(specific_heats_dict.items(), colors):
        plt.plot(temperatures, specific_heats, '-o', label=f'L = {L}', color=color, markersize=4)
        
        # 标记计算出的临界温度
        if L in critical_temps:
            Tc = critical_temps[L]
            
            # 找到最接近Tc的温度点
            closest_idx = np.argmin(np.abs(temperatures - Tc))
            c_at_tc = specific_heats[closest_idx]
            
            # 如果需要更精确的峰值高度，可以使用插值
            from scipy.interpolate import interp1d
            try:
                # 在临界温度附近使用插值获取更精确的峰值高度
                window = max(5, len(temperatures) // 20)  # 5% of data points or at least 5 points
                start_idx = max(0, closest_idx - window)
                end_idx = min(len(temperatures), closest_idx + window + 1)
                
                if end_idx - start_idx > 3:  # 需要至少4个点进行三次插值
                    interp_t = temperatures[start_idx:end_idx]
                    interp_c = specific_heats[start_idx:end_idx]
                    
                    interp_func = interp1d(interp_t, interp_c, kind='cubic')
                    
                    # 确保Tc在插值区间内
                    if interp_t[0] <= Tc <= interp_t[-1]:
                        c_at_tc = float(interp_func(Tc))
            except:
                # 插值失败则使用最近点的值
                pass
            
            plt.plot(Tc, c_at_tc, 'x', color='black', markersize=8)
            plt.text(Tc, c_at_tc*1.05, f'Tc({L})={Tc:.3f}', 
                     ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Temperature (T)')
    plt.ylabel('Specific Heat (C)')
    plt.title('Specific Heat vs Temperature')
    plt.legend()
    plt.grid(True)
    
    # 添加垂直线表示理论临界温度
    plt.axvline(x=2.269, color='r', linestyle='--', label='Theoretical Tc = 2.269')
    
    # 设置更合理的y轴范围，排除异常值
    ax = plt.gca()
    if max_c > 0:
        ax.set_ylim(0, max_c * 1.2)  # 给最大值留出20%的空间
    
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'specific_heat_vs_temperature.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'specific_heat_vs_temperature.pdf'))
    
    # 创建第二个图表，显示完整的数据范围，包括低温区的异常值
    plt.figure(figsize=(10, 6))
    for (L, specific_heats), color in zip(specific_heats_dict.items(), colors):
        plt.plot(temperatures, specific_heats, '-o', label=f'L = {L}', color=color, markersize=4)
    
    plt.xlabel('Temperature (T)')
    plt.ylabel('Specific Heat (C)')
    plt.title('Specific Heat vs Temperature (Full Range)')
    plt.legend()
    plt.grid(True)
    plt.axvline(x=2.269, color='r', linestyle='--', label='Theoretical Tc = 2.269')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'specific_heat_vs_temperature_full.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'specific_heat_vs_temperature_full.pdf'))
    
    plt.show()


def plot_critical_temperature_vs_size(system_sizes, critical_temperatures, 
                                     method='chi', Tc_inf=None, output_dir=None):
    """
    绘制临界温度与系统大小的关系图
    
    参数:
        system_sizes: 系统大小列表
        critical_temperatures: 临界温度列表
        method: 临界温度的估计方法 ('chi'或'C')
        Tc_inf: 估计的无穷大系统临界温度
        output_dir: 输出目录
    """
    plt.figure(figsize=(10, 6))
    
    # 系统大小的倒数
    L_inverse = 1.0 / np.array(system_sizes)
    
    # 绘制数据点
    plt.plot(L_inverse, critical_temperatures, 'o', label='Data points', markersize=8)
    
    # 绘制拟合曲线
    x_fit = np.linspace(min(L_inverse)*0.9, max(L_inverse)*1.1, 100)
    
    # 首先绘制线性拟合
    from scipy.stats import linregress
    result = linregress(L_inverse, critical_temperatures)
    y_linear = result.slope * x_fit + result.intercept
    plt.plot(x_fit, y_linear, '--', label=f'Linear fit: Tc(∞) = {result.intercept:.4f}', color='green')
    
    # 如果有非线性拟合结果，也绘制
    if Tc_inf is not None:
        # 从analysis模块导入scaling_func
        from analysis import finite_size_scaling
        
        def scaling_func(L_inv, Tc_inf, a, inv_v):
            return Tc_inf + a * L_inv**inv_v
        
        # 使用反向工程确定a和inv_v（如果v有值）
        # Tc(L) = Tc(∞) + aL^(-1/v)
        # 假设用户传入的Tc_inf来自非线性拟合
        
        # 尝试用传入的数据反向估计参数
        try:
            from scipy.optimize import curve_fit
            
            # 修复参数Tc_inf，只拟合a和inv_v
            def fixed_Tc_func(L_inv, a, inv_v):
                return Tc_inf + a * L_inv**inv_v
            
            popt, _ = curve_fit(fixed_Tc_func, L_inverse, critical_temperatures, 
                              p0=[1.0, 1.0], method='trf')
            a, inv_v = popt
            
            # 绘制非线性拟合结果
            y_nonlinear = scaling_func(x_fit, Tc_inf, a, inv_v)
            plt.plot(x_fit, y_nonlinear, '-', 
                   label=f'Non-linear fit: Tc(∞) = {Tc_inf:.4f}, v = {1/inv_v:.2f}', 
                   color='red')
        except Exception as e:
            print(f"Warning: Could not generate non-linear fit plot: {e}")
    
    plt.xlabel('1/L')
    
    method_name = "Susceptibility" if method == 'chi' else "Specific Heat"
    plt.ylabel(f'Critical Temperature Tc (from {method_name})')
    
    plt.title(f'Critical Temperature vs System Size (from {method_name})')
    
    # 添加理论值
    plt.axhline(y=2.269, color='blue', linestyle='--', label='Theoretical Tc = 2.269')
    
    plt.legend()
    plt.grid(True)
    
    # 添加最小尺寸标记
    min_size = min(system_sizes)
    plt.text(1/min_size, min([tc for tc in critical_temperatures]), 
             f'L = {min_size}', ha='right', va='bottom')
    
    # 添加最大尺寸标记
    max_size = max(system_sizes)
    plt.text(1/max_size, max([tc for tc in critical_temperatures]), 
             f'L = {max_size}', ha='left', va='top')
    
    ax = plt.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    plt.tight_layout()
    
    if output_dir:
        # 根据分析方法生成不同的文件名
        method_suffix = "_susceptibility" if method == 'chi' else "_specific_heat"
        plt.savefig(os.path.join(output_dir, f'critical_temperature_vs_size{method_suffix}.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, f'critical_temperature_vs_size{method_suffix}.pdf'))
    
    plt.show()


def plot_susceptibility_scaling(system_sizes, chi_max_values, scaling_exponent, r_squared, output_dir=None):
    """
    绘制磁化率峰值与系统尺寸的标度关系
    
    参数:
        system_sizes: 系统尺寸列表
        chi_max_values: 对应的磁化率峰值
        scaling_exponent: 拟合得到的标度指数
        r_squared: 拟合的R²值
        output_dir: 输出目录
    """
    plt.figure(figsize=(10, 6))
    
    # 对数坐标制作散点图
    plt.loglog(system_sizes, chi_max_values, 'o', markersize=8, label='Simulation data')
    
    # 拟合曲线
    x_range = np.logspace(np.log10(min(system_sizes)*0.9), np.log10(max(system_sizes)*1.1), 100)
    reference_point = (system_sizes[0], chi_max_values[0])
    y_fit = reference_point[1] * (x_range / reference_point[0])**scaling_exponent
    
    plt.loglog(x_range, y_fit, '-', 
              label=f'Fit: χ_max ∝ L^{scaling_exponent:.4f}, R² = {r_squared:.4f}')
    
    # 理论曲线
    theo_exponent = 7.0/4.0  # γ/ν = 7/4 for 2D Ising model
    y_theo = reference_point[1] * (x_range / reference_point[0])**theo_exponent
    
    plt.loglog(x_range, y_theo, '--', 
              label=f'Theory: χ_max ∝ L^{theo_exponent:.4f}')
    
    # 设置图表样式
    plt.xlabel('System Size (L)')
    plt.ylabel('Maximum Susceptibility (χ_max)')
    plt.title('Susceptibility Scaling: χ_max ∝ L^(γ/ν)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    # 添加系统尺寸标签
    for i, L in enumerate(system_sizes):
        plt.annotate(f'L={L}', 
                   xy=(L, chi_max_values[i]),
                   xytext=(5, 5),
                   textcoords='offset points',
                   fontsize=8)
    
    # 添加额外信息
    error_percent = abs(scaling_exponent - theo_exponent) / theo_exponent * 100
    plt.figtext(0.15, 0.15, 
               f"Scaling exponent: {scaling_exponent:.4f}\n"
               f"Theoretical value: {theo_exponent:.4f}\n"
               f"Relative error: {error_percent:.2f}%",
               bbox=dict(facecolor='white', alpha=0.8))
    
    # 创建插入图：log-log图
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    log_L = np.log(system_sizes)
    log_chi = np.log(chi_max_values)
    
    axins = inset_axes(plt.gca(), width="40%", height="30%", loc=4, borderpad=2)
    axins.scatter(log_L, log_chi, s=30, marker='o')
    
    # 绘制拟合线
    x_fit = np.linspace(min(log_L)*0.9, max(log_L)*1.1, 100)
    y_fit = scaling_exponent * x_fit + (log_chi[0] - scaling_exponent * log_L[0])
    axins.plot(x_fit, y_fit, 'r-')
    
    # 绘制理论线
    y_theo = theo_exponent * x_fit + (log_chi[0] - theo_exponent * log_L[0])
    axins.plot(x_fit, y_theo, 'g--')
    
    axins.set_xlabel('ln(L)', fontsize=8)
    axins.set_ylabel('ln(χ_max)', fontsize=8)
    axins.tick_params(axis='both', which='major', labelsize=6)
    axins.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'susceptibility_scaling.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'susceptibility_scaling.pdf'))
    
    plt.show() 