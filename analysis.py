"""
Data Analysis Module for Calculating Various Thermodynamic Quantities
"""
import numpy as np


def calculate_average_energy(energies, N):
    """
    Calculate the average energy
    
    Parameters:
        energies: List of energy measurements
        N: System size (number of sites)
        
    Returns:
        Average energy per site
    """
    return np.mean(energies) / N


def calculate_specific_heat(energies, temperature, N):
    """
    Calculate the specific heat
    
    Parameters:
        energies: List of energy measurements
        temperature: Temperature T
        N: System size (number of sites)
        
    Returns:
        Specific heat C
    """
    # C = (1/NkT²)⟨(δE)²⟩, where k=1
    energy_variance = np.var(energies)
    return energy_variance / (N * temperature**2)


def calculate_average_magnetization(magnetizations, N):
    """
    Calculate the average magnetization
    
    Parameters:
        magnetizations: List of magnetization measurements
        N: System size (number of sites)
        
    Returns:
        Average magnetization per site
    """
    # Note: For finite systems, M should be 0 above the critical temperature
    # but due to finite-size effects, we take the absolute value of magnetization
    # before averaging to avoid cancellation of positive and negative values
    return np.mean(np.abs(magnetizations)) / N


def calculate_susceptibility(magnetizations, temperature, N):
    """
    Calculate the magnetic susceptibility
    
    Parameters:
        magnetizations: List of magnetization measurements
        temperature: Temperature T
        N: System size (number of sites)
        
    Returns:
        Susceptibility χ
    """
    # χ = (1/NT)⟨(δM)²⟩
    mag_variance = np.var(magnetizations)
    return mag_variance / (N * temperature)


def calculate_binder_cumulant(magnetizations):
    """
    Calculate the Binder cumulant
    
    Parameters:
        magnetizations: List of magnetization measurements
        
    Returns:
        Binder cumulant U
    """
    # U = 1 - <M⁴>/(3<M²>²)
    m2 = np.mean(magnetizations**2)
    m4 = np.mean(magnetizations**4)
    return 1.0 - m4 / (3.0 * m2**2)


def find_critical_temperature(temperatures, observable, method='max'):
    """
    Estimate the critical temperature
    
    Parameters:
        temperatures: List of temperatures
        observable: Corresponding list of physical quantities (usually susceptibility or specific heat)
        method: Method for estimating critical temperature:
            - 'max': Taking the temperature corresponding to the maximum value
            - 'fit_gaussian': Fit a Gaussian curve to the peak region
            - 'fit_poly': Fit a polynomial curve to the peak region
        
    Returns:
        Estimated critical temperature Tc
    """
    # Only consider temperatures greater than 1.0 to avoid interference from low-temperature anomalies
    valid_indices = np.where(temperatures > 1.0)[0]
    if len(valid_indices) == 0:
        # If there are no data points with T>1, fall back to searching the entire range
        valid_indices = np.arange(len(temperatures))
    
    valid_temps = temperatures[valid_indices]
    valid_observable = observable[valid_indices]
    
    if method == 'max':
        max_idx = np.argmax(valid_observable)
        return valid_temps[max_idx]
    
    elif method == 'fit_gaussian':
        from scipy.optimize import curve_fit
        from scipy.signal import find_peaks
        
        # First find approximate peak position
        max_idx = np.argmax(valid_observable)
        
        # Define fit region around the peak (consider ~20% range around peak)
        window_size = max(5, len(valid_temps) // 5)
        start_idx = max(0, max_idx - window_size)
        end_idx = min(len(valid_temps), max_idx + window_size + 1)
        
        fit_temps = valid_temps[start_idx:end_idx]
        fit_observable = valid_observable[start_idx:end_idx]
        
        # Define Gaussian function
        def gaussian(x, a, x0, sigma):
            return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
        
        try:
            # Initial parameter guess
            peak_value = fit_observable[max_idx - start_idx]
            peak_temp = fit_temps[max_idx - start_idx]
            initial_sigma = (fit_temps[-1] - fit_temps[0]) / 5  # Rough estimate
            
            # Fit Gaussian
            popt, _ = curve_fit(gaussian, fit_temps, fit_observable, 
                              p0=[peak_value, peak_temp, initial_sigma])
            
            # Extract peak position (x0)
            fitted_tc = popt[1]
            
            print(f"Gaussian fit result: Tc = {fitted_tc:.6f} (original max: {valid_temps[max_idx]:.6f})")
            return fitted_tc
        
        except Exception as e:
            print(f"Gaussian fitting failed: {e}, falling back to max method")
            return valid_temps[max_idx]
    
    elif method == 'fit_poly':
        # Find approximate peak position
        max_idx = np.argmax(valid_observable)
        
        # Define fit region around the peak
        window_size = max(5, len(valid_temps) // 5)
        start_idx = max(0, max_idx - window_size)
        end_idx = min(len(valid_temps), max_idx + window_size + 1)
        
        fit_temps = valid_temps[start_idx:end_idx]
        fit_observable = valid_observable[start_idx:end_idx]
        
        try:
            # Fit 4th order polynomial (enough to capture the peak shape)
            coeff = np.polyfit(fit_temps, fit_observable, 4)
            
            # Find analytical maximum - derivative of polynomial = 0
            # For 4th order polynomial: a*x^4 + b*x^3 + c*x^2 + d*x + e
            # Derivative: 4*a*x^3 + 3*b*x^2 + 2*c*x + d
            poly_deriv = np.polyder(coeff)
            
            # Find roots of the derivative
            roots = np.roots(poly_deriv)
            
            # Find the real root closest to the original peak
            real_roots = roots[np.isreal(roots)].real
            
            # Filter roots that are within the temperature range
            valid_roots = real_roots[(real_roots >= np.min(fit_temps)) & 
                                    (real_roots <= np.max(fit_temps))]
            
            if len(valid_roots) == 0:
                # Fall back to max method if no valid roots
                return valid_temps[max_idx]
            
            # Evaluate polynomial at each root to find the highest peak
            poly = np.poly1d(coeff)
            root_values = poly(valid_roots)
            max_root_idx = np.argmax(root_values)
            fitted_tc = valid_roots[max_root_idx]
            
            print(f"Polynomial fit result: Tc = {fitted_tc:.6f} (original max: {valid_temps[max_idx]:.6f})")
            return fitted_tc
            
        except Exception as e:
            print(f"Polynomial fitting failed: {e}, falling back to max method")
            return valid_temps[max_idx]
    
    else:
        raise ValueError(f"Unknown method: {method}")


def finite_size_scaling(critical_temperatures, system_sizes):
    """
    Finite-size scaling analysis
    
    Parameters:
        critical_temperatures: Critical temperatures for different system sizes
        system_sizes: List of system sizes
        
    Returns:
        (Critical temperature for infinite system Tc, Critical exponent v)
    """
    # Check data is reasonable
    print("\nFinite-size scaling analysis:")
    print("System sizes:", system_sizes)
    print("Critical temperatures:", [f"{tc:.6f}" for tc in critical_temperatures])
    
    # Inverse of system size
    L_inverse = 1.0 / np.array(system_sizes)
    
    # First perform linear fit, this can be used as an initial guess for non-linear fit
    from scipy.stats import linregress
    result = linregress(L_inverse, critical_temperatures)
    Tc_inf_linear = result.intercept
    
    print(f"Linear fit results: Tc(∞) = {Tc_inf_linear:.6f}, slope = {result.slope:.6f}, r-value = {result.rvalue:.6f}")
    
    # Non-linear fit: Tc(L) = Tc(∞) + aL^(-1/v)
    try:
        from scipy.optimize import curve_fit
        
        def scaling_func(L_inv, Tc_inf, a, inv_v):
            return Tc_inf + a * L_inv**inv_v
        
        # Better initial parameters: Use linear fit as initial guess
        # Tc_inf≈intercept of linear fit, a≈slope of linear fit, inv_v≈1.0
        initial_guess = [Tc_inf_linear, result.slope, 1.0]
        
        # Set reasonable parameter boundaries
        bounds = ([2.0, -np.inf, 0.1], [2.5, np.inf, 10.0])  # Tc(∞) in 2.0-2.5, inv_v in 0.1-10
        
        print(f"Starting non-linear fit with initial parameters: {initial_guess}")
        
        # Use more robust fitting method
        popt, pcov = curve_fit(scaling_func, L_inverse, critical_temperatures, 
                               p0=initial_guess, bounds=bounds, method='trf', 
                               ftol=1e-8, xtol=1e-8, max_nfev=10000)
        
        Tc_inf, a, inv_v = popt
        perr = np.sqrt(np.diag(pcov))  # Parameter error estimate
        
        print(f"Non-linear fit results: Tc(∞) = {Tc_inf:.6f}±{perr[0]:.6f}, "
              f"a = {a:.6f}±{perr[1]:.6f}, 1/v = {inv_v:.6f}±{perr[2]:.6f}")
        
        # Calculate fit goodness
        y_fit = scaling_func(L_inverse, *popt)
        residuals = critical_temperatures - y_fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((critical_temperatures - np.mean(critical_temperatures))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"Non-linear fit goodness: R² = {r_squared:.6f}")
        
        # Check if non-linear fit result is accepted
        if r_squared > 0.95 and 0.5 < inv_v < 5.0:
            print("Non-linear fit accepted as valid")
            return Tc_inf, 1.0 / inv_v
        else:
            print("Non-linear fit rejected due to poor quality, using linear fit instead")
            return Tc_inf_linear, None
    
    except Exception as e:
        print(f"Non-linear fit failed with error: {e}")
        print("Using linear fit results instead")
        return Tc_inf_linear, None 


def verify_susceptibility_scaling(system_sizes, susceptibility_data, temperatures, method='fit_poly'):
    """
    Verify whether the maximum susceptibility follows the scaling relation χ_max ∝ L^(γ/ν)
    For 2D Ising model, theoretical value is γ/ν ≈ 7/4 = 1.75
    
    Parameters:
        system_sizes: List of system sizes
        susceptibility_data: Dictionary of susceptibility data for each system size
        temperatures: List of temperatures
        method: Method for finding peak ('max' or 'fit_poly' or 'fit_gaussian')
        
    Returns:
        (Scaling exponent, R-squared value)
    """
    print("\nVerifying susceptibility scaling relation χ_max ∝ L^(γ/ν):")
    print("Theoretical value for 2D Ising model: γ/ν ≈ 7/4 = 1.75")
    
    # Get maximum susceptibility for each system size
    chi_max_values = []
    
    for L in system_sizes:
        if method == 'max':
            # Simply find the maximum value
            # Only consider temperatures > 1.0 to avoid low-T anomalies
            valid_indices = np.where(temperatures > 1.0)[0]
            valid_temps = temperatures[valid_indices]
            valid_chi = susceptibility_data[L][valid_indices]
            
            max_idx = np.argmax(valid_chi)
            chi_max = valid_chi[max_idx]
            T_max = valid_temps[max_idx]
        else:
            # Use the same peak finding method as in find_critical_temperature
            # This ensures consistency in identifying the peaks
            T_max = find_critical_temperature(temperatures, susceptibility_data[L], method=method)
            
            # Find the corresponding susceptibility value
            # Interpolate if necessary to get a more accurate peak height
            from scipy.interpolate import interp1d
            
            # Create interpolation function for more accurate peak height
            # Focus on a narrow region around the peak
            peak_idx = np.argmin(np.abs(temperatures - T_max))
            window = max(5, len(temperatures) // 20)  # 5% of data points or at least 5 points
            start_idx = max(0, peak_idx - window)
            end_idx = min(len(temperatures), peak_idx + window + 1)
            
            interp_region_t = temperatures[start_idx:end_idx]
            interp_region_chi = susceptibility_data[L][start_idx:end_idx]
            
            if len(interp_region_t) > 3:  # Need at least 4 points for cubic interpolation
                # Create cubic spline interpolation
                interp_func = interp1d(interp_region_t, interp_region_chi, kind='cubic')
                
                # Fine sampling around the peak
                fine_t = np.linspace(interp_region_t[0], interp_region_t[-1], 1000)
                fine_chi = interp_func(fine_t)
                
                # Find maximum in the interpolated data
                max_idx = np.argmax(fine_chi)
                chi_max = fine_chi[max_idx]
            else:
                # If not enough points for interpolation, use the nearest value
                nearest_idx = np.argmin(np.abs(temperatures - T_max))
                chi_max = susceptibility_data[L][nearest_idx]
        
        chi_max_values.append(chi_max)
        print(f"L = {L:2d}: χ_max = {chi_max:.6f} at T = {T_max:.6f}")
    
    # Perform log-log linear regression to find the scaling exponent
    from scipy.stats import linregress
    
    log_L = np.log(system_sizes)
    log_chi_max = np.log(chi_max_values)
    
    result = linregress(log_L, log_chi_max)
    scaling_exponent = result.slope
    
    # Calculate R-squared
    r_squared = result.rvalue**2
    
    # Calculate theoretical line
    theo_exponent = 7.0/4.0  # γ/ν = 7/4 for 2D Ising model
    
    print(f"\nFitted scaling exponent: {scaling_exponent:.4f}")
    print(f"Theoretical exponent: {theo_exponent:.4f}")
    print(f"Relative error: {abs(scaling_exponent - theo_exponent)/theo_exponent*100:.2f}%")
    print(f"R-squared: {r_squared:.6f}")
    
    return scaling_exponent, r_squared 