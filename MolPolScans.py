###################################################################################################
# MOLPOL SINGLE MAGNET SCAN PLOTTER                                                               #
# ERIC KING -- 2024/2025                                                                          #
#                                                                                                 #
# python3 MolPolScans.py --file sbs5passDipoleNew.csv --energy 10.7 --magnet 5 --setpoint 1.275   #
# --file: file to read                                                                            #
# --energy: beam energy (TODO: Change to fetch from metadata line)                                #
# --magnet: specify scan magnet                                                                   #
# --setpoint: (optional) adds set point to plot and table outputs                                 #
###################################################################################################
import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import CubicSpline, interp1d, UnivariateSpline
from scipy.signal import savgol_filter
from scipy.stats import chi2

from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import chi2

from PyPDF2 import PdfMerger
from fpdf import FPDF
from datetime import datetime

# Set up argument parser
parser = argparse.ArgumentParser(description="MolPol Simulation Scan Processing")
parser.add_argument("--file", type=str, required=True, help="Path to the CSV file")
parser.add_argument("--magnet", type=int, required=True, choices=range(1, 7), help="Magnet number (1-6)")
parser.add_argument("--energy", type=str, required=True, help="Energy value")
parser.add_argument("--setpoint", type=float, required=False, default=None, help="Tune Selection [Pole Tip]")

# Parse arguments
args = parser.parse_args()

flname = args.file
energy = args.energy
magnet = args.magnet
setpnt = args.setpoint

print(f'Filename: {flname}')
print(f'Scan Mag: {magnet}')
print(f'  Energy: {energy}')
print(f'SetPoint: {setpnt if setpnt is not None else "None"}')

# Validate magnet input
if magnet < 1 or magnet > 6:
    raise ValueError("Magnet argument must be an integer between 1 and 6.")

# Read the first line manually
with open(flname, 'r') as f:
    first_line = f.readline().strip()

metadata = {}
for item in first_line.split(','):
    if '=' in item:
        key, value = item.split('=')
        metadata[key.strip()] = value.strip()
        
# MolPol magnet names
magnet_map = {
    1: "Quad1",
    2: "Quad2",
    3: "Quad3",
    4: "Quad4",
    5: "Dipole"
}
magnet_name = magnet_map[magnet]

def compute_fx(cn, mag):
    coefficients = {
        1: (0.0110605,    5.33237,  -0.0142794,     0.259313,  0.0058174,   -0.831887, 36.5723),
        2: (0.0196438,    5.35443,   0.0297273,     0.103505, -0.0449275,   -0.211868, 44.76),
        3: (0.000632446,  5.15178,  -0.00262778,   -0.107635,  0.00209902,  -0.640635, 36.74),
        4: (0.0001732,    5.2119,   -0.000732518,  -0.133423,  0.000618402, -0.647082, 36.50)
    }
    if mag in coefficients:
        a, b, c, d, e, f, denominator = coefficients[mag]
        return ((a + b * cn + c * cn**2 + d * cn**3 + e * cn**4 + f * cn**5) * 10 * 5.08) / denominator
    elif mag == 5:
        return -0.39026e-04 + 0.027051 * cn * 300 - 0.17799e-08 * (cn * 300) ** 2
    return None

def convert_mag_to_current(dframe, input_col, output_col, mag):
    def compute_current(value):
        x, tol, h = 1.0, 1e-9, 1e-6
        for _ in range(1000):
            cn = x / 300.0
            fx = compute_fx(cn, mag)
            if fx is None:
                return None
            # Be sure to convert T to kG
            f_val = value*10. - fx
            xp, xm = x + h, x - h
            fxp = compute_fx(xp / 300.0, mag)
            fxm = compute_fx(xm / 300.0, mag)
            if fxp is None or fxm is None:
                return None
            d_val = (fxp - fxm) / (2 * h)
            if d_val == 0:
                break
            x_new = x + f_val / d_val
            if abs(x_new - x) < tol:
                x = x_new
                break
            x = x_new
        return x
    dframe[output_col] = dframe[input_col].apply(compute_current)
    return dframe

# TODO: Convert Q{i} values from MagPoleTip to MagCurrent
#q_values = ', '.join([f"Q{i}: {metadata[f'Q{i}']} T" for i in range(1, 6) if f'Q{i}' in metadata and i != magnet])
q_values_converted = []
for i in range(1, 7):
    if f'Q{i}' in metadata and i != magnet:
        try:
            mag_pole_tip = float(metadata[f'Q{i}'])  # Convert Q{i} value from string to float
            df_temp = pd.DataFrame({ 'MagPoleTip': [mag_pole_tip] })  # Temporary DataFrame for conversion
            convert_mag_to_current(df_temp, 'MagPoleTip', 'MagCurrent', i)  # Convert MagPoleTip to MagCurrent
            mag_current = df_temp['MagCurrent'].values[0]
            if i<6:
                q_values_converted.append(f"Q{i}: {metadata[f'Q{i}']} T ({mag_current:.3f} A)")
            elif i==6:
                q_values_converted.append(f"Q{i}: {metadata[f'Q{i}']} T")
        except ValueError:
            continue  # Skip if conversion fails
q_values = ', '.join(q_values_converted)

desc_beamE = f"{', '.join([metadata.get('DESC', 'N/A'), metadata.get('beamE', 'N/A'), f'Magnet: {magnet_name} Scan'])}"

# Skip first line (metadata) and use the second line as data header
df = pd.read_csv(flname, skiprows=[0])
df.columns = df.columns.str.strip()
# TODO: Select axis (Unsure if I'll ever do this as an option)
horizontal_axis = 'MagCurrent'  # Default to MagCurrent may change to 'MagPoleTip' when needed

# Change norm rate from fraction to percent
for col in df.columns:
    if col.endswith('NormRt'):
        df[col] = df[col] * 100.
    
# Convert pole tips to currents
convert_mag_to_current(dframe=df,input_col='MagPoleTip',output_col='MagCurrent',mag=magnet)

# Define iterative smoothing function
def iterative_smoothing(measurement_points, measurement_values, convergence_threshold=0.005, max_iterations=50):
    measurement_fit = measurement_values.copy()
    iteration = 0
    converged = False
    while not converged and iteration < max_iterations:
        smoothed_values = savgol_filter(measurement_fit, window_length=5, polyorder=3)
        for i in range(len(smoothed_values) - 2, -1, -1):
            if measurement_points[i] < 0.2 and smoothed_values[i] > smoothed_values[i + 1]:
                smoothed_values[i] = smoothed_values[i + 1] + 1e-6
        delta_max = np.max(np.abs(smoothed_values - measurement_fit))
        if delta_max < convergence_threshold:
            converged = True
        measurement_fit[:] = smoothed_values.copy()
        iteration += 1
    return measurement_fit, iteration

def gpr_smoothing(measurement_points, measurement_values, measurement_errors, length_scale=0.9):
    """
    Applies Gaussian Process Regression (GPR) smoothing to a dataset.

    Parameters:
    measurement_points (numpy array): The x-values of the dataset.
    measurement_values (numpy array): The original measured y-values.
    measurement_errors (numpy array): The measurement uncertainties.
    length_scale (float): The length scale for the RBF kernel (default: 0.1).
    
    Returns:
    pd.DataFrame: A DataFrame containing the original and smoothed data.
    """
    scaler = MinMaxScaler()
    measurement_points_scaled = scaler.fit_transform(measurement_points.reshape(-1, 1))

    # Define the Gaussian Process kernel (constant + RBF for smoothness)
    kernel = C(1.0) * RBF(length_scale=length_scale)

    # Initialize and fit the Gaussian Process Regressor
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=(0.5 * measurement_errors) ** 2, n_restarts_optimizer=10)
    #gpr.fit(measurement_points.reshape(-1, 1), measurement_values)
    gpr.fit(measurement_points_scaled.reshape(-1, 1), measurement_values)

    # Predict the smoothed values at the original measurement points
    # smoothed_values, sigma = gpr.predict(measurement_points.reshape(-1, 1), return_std=True)
    smoothed_values, sigma = gpr.predict(measurement_points_scaled.reshape(-1, 1), return_std=True)

    # Compute chi-squared statistic and p-value
    chi2_stat = np.sum(((smoothed_values - measurement_values) / measurement_errors) ** 2)
    dof = len(measurement_values) - 1
    p_value = 1 - chi2.cdf(chi2_stat, dof)

    # Store results in a DataFrame
    df_results = pd.DataFrame({
        "MeasurementPoint": measurement_points,
        "OriginalMeasurementValue": measurement_values,
        "SmoothedMeasurementValue": smoothed_values,
        "OriginalError": measurement_errors,
        "GPRUncertainty": sigma
    })

    return df_results["SmoothedMeasurementValue"], p_value
    #return pd.DataFrame(smoothed_values, index=measurement_points), p_value


# Create a smoothed curve with some NN averaging -- decided to weight overall error (error bars on smoothed curve are thus slightly smaller)
for label in ['Coin', 'Left', 'Rght']:
    df[f'{label}CorAzzSmooth'], _ = iterative_smoothing(df['MagPoleTip'].values, df[f'{label}CorAzz'].values)
    #df[f'{label}CorAzzSmooth'], _ = gpr_smoothing(df['MagPoleTip'].values, df[f'{label}CorAzz'].values, df[f'{label}CorErr'].values)

# Check if setpnt exists in MagPoleTip column -- if it does then we only want to highlight this row, else we will have to insert and interpolate
if setpnt is not None:
    if setpnt not in df['MagPoleTip'].values:
        # Insert the setpnt value at the appropriate location
        new_row = pd.DataFrame([{col: (setpnt if col == 'MagPoleTip' else np.nan) for col in df.columns}])
        df = pd.concat([df, new_row], ignore_index=True)  # Add empty row
        df = df.sort_values(by='MagPoleTip')  # Sort after adding the new row
        df = df.reset_index(drop=True)  # Reset index

    # Interpolate missing values for all columns at inserted index
    row_index = df.index[df['MagPoleTip'] == setpnt][0]
    #print(row_index)
    x1 = df['MagPoleTip'][row_index-1]
    x2 = df['MagPoleTip'][row_index+1]
    #print(f'x1:{x1}')
    #print(f'x2:{x2}')
    interval_fraction=(setpnt-x1)/(x2-x1)
    #print(f'intervalFrac:{interval_fraction}')
    for col in df.columns:
        if col != 'MagPoleTip':
            df.loc[row_index, col] = df.loc[row_index-1, col] + interval_fraction * (df.loc[row_index+1, col] - df.loc[row_index-1, col])
        # Print entire dataframe for debugging
        #print("[DEBUG] Full DataFrame before checking duplicates:")
        #print(df.to_string(index=False))

# Ensure no duplicate values in MagCurrent after interpolation
if df.duplicated(subset=['MagCurrent']).any():
    print("\n[DEBUG] Duplicate entries in MagCurrent detected:")
    print(df[df.duplicated(subset=['MagCurrent'], keep=False)].to_string(index=False))
    
    # Drop duplicate entries after debugging
    df = df.drop_duplicates(subset=['MagCurrent'])
    
##########################################################################
# CREATE PLOTS                                                           #
##########################################################################
scale = 1.5
fig, axs = plt.subplots(2, 3, figsize=(scale*11,scale*8.5))
plt.subplots_adjust(top=0.875,bottom=0.15,left=0.075,right=0.925,hspace=0.6,wspace=0.4)
current_date = datetime.now().strftime('%Y-%m-%d')
#fig.suptitle(f'{energy} GeV | {magnet_name} Scan | {current_date}', fontsize=15, fontweight='bold')
fig.text(0.5,0.95,f'{energy} GeV | {magnet_name} Scan | {current_date}',ha='center',fontsize=20,fontweight='bold')
fig.text(0.5,0.92,q_values,ha='center',fontsize=16,fontstyle='italic')

x_min = df[horizontal_axis].min()
x_max = df[horizontal_axis].max()
#print(f'x_min:{x_min}')
#print(f'x_max:{x_max}')
y_min = np.floor(min(df[['CoinUncAzz', 'CoinCorAzz', 'LeftUncAzz', 'LeftCorAzz', 'RghtUncAzz', 'RghtCorAzz']].min()*0.99) / 0.05) * 0.05
y_max = np.ceil(max(df[['CoinUncAzz', 'CoinCorAzz', 'LeftUncAzz', 'LeftCorAzz', 'RghtUncAzz', 'RghtCorAzz']].max()*1.01) / 0.05) * 0.05
#print(f'y_min:{y_min}')
#print(f'y_max:{y_max}')

# Axis limits (TODO: Really would like to fix this to something more like was done in the past in Excel by hand.)
# Set axis mean as tune uncorrected

if setpnt is not None:
    row_index = df.index[df['MagPoleTip'] == setpnt][0]
    coin_cor_azz_tune = df['CoinCorAzz'][row_index]
    coin_cor_min = min(df[['CoinCorAzz','CoinUncAzz']].min())
    coin_cor_max = max(df[['CoinCorAzz','CoinUncAzz']].max())
    #coin_y_min_zoom = coin_cor_azz_tune * 0.975
    #coin_y_max_zoom = coin_cor_azz_tune * 1.025
    coin_y_min_zoom = coin_cor_min * 0.9975
    coin_y_max_zoom = coin_cor_max * 1.0025
else:
    coin_cor_azz_tune = df['CoinUncAzz'].mean()
    coin_y_min_zoom = coin_cor_azz_tune * 0.975
    coin_y_max_zoom = coin_cor_azz_tune * 1.025

if setpnt is not None:
    row_index = df.index[df['MagPoleTip'] == setpnt][0]
    left_cor_azz_tune = df['LeftCorAzz'][row_index]
    left_cor_min = min(df[['LeftCorAzz','LeftUncAzz']].min())
    left_cor_max = max(df[['LeftCorAzz','LeftUncAzz']].max())
    left_y_min_zoom = left_cor_min * 0.9975
    left_y_max_zoom = left_cor_max * 1.0025
else:
    left_cor_azz_tune = df['LeftUncAzz'].mean()
    left_y_min_zoom = left_cor_azz_tune * 0.975
    left_y_max_zoom = left_cor_azz_tune * 1.025

if setpnt is not None:
    row_index = df.index[df['MagPoleTip'] == setpnt][0]
    rght_cor_azz_tune = df['RghtCorAzz'][row_index]
    rght_cor_min = min(df[['RghtCorAzz','RghtUncAzz']].min())
    rght_cor_max = max(df[['RghtCorAzz','RghtUncAzz']].max())
    rght_y_min_zoom = rght_cor_min * 0.9975
    rght_y_max_zoom = rght_cor_max * 1.0025
else:
    rght_cor_azz_tune = df['RghtUncAzz'].mean()
    rght_y_min_zoom = rght_cor_azz_tune * 0.975
    rght_y_max_zoom = rght_cor_azz_tune * 1.025

rate_max_values = {}
rate_max_positions = {}
for label in ['Coin', 'Left', 'Rght']:
    cs = CubicSpline(df[horizontal_axis], df[f'{label}NormRt'])
    x_smooth = np.linspace(df[horizontal_axis].min(), df[horizontal_axis].max(), 1000)
    y_smooth = cs(x_smooth)
    rate_max_values[label] = max(y_smooth)
    rate_max_positions[label] = x_smooth[np.argmax(y_smooth)]
rate_max = np.ceil(max(df[['CoinNormRt', 'LeftNormRt', 'RghtNormRt']].max()) / 5.) * 5.

# Function to apply consistent grid styling
def apply_grid(ax1):
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

def plot_with_error_test(ax1, ax2, df, ylabel, title, setpnt=None):
    x = df[horizontal_axis]
    unc_azz = df[f'{ylabel}UncAzz']
    cor_azz = df[f'{ylabel}CorAzz']
    cor_err = df[f'{ylabel}CorErr']
    norm_rt = df[f'{ylabel}NormRt']
    cor_azz_smooth = df[f'{ylabel}CorAzzSmooth']
    cor_azz_smooth_err = df[f'{ylabel}CorErr']
    ax1.set_title(title)
    axis_label = 'MagPoleTip [T]' if horizontal_axis == 'MagPoleTip' else 'MagCurrent [A]'
    ax1.set_xlabel(axis_label)
    x_smooth = np.linspace(min(x), max(x), 300)
    ax1.fill_between(x, cor_azz_smooth - cor_azz_smooth_err, cor_azz_smooth + cor_azz_smooth_err, color='peachpuff')
    cs_smooth = CubicSpline(x, cor_azz_smooth)
    ax1.plot(x_smooth, cs_smooth(x_smooth), label=f'{ylabel}CorAzzSmoothed', color='darkorange', linestyle='solid', linewidth=1.2)
    ax1.plot(x, unc_azz, label=f'{ylabel}UncAzz', color='mediumblue', linewidth=1, linestyle='solid')
    ax1.errorbar(x, cor_azz, yerr=cor_err, fmt='o', label=f'{ylabel}CorAzz', color='firebrick', capsize=3, markersize=3)
    x_smooth = np.linspace(min(x), max(x), 300)
    cs = CubicSpline(x, norm_rt)
    ax2.plot(x_smooth, cs(x_smooth), color='darkslategray', linestyle='solid', linewidth=1.0, label=f'{ylabel} Norm Rate')
    ax2.scatter(x, norm_rt, color='darkslategray', marker='.', s=20)
    ax2.scatter(rate_max_positions[ylabel], rate_max_values[ylabel], color='yellow', edgecolors='darkslategray', 
                marker="v", s=40, linewidth=0.5, zorder=3, label=f'{ylabel} Rate Max')

    if setpnt is not None:
        setpnt_value = df[horizontal_axis][df['MagPoleTip'] == setpnt].values[0]
        setpnt_cor_azz_smooth = df[f'{ylabel}CorAzzSmooth'][df['MagPoleTip'] == setpnt].values[0]
        setpnt_norm_rt = df[f'{ylabel}NormRt'][df['MagPoleTip'] == setpnt].values[0]
        # Add the vertical dashed line at the correct setpoint MagCurrent position
        ax1.axvline(x=setpnt_value, color='red', linestyle='solid', linewidth=0.5, label='Tune Set Value')
        ax1.scatter(setpnt_value, setpnt_cor_azz_smooth, color='darkorange', edgecolors='black',
                    marker='*', s=90, linewidth=0.5, zorder=5)
        ax2.scatter(setpnt_value, setpnt_norm_rt, color='yellow', edgecolors='black',
                    marker='*', s=90, linewidth=0.5, zorder=5)
        # Draw lines and add text labels for azz and rate set values on axes
        ax1.hlines(setpnt_cor_azz_smooth, x_min, setpnt_value, linestyle='solid', colors='darkorange', linestyles='solid', linewidth=0.5)
        ax2.hlines(setpnt_norm_rt, setpnt_value, x_max, linestyle='solid', colors='darkslategray', linestyles='solid', linewidth=0.5)
        ax1.text(x_min - 0.0125*(x_max-x_min), setpnt_cor_azz_smooth, 
                 f"{setpnt_cor_azz_smooth:.5f}", color='darkorange', fontsize=7,
                 verticalalignment='center', horizontalalignment='right')
        ax2.text(x_max + 0.0125*(x_max-x_min), setpnt_norm_rt, 
                 f"{setpnt_norm_rt:.1f}", color='darkslategray', fontsize=7,
                 verticalalignment='center', horizontalalignment='left')       
        # Add text labels at vertical line for rate max current and set current
        tune_current = setpnt_value + 0.025*(x_max-x_min)
        ratemax_place = setpnt_value - 0.025*(x_max-x_min)
        #ax1.scatter(setpnt_value, df[f'{ylabel}CorAzz'][df['MagPoleTip'] == setpnt], 
        #            color='red', edgecolors='black', marker='*', s=80, zorder=3)
        #ax2.scatter(setpnt_value, df[f'{ylabel}NormRt'][df['MagPoleTip'] == setpnt], 
        #            color='red', edgecolors='black', marker='*', s=80, zorder=3)
        ax2.text(tune_current, 0+0.02*rate_max, f"{setpnt_value:.2f} Amps", 
                 color='red', fontsize=10, verticalalignment='bottom', 
                 bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))
        ax2.text(setpnt_value - 0.025*(x_max-x_min), 0+0.02*rate_max, 
                 f"{rate_max_positions[ylabel]:.2f} Amps", 
                 color='magenta', fontsize=10, verticalalignment='bottom',horizontalalignment='right',
                 bbox=dict(facecolor='white', edgecolor='magenta', boxstyle='round,pad=0.3'))

    ax1.set_ylabel(f'{ylabel}UncAzz & {ylabel}CorAzz', labelpad=4)
    ax2.set_ylabel(f'{ylabel}NormRt (% of Generated)', labelpad=7)
    ax2.set_ylim(0, rate_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)
    # Move all ticks to the inside of the plot
    ax1.tick_params(axis='both', direction='in', length=4, width=0.8)
    ax2.tick_params(axis='both', direction='in', length=4, width=0.8)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=8.5)
    apply_grid(ax1)
  
# Plot 1: Coin data
ax1 = axs[0, 0]
ax2 = ax1.twinx()
plot_with_error_test(ax1, ax2, df, 'Coin', 'Coincidence Hits', setpnt)

# Plot 4: Coin data zoom
ax1 = axs[1, 0]
ax2 = ax1.twinx()
plot_with_error_test(ax1, ax2, df, 'Coin', 'Coincidence Hits (Zoomed)', setpnt)
ax1.set_ylim(coin_y_min_zoom, coin_y_max_zoom)
# Set up tick locations and format labels for left vertical axis
ax1.set_yticks(np.linspace(coin_y_min_zoom, coin_y_max_zoom, 6))
ax1.set_yticklabels([f'{tick:.5f}' for tick in np.linspace(coin_y_min_zoom, coin_y_max_zoom, 6)])

# Plot 2: Left data
ax1 = axs[0, 1]
ax2 = ax1.twinx()
plot_with_error_test(ax1, ax2, df, 'Left', 'Left Calorimeter Hits', setpnt)

# Plot 5: Left data zoom
ax1 = axs[1, 1]
ax2 = ax1.twinx()
plot_with_error_test(ax1, ax2, df, 'Left', 'Left Calorimeter Hits (Zoomed)', setpnt)
ax1.set_ylim(left_y_min_zoom, left_y_max_zoom)
ax1.set_yticks(np.linspace(left_y_min_zoom, left_y_max_zoom, 6))
ax1.set_yticklabels([f'{tick:.5f}' for tick in np.linspace(left_y_min_zoom, left_y_max_zoom, 6)])

# Plot 3: Right data
ax1 = axs[0, 2]
ax2 = ax1.twinx()
plot_with_error_test(ax1, ax2, df, 'Rght', 'Right Calorimeter Hits', setpnt)

# Plot 6: Right data zoom
ax1 = axs[1, 2]
ax2 = ax1.twinx()
plot_with_error_test(ax1, ax2, df, 'Rght', 'Rght Calorimeter Hits (Zoomed)', setpnt)
ax1.set_ylim(rght_y_min_zoom, rght_y_max_zoom)
ax1.set_yticks(np.linspace(rght_y_min_zoom, rght_y_max_zoom, 6))
ax1.set_yticklabels([f'{tick:.5f}' for tick in np.linspace(rght_y_min_zoom, rght_y_max_zoom, 6)])

############################################################
## CREATE PDF TABLE                                        #
############################################################

# TODO: The intent was to use this one key:value pair but seem to have issues
headerColors = {'Coin':'cornflowerblue','Left':'turquoise','Rght':'orchid'}

cellWidth=24
cellHeight=4.5

from scipy.optimize import curve_fit

def linear_fit(x, a, b):
    return a * x + b

# Create a single PDF for all tables
pdf = FPDF(orientation='L')
pdf.set_auto_page_break(auto=True, margin=10)
pdf.set_font("Arial", size=10)

for i, label in enumerate(['Coin', 'Left', 'Rght']):
    pdf.set_fill_color(*{
        'Coin': (100, 149, 237),  # Cornflower Blue
        'Left': (64, 224, 208),  # Turquoise
        'Rght': (218, 112, 214)  # Orchid
    }[label])
    if f'{label}AzzSlope' not in df.columns:
        df[f'{label}AzzSlope']  = -314159.0
        df[f'{label}AzzSlopeE'] = -314159.0
        df[f'{label}AzzSlopeX'] = -314159.0
        df[f'{label}Sensitivity'] = -314159.0
        
df.columns = df.columns.str.strip()

# Calculate CoinAzzSlope, CoinAzzSlopeE, and CoinAzzSlopeX
for label in ['Coin', 'Left', 'Rght']:
    for i in range(2, len(df) - 2):
        subset = df.iloc[i-2:i+3] 
        if len(subset) == 5:
            x_values = subset['MagCurrent'].values
            y_values = subset[f'{label}CorAzz'].values
            y_errors = subset[f'{label}CorErr'].values
            popt, pcov = curve_fit(linear_fit, x_values, y_values, sigma=y_errors, absolute_sigma=True)
            a, b = popt
            a_err = np.sqrt(pcov[0, 0])
            residuals = y_values - linear_fit(x_values, a, b)
            chi2 = np.sum((residuals / y_errors) ** 2)
            ndf = len(x_values) - 2
            df.at[i, f'{label}AzzSlope'] = a
            df.at[i, f'{label}AzzSlopeE'] = a_err
            df.at[i, f'{label}AzzSlopeX'] = chi2 / ndf
            
    # Calculate Sensitivity here with a local gradient using numpy
    interp_func = interp1d(df['MagCurrent'], df[f'{label}CorAzzSmooth'], kind='cubic', fill_value='extrapolate')
    df[f'{label}Sensitivity'] = np.gradient(interp_func(df['MagCurrent']), df['MagCurrent']) / df[f'{label}CorAzzSmooth'] * 100
    df_selected = df[['MagPoleTip', 'MagCurrent', 
                  f'{label}UncAzz', f'{label}CorAzz', f'{label}CorAzzSmooth', f'{label}CorErr', 
                  f'{label}NormRt', f'{label}AzzSlope', f'{label}AzzSlopeE', f'{label}AzzSlopeX', f'{label}Sensitivity']]

    pdf.add_page()
    pdf.set_font("Arial", size=8)
    pdf.set_fill_color(*{
        'Coin': (100, 149, 237),  # Cornflower Blue
        'Left': (64, 224, 208),   # Turquoise
        'Rght': (218, 112, 214)   # Orchid
    }[label])
    
    # Add metadata information
    pdf.set_font("Arial", style='B', size=12)
#    desc_beamE = f"{', '.join([metadata.get('DESC', 'N/A'), metadata.get('beamE', 'N/A'), f'Magnet: {magnet_name} Scan'])}"
    pdf.cell(0, 5, desc_beamE, ln=True, align='L')
    pdf.ln(1)

#    q_values = ', '.join([f"Q{i}: {metadata[f'Q{i}']} T" for i in range(1, 6) if f'Q{i}' in metadata and i != magnet])
    pdf.cell(0, 5, q_values, ln=True, align='L')
    pdf.ln(2)
    pdf.set_font("Arial", style='B', size=12)
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(0, 5, f"{label} Data Table", ln=True, align='L')
    pdf.set_font("Arial", size=10)
    pdf.ln(2)
    
    # Add additional header row
    pdf.set_font("Arial", style='B', size=9)
    pdf.set_fill_color(*{
        'Coin': (100, 149, 237),  # Cornflower Blue
        'Left': (64, 224, 208),   # Turquoise
        'Rght': (218, 112, 214)   # Orchid
    }[label])
    # Merge first six cells for 'Simulation Data',Merge next three cells for 'Fit Data'
    pdf.cell(cellWidth * 7, cellHeight, "Simulation Data", border=1, fill=True, align='C')
    pdf.cell(cellWidth * 4, cellHeight, "Fit Data & Calc", border=1, fill=True, align='C')
    
    pdf.ln()

    # Add table header
    for col in df_selected.columns:
        # Remove 'Coin', 'Left', 'Rght' from column names
        col_label = col[len(label):] if col.startswith(label) else col
        if col.endswith('AzzSlope') or col.endswith('AzzSlopeE') or col.endswith('AzzSlopeX') or col.endswith('Sensitivity'):
            col_label += {f'{label}AzzSlope': ' ¹', f'{label}AzzSlopeE': ' ²', f'{label}AzzSlopeX': '2 ³', f'{label}Sensitivity':' *'}[col]  # Add superscript footnote
        pdf.cell(cellWidth, cellHeight, col_label, border=1, fill=True, align='C')
    pdf.ln()

    for index, row in df_selected.iterrows():
        shading_colors = {
        # Not an elegant solution to problem with label but it works. TODO: Make better
            'Coin': {'MagCurrent': (240, 244, 253), 'CorAzzSmooth': (240, 244, 253), f'{label}CorAzzSmooth': (240, 244, 253)},
            'Left': {'MagCurrent': (236, 252, 250), 'CorAzzSmooth': (236, 252, 250), f'{label}CorAzzSmooth': (236, 252, 250)},
            'Rght': {'MagCurrent': (251, 241, 251), 'CorAzzSmooth': (251, 241, 251), f'{label}CorAzzSmooth': (251, 241, 251)}
        }
        row_formatted = [
        f"{row['MagPoleTip']:.3f}",
        f"{row['MagCurrent']:.3f}",
        f"{row[f'{label}UncAzz']:.6f}",
        f"{row[f'{label}CorAzz']:.6f}",
        f"{row[f'{label}CorAzzSmooth']:.6f}",
        f"{row[f'{label}CorErr']:.6f}",
        f"{row[f'{label}NormRt']:.2f}",
        "******" if row[f'{label}AzzSlope'] == -314159.0 else f"{row[f'{label}AzzSlope']:.6f}",
        "******" if row[f'{label}AzzSlopeE'] == -314159.0 else f"{row[f'{label}AzzSlopeE']:.6f}",
        "******" if row[f'{label}AzzSlopeX'] == -314159.0 else f"{row[f'{label}AzzSlopeX']:.2f}",
        "******" if row[f'{label}Sensitivity'] == -314159.0 else f"{row[f'{label}Sensitivity']:.6f}"
        ]

        pdf.set_font("Arial", style='B' if row['MagPoleTip'] == setpnt else '', size=8)
        # Not a python expert. Having a problem getting this to use the key:value pairs declared at beginning of plot section.
        for col, value in zip(df_selected.columns, row_formatted):
            if row['MagPoleTip'] == setpnt:
                pdf.set_fill_color(*{
                    'Coin': (100, 149, 237),  # Cornflower Blue
                    'Left': (64, 224, 208),   # Turquoise
                    'Rght': (218, 112, 214)   # Orchid
                }[label])
            elif col.endswith('CorAzzSmooth') or col in shading_colors[label]:
                pdf.set_fill_color(*shading_colors[label][col])
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.cell(cellWidth, cellHeight, value, border=1, align='C', fill=True)
        pdf.ln()  
      
    pdf.ln(2)
    reference_note = (
        "¹ dAzz/dA slope derived from linear fit over data point +/- 2 points of CorAzzSmooth.\n"
        "² Error on the first-order parameter from the linear fit.\n"
        "³ Chi2/NDF value of the linear fit for quality assessment.\n"
        "* Relative sensitivity calculated using SciPy interp1d of CorAzzSmooth to MagCurrent "
    )
    pdf.multi_cell(0, 5, reference_note)

##########################################################################
# SAVE OUTPUTS                                                           #
##########################################################################

# Save the plot (maybe move to end where combined pdf is constructed (TODO:???)
plots_filename = f"plot_output_{energy}_{magnet_name}.pdf"
plt.savefig( plots_filename )  # Uncomment this line to save the plot as a PDF
# Save the tables
tables_filename = f"output_table_{energy}_{magnet_name}_combined.pdf"
pdf.output(tables_filename)
# Merge the PDFs
merged_filename = f"MolPol_summary_{energy}_{magnet_name}_plots_tables.pdf"
merger = PdfMerger()
merger.append(plots_filename)
merger.append(tables_filename)
merger.write(merged_filename)
merger.close()

print(f"Huzzah!!!! Merged PDF saved as: {merged_filename}")
