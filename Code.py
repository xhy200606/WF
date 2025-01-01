import pandas as pd
import os
import numpy as np
from scipy.optimize import linprog

folder_path = r'Z:\工作\论文\水足迹\数据\优化\地区'
years = list(range(2002, 2023))
crops = ['Rice', 'Maize', 'Peanut', 'Cotton', 'Wheat', 'Rape']

def extract_stats(data):
    return {
        'am1': data.quantile(0.25),
        'am2': data.mean(),
        'am3': data.quantile(0.75),
        'am4': data.min(),
        'am5': data.max()
    }

def calculate_membership_functions(am1, am2, am3, am4, am5):
    mu_m = (am1 + 2 * am2 + am3) / 4
    nu_m = (am4 + 2 * am2 + am5) / 4
    return mu_m, nu_m

def calculate_triangular_fuzzy_number(mu_m, nu_m):
    T_m = (mu_m + nu_m) / 2
    return T_m

WF_blue = np.zeros((21, len(crops)))
S_j = np.zeros(len(crops))
P_j = np.zeros(len(crops))

eco_df = pd.read_csv(os.path.join(folder_path, 'Eco.csv'))

for i in range(1, 22):
    file_name = f'Z{i}.csv'
    file_path = os.path.join(folder_path, file_name)
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        
        if 'Year' not in df.columns:
            df.rename(columns={'年份': 'Year'}, inplace=True)
        
        for idx, crop in enumerate(crops):
            col_name_b = f'{crop}_B'
            col_name_s = f'{crop}_S'
            col_name_p = f'{crop}_P'
            
            if col_name_b in df.columns:
                crop_data_b = df[col_name_b]
                stats_b = extract_stats(crop_data_b)
                mu_b, nu_b = calculate_membership_functions(stats_b['am1'], stats_b['am2'], stats_b['am3'], stats_b['am4'], stats_b['am5'])
                WF_blue[i-1, idx] = calculate_triangular_fuzzy_number(mu_b, nu_b)

            if col_name_s in eco_df.columns:
                crop_data_s = eco_df[col_name_s]
                stats_s = extract_stats(crop_data_s)
                mu_s, nu_s = calculate_membership_functions(stats_s['am1'], stats_s['am2'], stats_s['am3'], stats_s['am4'], stats_s['am5'])
                S_j[idx] = calculate_triangular_fuzzy_number(mu_s, nu_s)

            if col_name_p in eco_df.columns:
                crop_data_p = eco_df[col_name_p]
                stats_p = extract_stats(crop_data_p)
                mu_p, nu_p = calculate_membership_functions(stats_p['am1'], stats_p['am2'], stats_p['am3'], stats_p['am4'], stats_p['am5'])
                P_j[idx] = calculate_triangular_fuzzy_number(mu_p, nu_p)

Y_j = np.random.rand(6)
MD_j = np.random.rand(6)
A_i = np.random.rand(21)
eta_i = np.random.rand(21)

f1 = WF_blue.flatten()
f2 = (S_j - P_j).flatten()
f3 = np.random.rand(21 * 6)

objective = np.concatenate([f1, -f2, f3])

A_eq_1 = np.zeros((1, 126))
for i in range(21):
    for j in range(6):
        if j < 4:
            Q = 80 * 10**8
        else:
            Q = 110 * 10**8
        
        A_eq_1[0, i*6+j] = WF_blue[i, j] * Y_j[j] / eta_i[i]

b_eq_1 = [Q]

A_eq_2 = np.zeros((21, 126))
for i in range(21):
    for j in range(6):
        A_eq_2[i, i*6+j] = 1

b_eq_2 = A_i

bounds = [(0, None)] * 126

A_ineq = np.zeros((6, 126))
for j in range(6):
    for i in range(21):
        A_ineq[j, i*6+j] = Y_j[j]

b_ineq = MD_j

A = np.vstack([A_eq_1, A_eq_2, A_ineq])
b = np.concatenate([b_eq_1, b_eq_2, b_ineq])

result = linprog(c=objective, A_ub=None, b_ub=None, A_eq=A, b_eq=b, bounds=bounds, method='simplex')

if result.success:
    planting_areas = result.x.reshape((21, 6))
    print("Optimal planting areas (in hectares):")
    print(planting_areas)
else:
    print("Linear programming failed:", result.message)

blue_water_df = pd.DataFrame(WF_blue, columns=crops, index=[f'Z{i}' for i in range(1, 22)])
blue_water_df.to_csv(os.path.join(folder_path, 'blue_water_fuzzy_numbers.csv'))
