import numpy as np
import pandas as pd
import math
import os
import random
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import norm
from scipy.stats import linregress

def read_and_quantize(file_path):
    data = pd.read_csv(file_path, usecols=[4]).iloc[:, 0].dropna().values
    xmin, xmax = data.min(), data.max()
    if xmax == xmin:
        raise ValueError("Normalization not available")
    norm_data = (data - xmin) / (xmax - xmin)
    quantized = np.round(norm_data * 255).astype(int)
    return np.clip(quantized, 0, 255), norm_data

def calculate_hmin(symbols):
    counts = np.bincount(symbols, minlength=256)
    p_max = counts.max() / len(symbols)
    return -math.log2(p_max)

def compute_avg_hmin(symbols, segment_length, samples=10):
    N = len(symbols)
    if N < segment_length:
        raise ValueError(f"Sequence length {N} is less than segment length {segment_length}")
    h_vals = []
    for _ in range(samples):
        start = random.randint(0, N - segment_length)
        seg = symbols[start:start + segment_length]
        h_vals.append(calculate_hmin(seg))
    return sum(h_vals) / samples

def lyapunov_exponent_original(signal, emb_dim, tau, window_len):
    N = len(signal)
    min_req = window_len + emb_dim * tau
    if N < min_req:
        raise ValueError(f"signal length {N} < required {min_req} (window={window_len}, m={emb_dim}, τ={tau})")

    x = signal[:window_len]
    m = emb_dim
    M = window_len - (m - 1) * tau
    X = np.array([x[i : i + m*tau : tau] for i in range(M)])

    D = squareform(pdist(X))
    eps_time = m * tau
    nearest = np.full(M, -1, dtype=int)
    for i in range(M):
        lo = max(0, i - eps_time)
        hi = min(M, i + eps_time + 1)
        row = D[i].copy()
        row[lo:hi] = np.inf
        nearest[i] = np.argmin(row)

    d0 = np.array([D[i, nearest[i]] for i in range(M)])
    valid_i = np.where(d0 > 0)[0] 
    if len(valid_i) == 0:
        raise ValueError("All initial distance equals to 0")

    t_max = min(100, M - np.max(nearest))
    if t_max < 2:
        raise ValueError("Too few available steps")

    y_curve = np.zeros(t_max)
    counts = np.zeros(t_max, dtype=int)

    for i in valid_i:
        j = nearest[i]
        for t in range(1, t_max + 1):
            if i + t >= M or j + t >= M:
                break
            d_it = norm(X[i + t] - X[j + t])
            if d_it <= 0:
                continue  
            y_curve[t - 1] += math.log(d_it / d0[i])
            counts[t - 1] += 1

    t_axis = []
    y_vals = []
    for idx in range(t_max):
        if counts[idx] > 0:
            t_axis.append(idx + 1)
            y_vals.append(y_curve[idx] / counts[idx])

    if len(t_axis) < 2:
        raise ValueError("Too few available steps")

    t_axis = np.array(t_axis)
    y_vals = np.array(y_vals)

    slope, _, _, _, _ = linregress(t_axis, y_vals)
    return slope

def compute_le_with_fallback(signal):
    param_list = [
        (8, 2, 10000),
        (8, 2, 5000),
        (8, 2, 2000),
        (6, 2, 10000),
        (6, 2, 5000),
        (6, 2, 2000),
        (8, 4, 10000),
        (8, 4, 5000),
        (8, 4, 2000),
    ]

    last_error = None
    for (m, tau, wlen) in param_list:
        try:
            le = lyapunov_exponent_original(signal, emb_dim=m, tau=tau, window_len=wlen)
            return le, m, tau, wlen
        except Exception as e:
            last_error = e
            continue

    print(f"[Warning] 所有尝试均失败，signal length={len(signal)}，最后一次错误：{last_error}")
    return np.nan, None, None, None

if __name__ == "__main__":
    file_paths = ['Demo: File_Path.csv']

    results = []
    for fp in file_paths:
        fname = os.path.basename(fp)
        try:
            symbols, norm_data = read_and_quantize(fp)
            avg_1k   = compute_avg_hmin(symbols, 1_000,    samples=10)
            avg_10k  = compute_avg_hmin(symbols, 10_000,   samples=10)
            avg_100k = compute_avg_hmin(symbols, 100_000,  samples=10)
            avg_1M   = compute_avg_hmin(symbols, 1_000_000, samples=10)
            h_full   = calculate_hmin(symbols)

            le, used_m, used_tau, used_wlen = compute_le_with_fallback(norm_data)

            results.append({
                'filename':      fname,
                'avg_hmin_1k':   avg_1k,
                'avg_hmin_10k':  avg_10k,
                'avg_hmin_100k': avg_100k,
                'avg_hmin_1M':   avg_1M,
                'hmin_full':     h_full,
                'lyapunov_exp':  le,
                'emb_dim':       used_m,
                'tau':           used_tau,
                'window_len':    used_wlen
            })

        except Exception as e:
            print(f"[Error] file {fname} process failed：{e}")
            results.append({
                'filename':      fname,
                'avg_hmin_1k':   np.nan,
                'avg_hmin_10k':  np.nan,
                'avg_hmin_100k': np.nan,
                'avg_hmin_1M':   np.nan,
                'hmin_full':     np.nan,
                'lyapunov_exp':  np.nan,
                'emb_dim':       None,
                'tau':           None,
                'window_len':    None
            })

    df = pd.DataFrame(results, columns=[
        'filename',
        'avg_hmin_1k',
        'avg_hmin_10k',
        'avg_hmin_100k',
        'avg_hmin_1M',
        'hmin_full',
        'lyapunov_exp',
        'emb_dim',
        'tau',
        'window_len'
    ])
    df.to_csv('min_entropy_summary_with_le.csv', index=False)
