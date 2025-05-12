import numpy as np
from scipy import special
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point

# Parameters
c = 1.0  # Speed of light
mu1 = 100  # Spacetime point density
mu2 = 50   # Defect point density
M = 5.0   # Mass of particle
digits = 8  # Computation precision
A = 0.5   # ST->ST amplitude
B = -(M**2) / (mu1 + mu2/2)  # Stop amplitude

# Define polygon region
poly_points = [(1, 0), (1 + np.sqrt(2)/2, np.sqrt(2)/2), (1, np.sqrt(2)), 
               (1 - np.sqrt(2)/2, np.sqrt(2)/2)]
region = Polygon(poly_points)

# Function to generate Poisson points in a polygon
def poisson_points_in_polygon(mu, polygon):
    min_x, min_y, max_x, max_y = polygon.bounds
    area = polygon.area
    n_points = np.random.poisson(mu * area)
    x = np.random.uniform(min_x, max_x, n_points)
    y = np.random.uniform(min_y, max_y, n_points)
    points = []
    for xi, yi in zip(x, y):
        if polygon.contains(Point(xi, yi)):
            points.append([xi, yi])
    return np.array(points)

# Generate spacetime and defect points
pts1 = poisson_points_in_polygon(mu1, region)
pts2 = poisson_points_in_polygon(mu2, region)

# Construct "in" and "out" sets
d = pts2
if len(d) % 2 != 0:
    d = d[:-1]
pairs = []
remaining = d.tolist()
while len(remaining) >= 2:
    pair = np.random.choice(len(remaining), 2, replace=False)
    pairs.append([remaining[pair[0]], remaining[pair[1]]])
    remaining = [p for i, p in enumerate(remaining) if i not in pair]
pairs = np.array(pairs)

# Classify pairs as timelike (TPairs) or spacelike (SPairs)
TPairs, SPairs = [], []
for pair in pairs:
    dx = pair[0][0] - pair[1][0]
    dy = pair[0][1] - pair[1][1]
    a = c**2 * dy**2 - dx**2
    if a >= 0:
        TPairs.append(pair)
    else:
        SPairs.append(pair)
TPairs, SPairs = np.array(TPairs), np.array(SPairs)

# Assign Tin, Tout, sin, Sout
Tin, Tout, sin, Sout = [], [], [], []
for pair in TPairs:
    if pair[0][1] < pair[1][1]:
        Tin.append(pair[0])
        Tout.append(pair[1])
    else:
        Tin.append(pair[1])
        Tout.append(pair[0])
for pair in SPairs:
    if pair[0][1] < pair[1][1]:
        sin.append(pair[0])
        Sout.append(pair[1])
    else:
        sin.append(pair[1])
        Sout.append(pair[0])
Tin, Tout = np.array(Tin), np.array(Tout)
inset, outset = Tin, Tout

# Construct point set P and causal matrix
P = np.vstack([pts1, outset, inset])
b = len(P)
q = len(pts1)
g = len(outset)
matC1 = np.zeros((b, b))
for i in range(b):
    for j in range(b):
        dx = P[i][0] - P[j][0]
        dy = P[i][1] - P[j][1]
        a = c**2 * dy**2 - dx**2
        w = P[j][1] - P[i][1]
        if a >= 0:
            matC1[i, j] = 1 if w >= 0 else -1
ST1 = 0.5 * (matC1 + np.abs(matC1)) - np.eye(b)

# Function to construct ST2 based on model
def construct_ST2(ST1, model, q, g, b, A):
    ST2 = ST1.copy()
    
    if model in ['A', 'B']:
        kappa = 0.0
        epsilon = 0.75 if model == 'A' else 1.0
        Swap = np.eye(g)
        ST2[q:q+g, q:q+g] = np.zeros((g, g))
        ST2[0:q, q:q+g] = np.zeros((q, g))
        ST2[q+g:b, 0:q] = (kappa / A) * ST1[q+g:b, 0:q]
        ST2[q+g:b, q+g:b] = (kappa / A) * ST1[q+g:b, q+g:b]
        ST2[q+g:b, q:q+g] = (epsilon / A) * Swap
    
    elif model == 'C':
        epsilon = 0.55
        Swap = ST1[q+g:b, q:q+g].copy()
        for i in range(g):
            Swap[i, i] = epsilon / A
        ST2[q+g:b, q:q+g] = Swap
    
    elif model == 'D':
        xi = 0.1
        epsilon = 0.5
        ST2[q+g:b, 0:q+g] = (xi / A) * ST1[q+g:b, 0:q+g]
        ST2[0:q+g, q+g:b] = (xi / A) * ST1[0:q+g, q+g:b]
        ST2[q+g:b, q+g:b] = (epsilon / A) * ST2[q+g:b, q+g:b]
    
    elif model == 'E':
        kappa = 0.002
        ST2[q+g:b, 0:b] = (kappa / A) * ST1[q+g:b, 0:b]
    
    else:
        raise ValueError("Invalid model. Choose 'A', 'B', 'C', 'D', or 'E'.")
    
    return ST2

# Construct ST2 for a specific model
model = 'A'  # Change to 'B', 'C', 'D', or 'E' as needed
ST2 = construct_ST2(ST1, model, q, g, b, A)

# To run multiple models, uncomment the following loop:
# for model in ['A', 'B', 'C', 'D', 'E']:
#     print(f"Running Model {model}")
#     ST2 = construct_ST2(ST1, model, q, g, b, A)
#     # ... (insert rest of the code here, adjust plot titles with model)

# Compute propagators
Phi = A * ST2
GR = Phi @ np.linalg.inv(np.eye(b) - B * Phi)
PJ = GR - GR.T
GFR = 0.5 * (GR + GR.T)

# Compute imaginary part of Feynman propagator
eValues, eVectors = np.linalg.eig(1j * PJ)
nonZero = []
for i, val in enumerate(eValues):
    if val.real > 0:
        vec = eVectors[:, i]
        nonZero.append(val.real * np.outer(vec, vec.conj()).real)
GFI = sum(nonZero)

# Define fit functions and spacetime interval
def fr(W, m, x):
    return (W/4) * special.hankel2(0, m*x/c).imag
def fr2(W, m, x):
    return (W/4) * special.hankel2(0, m*x/c).real
def fi(W, m, x):
    return (W/4) * special.hankel2(0, -1j * m * x/c).imag
def fi2(W, m, x):
    return (W/4) * special.hankel2(0, -1j * m * x/c).real
def T(x, y):
    return np.abs((1/c) * np.sqrt(c**2 * (x[1] - y[1])**2 - (x[0] - y[0])**2))

# Average curve parameters
delta = 0.01
binlength = int(1.4 / delta)
Slimsize = 500

# Real part of Feynman propagator (timelike)
Pltpts3 = []
for n in range(b):
    for l in range(b):
        if ST1[n, l] == 1:
            Pltpts3.append([T(P[n], P[l]), GFR[n, l]])
Pltpts3 = np.array(Pltpts3)
Avg3 = []
for i in range(binlength):
    bin_vals = Pltpts3[(i*delta <= Pltpts3[:, 0]) & (Pltpts3[:, 0] <= (i+1)*delta), 1]
    if len(bin_vals) > 0:
        Avg3.append([(i*delta) + delta/2, np.mean(bin_vals)])
Avg3 = np.array(Avg3)

# Nonlinear fit for real part
def fit_fr2(x, lambda_, W):
    return fr2(W, lambda_, x)
popt3, pcov3 = curve_fit(fit_fr2, Avg3[:, 0], Avg3[:, 1], p0=[M, 1])
efM3, efW3 = popt3
ErrorM3, ErrorW3 = np.sqrt(np.diag(pcov3))

# Imaginary part of Feynman propagator (timelike)
Pltpts1 = []
for n in range(b):
    for l in range(b):
        if ST1[n, l] == 1:
            Pltpts1.append([T(P[n], P[l]), GFI[n, l]])
Pltpts1 = np.array(Pltpts1)
Avg1 = []
for i in range(binlength):
    bin_vals = Pltpts1[(i*delta <= Pltpts1[:, 0]) & (Pltpts1[:, 0] <= (i+1)*delta), 1]
    if len(bin_vals) > 0:
        Avg1.append([(i*delta) + delta/2, np.mean(bin_vals)])
Avg1 = np.array(Avg1)

# Nonlinear fit for imaginary part
def fit_fr(x, lambda_, W):
    return fr(W, lambda_, x)
popt1, pcov1 = curve_fit(fit_fr, Avg1[:, 0], Avg1[:, 1], p0=[M, 1])
efM1, efW1 = popt1
ErrorM1, ErrorW1 = np.sqrt(np.diag(pcov1))

# Imaginary part of Feynman propagator (spacelike)
Pltpts2 = []
for n in range(b):
    for l in range(b):
        if ST1[n, l] == ST1[l, n] == 0:
            Pltpts2.append([T(P[n] * 1j, P[l] * 1j), GFI[n, l]])
Pltpts2 = np.array(Pltpts2)
Avg2 = []
for i in range(binlength):
    bin_vals = Pltpts2[(i*delta <= Pltpts2[:, 0]) & (Pltpts2[:, 0] <= (i+1)*delta), 1]
    if len(bin_vals) > 0:
        Avg2.append([(i*delta) + delta/2, np.mean(bin_vals)])
Avg2 = np.array(Avg2)

# Nonlinear fit for spacelike imaginary part
def fit_fi(x, lambda_, W):
    return fi(W, lambda_, x)
popt2, pcov2 = curve_fit(fit_fi, Avg2[:, 0], Avg2[:, 1], p0=[M, 1])
efM2, efW2 = popt2
ErrorM2, ErrorW2 = np.sqrt(np.diag(pcov2))

# Compute average fit parameters
efWavg = (efW1 + efW3) / 2
efMavg = (efM1 + efM3) / 2
efWavgEr = 0.5 * np.sqrt(ErrorW1**2 + ErrorW3**2)
efMavgEr = 0.5 * np.sqrt(ErrorM1**2 + ErrorM3**2)

# Plotting
# Real part of Feynman propagator (timelike)
plt.figure()
plt.scatter(Pltpts3[:, 0], Pltpts3[:, 1], c='darkred', marker='D', label='Data')
x = np.linspace(0, 1.4, 100)
plt.plot(x, fr2(1, M, x), 'k--', label='Theoretical')
plt.xlim(0, 1.4)
plt.ylim(-0.35, 0.75)
plt.xlabel('|ds²|^{1/2}')
plt.ylabel('Re[G_F](x-y)')
plt.title(f'Timelike Pairs (Model {model})')
plt.legend()
plt.show()

# Average curve for real part
plt.figure()
plt.scatter(Avg3[:, 0], Avg3[:, 1], c='darkred', marker='D', label='Average Data')
plt.plot(x, fr2(1, M, x), 'k--', label='Theoretical')
plt.xlim(0, 1.4)
plt.ylim(-0.35, 0.45)
plt.xlabel('|ds²|^{1/2}')
plt.ylabel('Average Re[G_F](x-y)')
plt.title(f'Timelike Pairs (Model {model})')
plt.legend()
plt.show()

# Imaginary part of Feynman propagator (timelike)
plt.figure()
plt.scatter(Pltpts1[:, 0], Pltpts1[:, 1], c='darkred', marker='D', label='Data')
plt.plot(x, fr(1, M, x), 'k--', label='Theoretical')
plt.xlim(0, 1.4)
plt.ylim(-0.3, 0.6)
plt.xlabel('|ds²|^{1/2}')
plt.ylabel('Im[G_F](x-y)')
plt.title(f'Timelike Pairs (Model {model})')
plt.legend()
plt.show()

# Average curve for imaginary part
plt.figure()
plt.scatter(Avg1[:, 0], Avg1[:, 1], c='darkred', marker='D', label='Average Data')
plt.plot(x, fr(efWavg, efMavg, x), 'g-', label='Fit')
plt.plot(x, fr(1, M, x), 'k--', label='Theoretical')
plt.xlim(0, 1.4)
plt.ylim(-0.2, 0.3)
plt.xlabel('|ds²|^{1/2}')
plt.ylabel('Average Im[G_F](x-y)')
plt.title(f'Timelike Pairs (Model {model})')
plt.legend()
plt.show()

# Imaginary part of Feynman propagator (spacelike)
plt.figure()
plt.scatter(Pltpts2[:, 0], Pltpts2[:, 1], c='darkred', marker='D', label='Data')
plt.plot(x, fi(efWavg, efMavg, x), 'g-', label='Fit')
plt.plot(x, fi(1, M, x), 'k--', label='Theoretical')
plt.xlim(0, 1.4)
plt.ylim(-0.3, 0.6)
plt.xlabel('|ds²|^{1/2}')
plt.ylabel('Im[G_F](x-y)')
plt.title(f'Spacelike Pairs (Model {model})')
plt.legend()
plt.show()

# Average curve for spacelike imaginary
plt.figure()
plt.scatter(Avg2[:, 0], Avg2[:, 1], c='darkred', marker='D', label='Average Data')
plt.plot(x, fi(efWavg, efMavg, x), 'g-', label='Fit')
plt.plot(x, fi(1, M, x), 'k--', label='Theoretical')
plt.xlim(0, 1.4)
plt.ylim(-0.1, 0.3)
plt.xlabel('|ds²|^{1/2}')
plt.ylabel('Average Im[G_F](x-y)')
plt.title(f'Spacelike Pairs (Model {model})')
plt.legend()
plt.show()

# Spacelike real part (relevant for Model B, but GFR0 is undefined)
Pltpts4 = []
for n in range(b):
    for l in range(b):
        if ST1[n, l] == ST1[l, n] == 0:
            Pltpts4.append([T(P[n] * 1j, P[l] * 1j), GFR[n, l]])
Pltpts4 = np.array(Pltpts4)
Avg4 = []
for i in range(binlength):
    bin_vals = Pltpts4[(i*delta <= Pltpts4[:, 0]) & (Pltpts4[:, 0] <= (i+1)*delta), 1]
    if len(bin_vals) > 0:
        Avg4.append([(i*delta) + delta/2, np.mean(bin_vals)])
Avg4 = np.array(Avg4)

# Average curve for spacelike real part
plt.figure()
plt.scatter(Avg4[:, 0], Avg4[:, 1], c='darkred', marker='D', label='Average Data')
plt.xlim(0, 1.4)
plt.ylim(-0.1, 0.3)
plt.xlabel('|ds²|^{1/2}')
plt.ylabel('Average Re[G_F](x-y)')
plt.title(f'Spacelike Pairs (Model {model})')
plt.legend()
plt.show()