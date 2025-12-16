import numpy as np
from scipy import optimize
from scipy.ndimage import gaussian_filter


# Difference of Gaussians 2D function
def dog_2d(coords, amp1, amp2, x0, y0, sigma1_x, sigma1_y, sigma2_x, sigma2_y, offset, theta=0):
    """Difference of two 2D Gaussians"""
    x, y = coords
    x0 = float(x0)
    y0 = float(y0)
    
    # Rotation matrices for both Gaussians
    a1 = (np.cos(theta)**2)/(2*sigma1_x**2) + (np.sin(theta)**2)/(2*sigma1_y**2)
    b1 = -(np.sin(2*theta))/(4*sigma1_x**2) + (np.sin(2*theta))/(4*sigma1_y**2)
    c1 = (np.sin(theta)**2)/(2*sigma1_x**2) + (np.cos(theta)**2)/(2*sigma1_y**2)
    
    a2 = (np.cos(theta)**2)/(2*sigma2_x**2) + (np.sin(theta)**2)/(2*sigma2_y**2)
    b2 = -(np.sin(2*theta))/(4*sigma2_x**2) + (np.sin(2*theta))/(4*sigma2_y**2)
    c2 = (np.sin(theta)**2)/(2*sigma2_x**2) + (np.cos(theta)**2)/(2*sigma2_y**2)
    
    g1 = amp1 * np.exp(-(a1*(x-x0)**2 + 2*b1*(x-x0)*(y-y0) + c1*(y-y0)**2))
    g2 = amp2 * np.exp(-(a2*(x-x0)**2 + 2*b2*(x-x0)*(y-y0) + c2*(y-y0)**2))
    
    return offset + g1 - g2



def fit_dog_2d(image, blur_sigma=1):
    """Fit a Difference of Gaussians to image data"""
    
    img = gaussian_filter(np.abs(image), sigma=blur_sigma)
    ny, nx = img.shape
    y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    
    # Detect peak or dip
    peak_val = np.max(img)
    dip_val = np.min(img)
    mean_val = np.mean(img)
    
    if abs(peak_val - mean_val) > abs(dip_val - mean_val):
        peak_idx = np.unravel_index(np.argmax(img), img.shape)
        amp1_guess = (peak_val - mean_val) * 2  # Make center stronger
    else:
        peak_idx = np.unravel_index(np.argmin(img), img.shape)
        amp1_guess = (dip_val - mean_val) * 2
    
    peak_y, peak_x = peak_idx
    sigma_small = min(nx, ny) / 8  # Smaller center
    sigma_large = min(nx, ny) / 3  # Larger surround
    
    # Initial guess: [amp1, amp2, x0, y0, sig1_x, sig1_y, sig2_x, sig2_y, offset]
    initial_guess = [
        amp1_guess,           # amplitude of center Gaussian
        amp1_guess * 0.3,     # amplitude of surround (significantly smaller)
        peak_x,               # center x
        peak_y,               # center y
        sigma_small,          # center sigma x
        sigma_small,          # center sigma y
        sigma_large,          # surround sigma x (MUST be larger)
        sigma_large,          # surround sigma y
        mean_val              # offset
    ]
    
    # STRICT Bounds to enforce proper DoG structure
    lower_bounds = [
        -np.inf,              # amp1
        -np.inf,              # amp2
        0,                    # x0
        0,                    # y0
        0.5,                  # sigma1_x (minimum center width)
        0.5,                  # sigma1_y
        sigma_small * 1.5,    # sigma2_x MUST be at least 1.5x larger
        sigma_small * 1.5,    # sigma2_y MUST be at least 1.5x larger
        -np.inf               # offset
    ]
    
    upper_bounds = [
        np.inf,               # amp1
        np.inf,               # amp2
        nx,                   # x0
        ny,                   # y0
        sigma_large * 0.6,    # sigma1_x (center can't be too wide)
        sigma_large * 0.6,    # sigma1_y
        nx,                   # sigma2_x
        ny,                   # sigma2_y
        np.inf                # offset
    ]
    
    try:
        params, _ = optimize.curve_fit(
            lambda coords, *p: dog_2d(coords, *p, theta=0),
            (x.ravel(), y.ravel()),
            img.ravel(),
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000
        )
    except Exception as e:
        print(f"DoG fit failed: {e}")
        print("Trying with relaxed constraints...")
        # Try again with slightly relaxed constraints
        lower_bounds[6] = sigma_small * 1.3  # Relax to 1.3x
        lower_bounds[7] = sigma_small * 1.3
        try:
            params, _ = optimize.curve_fit(
                lambda coords, *p: dog_2d(coords, *p, theta=0),
                (x.ravel(), y.ravel()),
                img.ravel(),
                p0=initial_guess,
                bounds=(lower_bounds, upper_bounds),
                maxfev=15000
            )
        except:
            print("Fit failed even with relaxed constraints, using initial guess")
            params = initial_guess
    
    param_dict = {
        'amp1': params[0],
        'amp2': params[1],
        'x0': params[2],
        'y0': params[3],
        'sigma1_x': params[4],
        'sigma1_y': params[5],
        'sigma2_x': params[6],
        'sigma2_y': params[7],
        'offset': params[8],
        'theta': 0
    }
    
    return param_dict