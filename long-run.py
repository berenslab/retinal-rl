import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.load('sim_recs.npy', allow_pickle=True).tolist()

plcys = data["plcys"].detach().numpy()
hlths = data["hlths"]
uhlths = data["uhlths"]

# Reshape hlths and flatten ltnts for regression
y = hlths

# Average policy values
avg_plcys = plcys.mean(axis=-1)
avg_values = avg_plcys.ravel()  # Flatten the (2, 3) matrix into a single array

labels = ['Centre', 'Left', 'Right', 'Stationary', 'Forward', 'Backward']

fig, ax = plt.subplots(2, 1, figsize=(10, 10))  # 2 rows, 1 column

# Plcys histogram
ax[0].bar(labels, avg_values, color=['blue']*3 + ['red']*3)
ax[0].set_xlabel('Policy Actions')
ax[0].set_ylabel('Average Value')
ax[0].set_title('Average values of plcys')
# Compute the boolean mask for non-zero previous uhlths values
non_reset_mask = uhlths[:-1] != 0

# a. Count jumps in uhlths
jumps = np.diff(uhlths)[non_reset_mask]
hlths = hlths[1:][non_reset_mask]
non_zero_jump_indices = np.where(jumps != 0)[0]
jumps_diffs = jumps[jumps != 0]  # Filter out 0 changes


bins = [-27.5, -22.5, -17.5, -12.5, -7.5, -2.5, 5, 15, 25, 35, 45, 55]
bin_counts, _ = np.histogram(jumps_diffs, bins=bins)

# Plot
ax[1].bar(range(len(bin_counts)), bin_counts, tick_label=[f"{(bins[i+1]+bins[i])/2}" for i in range(len(bins)-1)])
ax[1].set_xlabel("Jump Range")
ax[1].set_ylabel("Count")
ax[1].set_title("Histogram of uhlths jumps")

over_healths = 0

for idx in non_zero_jump_indices:
    jump = jumps[idx]
    if uhlths[idx] != 0 and (hlths[idx-1] + jump - 100) > 0:  # hlths[idx] corresponds to jumps[idx]
        over_healths += (hlths[idx-1] + jump - 100)
        print(idx,hlths[idx],hlths[idx-1], jump, over_healths)

print(f"Amount added to over_healths: {over_healths}")

# Save the figure
plt.tight_layout()
plt.savefig("combined_histograms.png")
# plt.show()
