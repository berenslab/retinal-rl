import matplotlib.pyplot as plt
import os

results_folder = "./training_results"
data_files = os.listdir(results_folder)
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

for file in data_files:
    path = os.path.join(results_folder, file)
    with open(path, "r") as f:
        f.readline()  # Skip the lambda line
        f.readline()  # Skip header
        epochs, recon_losses, class_losses = [], [], []
        for line in f:
            epoch, recon, class_loss = line.strip().split(",")
            epochs.append(int(epoch))
            recon_losses.append(float(recon))
            class_losses.append(float(class_loss))
    label = file.split("_")[-1].replace(".txt", "")  # Extract lambda from filename
    axes[0].plot(epochs, recon_losses, label=f"Lambda={label}")
    axes[1].plot(epochs, class_losses, label=f"Lambda={label}")

axes[0].set_title("Reconstruction Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].set_title("Classification Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(results_folder, "training_plots.png"))
plt.show()
