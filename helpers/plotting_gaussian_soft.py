import torch
import matplotlib.pyplot as plt

def gaussian_soft_targets(true_bin, num_bins=201, temperature=2.0):
    """
    true_bin: integer index (0..num_bins-1)
    """
    device = "cpu"
    true_bin = torch.tensor([true_bin], device=device)

    indices = torch.arange(num_bins, device=device).unsqueeze(0)
    true_exp = true_bin.unsqueeze(1).float()

    dist = torch.abs(indices - true_exp)
    soft = torch.exp(-dist.pow(2) / (2 * (temperature ** 2)))
    soft = soft / soft.sum(dim=1, keepdim=True)
    return soft.squeeze(0).cpu().numpy()


def plot_soft_targets(true_bin=100, num_bins=201, temps=[0.5, 1.0, 2.0, 4.0]):
    """
    Plot how the Gaussian soft targets vary with temperature.
    """
    plt.figure(figsize=(12, 6))
    x = list(range(num_bins))

    for T in temps:
        y = gaussian_soft_targets(true_bin=true_bin,
                                  num_bins=num_bins,
                                  temperature=T)
        plt.plot(x, y, label=f"Temperature={T}")

    plt.title(f"Gaussian Soft Targets (true_bin={true_bin})")
    plt.xlabel("Bin index")
    plt.ylabel("Soft target probability")
    plt.legend()
    plt.grid(False)
    plt.show()


if __name__ == "__main__":
    # Change these params if you want
    plot_soft_targets(
        true_bin=100,          # center position
        num_bins=201,
        temps=[2.0, 3.0, 4.0, 5.0, 10.0]   # multiple temperatures
    )
