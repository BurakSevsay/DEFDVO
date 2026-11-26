import matplotlib.pyplot as plt
from evo.core import sync
from evo.tools.settings import SETTINGS
SETTINGS.plot_backend = "Agg"  # evo.tools.plot switches to this backend on import
from evo.tools import plot


def plot_trajectory(pred_traj, gt_traj, title, filename):
    gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)
    pred_traj.align(gt_traj, correct_scale=True)

    plot_collection = plot.PlotCollection("PlotCol")
    fig = plt.figure(figsize=(8, 8))
    plot_mode = plot.PlotMode.xz # ideal for planar movement
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(title)
    plot.traj(ax, plot_mode, gt_traj, '--', 'gray', "Ground Truth")
    plot.traj(ax, plot_mode, pred_traj, '-', 'blue', "Predicted")
    plot_collection.add_figure("traj (error)", fig)
    plot_collection.export(filename, confirm_overwrite=False)
    plt.close(fig=fig)
    print(f"Saved {filename}")
