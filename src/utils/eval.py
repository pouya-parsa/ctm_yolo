import json
import argparse
import numpy as np
from ctm.ctm import CTMLink

# ---------------------------- helpers ----------------------------

def load_gt(path):
    with open(path) as f:
        d = json.load(f)
    T = max(int(k) for k in d)
    up = np.zeros(T)
    down = np.zeros(T)
    for k, (u, dn) in d.items():
        idx = int(k) - 1
        up[idx] = u
        down[idx] = dn
    return up, down


def gt_to_inflow(up_cum):
    # per-step vehicles entering = diff of cumulative
    return np.diff(np.insert(up_cum, 0, 0)).tolist()

# ---------------------------- evaluation ----------------------------

def run_ctm(
    up_cum_gt,
    length: float = 150.0,
    cell_len: float = 30.0,
    dt: float = 1.0,
    v_f: float = 27.0,
    w: float = 5.0,
    k_j: float = 0.15,
    C: float | None = None,
    n_lanes: int = 1,
):
    T = len(up_cum_gt)
    inflow_series = gt_to_inflow(up_cum_gt)

    link = CTMLink(
        length=length,
        cell_length=cell_len,
        dt=dt,
        v_f=v_f,
        w=w,
        k_j=k_j,
        C=C,
        n_lanes=n_lanes,
    )
    link.reset()

    flows = link.run(T, inflow=inflow_series)
    up_flow = flows[:, 0, 0]
    down_flow = flows[:, -1, 1]

    up_mod = np.cumsum(up_flow)
    down_mod = np.cumsum(down_flow)
    return up_mod, down_mod


def compare(up_gt, down_gt, up_mod, down_mod):
    mae_up = np.mean(np.abs(up_gt - up_mod))
    mae_down = np.mean(np.abs(down_gt - down_mod))
    return mae_up, mae_down

def plot_mae_vs_cell_length(
    up_gt: np.ndarray,
    down_gt: np.ndarray,
    length: float,
    dt: float,
    v_f: float,
    w: float,
    k_j: float,
    C: float | None,
    n_lanes: int,
) -> None:
    """
    Sweep cell_length from 35 to length (step 10), compute MAE, and plot.
    """
    import matplotlib.pyplot as plt

    cell_lengths = list(range(30, int(length) + 1, 10))
    mae_up_list = []
    mae_down_list = []

    for cl in cell_lengths:
        up_mod, down_mod = run_ctm(
            up_gt,
            length=length,
            cell_len=cl,
            dt=dt,
            v_f=v_f,
            w=w,
            k_j=k_j,
            C=C,
            n_lanes=n_lanes,
        )
        m_up, m_down = compare(up_gt, down_gt, up_mod, down_mod)
        mae_up_list.append(m_up)
        mae_down_list.append(m_down)
    

    print("mae_up_list: ", mae_up_list)

    # plt.plot(cell_lengths, mae_up_list, label='Upstream MAE', color='blue')
    # plt.plot(cell_lengths, mae_up_list, label='Upstream MAE', color='blue', marker='o')
    plt.plot(cell_lengths, mae_down_list, label='Downstream MAE', color='red', marker='o')
    plt.xlabel('Cell length (m)')
    plt.ylabel('Mean Absolute Error (veh)')
    plt.title('MAE vs Cell Length')
    plt.legend()
    plt.grid(True)
    plt.show()


# ---------------------------- script entry ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_json', help='Ground-truth cumulative counts JSON')
    parser.add_argument('--length',    type=float, default=118.87, help='link length [m]')
    # parser.add_argument('--cell_len',  type=float, default=30.0,  help='cell length [m]')
    parser.add_argument('--dt',        type=float, default=1.0,   help='time step [s]')
    parser.add_argument('--v_f',       type=float, default=110.0,  help='free-flow speed [m/s]')
    parser.add_argument('--w',         type=float, default=55.0,   help='backward wave speed [m/s]')
    parser.add_argument('--k_j',       type=float, default=0.22,  help='jam density [veh/m/lane]')
    parser.add_argument('--C',         type=float, default=2.24,  help='capacity [veh/s/lane]')
    parser.add_argument('--n_lanes',   type=int,   default=4,     help='number of lanes')
    args = parser.parse_args()

    up_gt, down_gt = load_gt(args.gt_json)
    # optional visualization
    plot_mae_vs_cell_length(
        up_gt, down_gt,
        length=args.length,
        dt=args.dt,
        v_f=args.v_f,
        w=args.w,
        k_j=args.k_j,
        C=args.C,
        n_lanes=args.n_lanes,
    )



