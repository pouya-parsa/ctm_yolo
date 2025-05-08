import os
import time
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
    os.makedirs('output', exist_ok=True)
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
    

    # plt.plot(cell_lengths, mae_up_list, label='Upstream MAE', color='blue')
    # plt.plot(cell_lengths, mae_up_list, label='Upstream MAE', color='blue', marker='o')
    plt.plot(cell_lengths, mae_down_list, label='Downstream MAE', color='red', marker='o')
    plt.xlabel('Cell length (m)')
    plt.ylabel('Mean Absolute Error (veh)')
    plt.title('MAE vs Cell Length')
    plt.legend()
    plt.grid(True)
    plt.savefig('output/mae_vs_cell_length.pdf')
    plt.show()

def plot_runtime_vs_cell_length(
    up_cum_gt,
    length, dt, v_f, w, k_j, C, n_lanes
):
    import matplotlib.pyplot as plt
    os.makedirs('output', exist_ok=True)
    cell_lengths = list(range(35, int(length) + 1, 10))
    runtimes = []
    for cl in cell_lengths:
        start = time.perf_counter()
        run_ctm(
            up_cum_gt, length=length, cell_len=cl,
            dt=dt, v_f=v_f, w=w, k_j=k_j, C=C, n_lanes=n_lanes
        )
        end = time.perf_counter()
        runtimes.append((end - start) * 1_000)

    plt.plot(cell_lengths, runtimes, label='Run time', color='green', marker='x')
    plt.xlabel('Cell length (m)')
    plt.ylabel('Computation Time (ms)')
    plt.title('CTM run time vs Cell Length')
    plt.legend()
    plt.grid(True)
    plt.savefig('output/runtime_vs_cell_length.pdf')
    plt.show()


def run_chain(up_cum_gt, link_cfgs):
    """
    Simulate a *series* of CTM links.
    Parameters
    ----------
    up_cum_gt : 1‑D np.ndarray
        cumulative upstream count for *link 1* (vehicles entering the chain)
    link_cfgs : list[dict]
        one dict per link, keys matching CTMLink’s constructor
    Returns
    -------
    up_mod  : modelled cumulative count at entry to link‑1
    down_mod: modelled cumulative count exiting the final link
    """
    T              = len(up_cum_gt)
    inflow_series  = gt_to_inflow(up_cum_gt)          # vehicles entering first link
    up_flow_first  = None

    for i, cfg in enumerate(link_cfgs):
        link = CTMLink(**cfg)
        link.reset()                                  # always reset densities first
        print("inflow_series: ", len(inflow_series))
        print("T: ", T)
        flows   = link.run(T, inflow=inflow_series)   # shape: (T, n_cells, 2)
        if i == 0:                                    # remember true upstream flow
            up_flow_first = flows[:, 0, 0]
        inflow_series = flows[:, -1, 1]               # downstream flow becomes next inflow
        inflow_series = inflow_series.tolist()

    up_mod   = np.cumsum(up_flow_first)               # cumulative at chain entrance
    down_mod = np.cumsum(inflow_series)               # cumulative at chain exit
    return up_mod, down_mod



def sweep_chain_cell_length(
    up_gt: np.ndarray,
    down_gt: np.ndarray,
    base_link_cfgs: list[dict],
    min_link_len: int,
):
    cell_lengths = list(range(30, min_link_len + 1, 10))
    mae_down_vec = []
    runtime_vec  = []

    for cl in cell_lengths:
        # clone base configs but overwrite cell_length
        link_cfgs = [
            {**cfg, "cell_length": cl} for cfg in base_link_cfgs
        ]
        start = time.perf_counter()
        _, down_mod = run_chain(up_gt, link_cfgs)
        elapsed_ms = (time.perf_counter() - start) * 1_000

        # MAE (downstream only)
        _, mae_down = compare(up_gt, down_gt, up_gt, down_mod)  # upstream MAE not needed
        mae_down_vec.append(mae_down)
        runtime_vec.append(elapsed_ms)

    return cell_lengths, mae_down_vec, runtime_vec

# ---------------------------- script entry ----------------------------
# 118.87
# 86.34
# 101.05 

# --------------- main script ---------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("link_json", nargs=3, help="GT JSON files for links 1‑3 (order matters)")
    p.add_argument("--lengths", nargs=3, type=float, required=True, help="link lengths [m]")
    p.add_argument("--cell_len", type=float, default=30.0)
    p.add_argument("--dt",       type=float, default=1.0)
    p.add_argument("--v_f",      type=float, default=30.56)   # 110 km/h
    p.add_argument("--w",        type=float, default=5.0)
    p.add_argument("--k_j",      type=float, default=0.14)
    p.add_argument("--C",        type=float, default=1.46)
    p.add_argument("--n_lanes",  type=int,   default=4)
    args = p.parse_args()

    # ---------- load ground truth ----------
    up1, _   = load_gt(args.link_json[0])             # use upstream of link‑1 as chain inflow
    _, dn3   = load_gt(args.link_json[2])             # downstream of link‑3 for MAE

    # ---------- build per‑link config list ----------
    link_cfgs = [
        dict(length=args.lengths[0], cell_length=args.cell_len, dt=args.dt,
             v_f=args.v_f, w=args.w, k_j=args.k_j, C=args.C, n_lanes=args.n_lanes),
        dict(length=args.lengths[1], cell_length=args.cell_len, dt=args.dt,
             v_f=args.v_f, w=args.w, k_j=args.k_j, C=args.C, n_lanes=args.n_lanes),
        dict(length=args.lengths[2], cell_length=args.cell_len, dt=args.dt,
             v_f=args.v_f, w=args.w, k_j=args.k_j, C=args.C, n_lanes=args.n_lanes),
    ]

    # # ---------- run the chain ----------
    # up_mod, down_mod = run_chain(up1, link_cfgs)

    # # ---------- evaluate ----------
    # mae_up, mae_down = compare(up1, dn3, up_mod, down_mod)
    # print(f"MAE at chain entrance : {mae_up:7.3f} veh")
    # print(f"MAE at chain exit     : {mae_down:7.3f} veh")

    min_link_len = int(min(args.lengths))
    cell_lengths, mae_vec, rt_vec = sweep_chain_cell_length(
        up_gt=up1,
        down_gt=dn3,
        base_link_cfgs=link_cfgs,
        min_link_len=min_link_len,
    )

    import matplotlib.pyplot as plt
    os.makedirs('output', exist_ok=True)

    # MAE vs Δx
    plt.figure()
    plt.plot(cell_lengths, mae_vec, marker='o', color='red')
    plt.xlabel('Cell length (m)')
    plt.ylabel('Downstream MAE (veh)')
    plt.title('MAE vs Cell Length (Three–link chain)')
    plt.grid(True)
    plt.savefig('output/mae_vs_cell_length_chain.pdf')
    plt.show()

    # Runtime vs Δx
    plt.figure()
    plt.plot(cell_lengths, rt_vec, marker='x', color='green')
    plt.xlabel('Cell length (m)')
    plt.ylabel('Runtime (ms)')
    plt.title('Runtime vs Cell Length (Three–link chain)')
    plt.grid(True)
    plt.savefig('output/runtime_vs_cell_length_chain.pdf')
    plt.show()
