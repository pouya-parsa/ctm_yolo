from __future__ import annotations
from math import ceil
from typing import Sequence, Union
import numpy as np

class CTMLink:
    """Cell Transmission Model for a single road link."""

    def __init__(
        self,
        length: float,
        cell_length: float = 50.0,
        dt: float = 5.0,
        v_f: float = 27.0,
        w: float = 5.0,
        k_j: float = 0.15,
        C: float | None = None,
        n_lanes: int = 1,
    ) -> None:
        self.length = length
        self.cell_len = cell_length
        self.dt = dt
        self.v_f = v_f
        self.w = w
        self.k_j = k_j
        self.n_lanes = n_lanes
        self.C = C if C is not None else 0.5 * k_j * v_f
        self.n_cells = ceil(length / cell_length)
        self.k = np.zeros(self.n_cells)

    @property
    def q(self) -> np.ndarray:
        return self.k * self.v_f

    @property
    def s(self) -> np.ndarray:
        return np.minimum(self.q, self.C)

    @property
    def r(self) -> np.ndarray:
        return np.minimum(self.C, self.w * (self.k_j - self.k))

    def reset(self, k0: float = 0.0) -> None:
        self.k.fill(k0)

    def set_density(self, densities: Sequence[float]) -> None:
        if len(densities) != self.n_cells:
            raise ValueError("length mismatch with number of cells")
        self.k[:] = densities

    def step(self, inflow: float = 0.0, outflow_demand: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Advance one Δt, return (enter_flows, exit_flows) for each cell."""
        out_cap = self.C if outflow_demand is None else min(outflow_demand, self.C)
        S = self.s; R = self.r
        S_up = min(inflow, self.C); R_down = out_cap
        y = np.minimum(S, np.roll(R, -1))
        y[-1] = min(S[-1], R_down)
        y_in = min(S_up, R[0])
        # convert to veh per dt
        y = y * self.dt; y_in = y_in * self.dt
        exit_flow = y.copy()
        enter_flow = np.roll(y, 1)
        enter_flow[0] = y_in
        # update densities
        delta = (enter_flow - exit_flow) / (self.cell_len * self.n_lanes)
        self.k += delta
        return enter_flow, exit_flow

    def run(
        self,
        T: int,
        inflow: Union[float, Sequence[float]] = 0.0,
        outflow_demand: float | None = None,
    ) -> np.ndarray:
        """Simulate T steps and return array shape (T, n_cells, 2) with enter/exit flows."""
        if isinstance(inflow, Sequence):
            if len(inflow) != T:
                raise ValueError("inflow sequence length must equal T")
            inflow_seq = inflow
        else:
            inflow_seq = [inflow] * T
        flows = np.zeros((T, self.n_cells, 2))
        for t in range(T):
            enter_flow, exit_flow = self.step(inflow=inflow_seq[t], outflow_demand=outflow_demand)
            flows[t, :, 0] = enter_flow
            flows[t, :, 1] = exit_flow
        return flows

if __name__ == "__main__":
    link = CTMLink(length=118.76, cell_length=50.0, dt=5.0)
    link.reset(0.02)
    inflows = [0.3] * 5
    out = link.run(T=5, inflow=inflows)
    # out.shape == (5, n_cells, 2)
    print(out.shape)  # e.g. (5,3,2)
    print(out)