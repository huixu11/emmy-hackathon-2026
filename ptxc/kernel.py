"""Load one Kernel IR file: the kernel body, the buffers it touches, and its launch contract.

The typed body comes from emmy's own deserializer (`Graph.from_dict`). Buffer shapes and dtypes
come from the plain graph level of the same json. The parameter order and the launch geometry
come from the problem's meta.json next to the IR, which is the contract: emmy may reorder
arguments between pipeline stages, so the IR node's `inputs` list is only a fallback.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.kernel.ir import KernelOp

from .errors import CompileError


@dataclass(frozen=True)
class Buffer:
    name: str
    shape: tuple[int, ...]
    dtype: str
    is_output: bool


@dataclass(frozen=True)
class Kernel:
    entry: str
    body: tuple                    # the KernelOp body: a tuple of IR statements
    params: tuple[str, ...]        # buffer names in .param order
    buffers: dict[str, Buffer]
    grid: int | None               # blocks and threads per block of the 1-D launch in meta.json,
    block: int | None              # None when there is no meta.json
    dynamic_smem_bytes: int | None  # the launch's dynamic shared memory, None when unknown

    @property
    def launched_threads(self) -> int | None:
        return None if self.grid is None else self.grid * self.block


def load_kernel(ir_path: Path) -> Kernel:
    raw = json.loads(ir_path.read_text())
    graph = Graph.from_dict(raw)
    kernel_ids = [nid for nid, node in graph.nodes.items() if isinstance(node.op, KernelOp)]
    if len(kernel_ids) != 1:
        raise CompileError(f"expected one KernelOp in {ir_path.name}, found {len(kernel_ids)}")
    node_id = kernel_ids[0]
    op = graph.nodes[node_id].op
    if op.serial:
        raise CompileError(f"serial axes {op.serial} are not supported")

    raw_node = raw["nodes"][node_id]
    outputs = [o["name"] for o in raw_node["outputs"]]
    buffers = {
        o["name"]: Buffer(o["name"], tuple(o["shape"]), o["dtype"], o["name"] in outputs)
        for node in raw["nodes"].values()
        for o in node["outputs"]
    }

    launch = _launch(ir_path.parent / "meta.json", op.name)
    if launch is not None:
        params = tuple(launch["arg_order"])
        if math.prod(launch["grid"][1:]) * math.prod(launch["block"][1:]) != 1:
            raise CompileError(f"only 1-D launches are supported, got {launch['grid']} x {launch['block']}")
        grid, block = launch["grid"][0], launch["block"][0]
        smem = launch.get("smem_bytes", 0)
    else:
        params = tuple(raw_node["inputs"]) + tuple(outputs)
        grid = block = smem = None
    unknown = [p for p in params if p not in buffers]
    if unknown:
        raise CompileError(f"parameters {unknown} name no buffer of the graph")
    return Kernel(op.name, op.body, params, buffers, grid, block, smem)


def _launch(meta_path: Path, entry: str) -> dict | None:
    """This kernel's entry in meta.json's `kernels` list, if the file is there."""
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    return next((k for k in meta.get("kernels", []) if k.get("entry") == entry), None)
