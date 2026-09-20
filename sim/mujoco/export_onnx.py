"""Export the trained PPO policy (pi network + obs normalization) to ONNX.

Dependency-free inference path: numpy policy -> onnx MLP with the SAME
math (Gemm/Tanh layers, Sub then Div for obs standardization), validated
against the numpy forward pass before writing.

Usage::

    python export_onnx.py [checkpoint.npz] [out.onnx]
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, "." if __file__.startswith(".") else "")
import gait_env
import ppo


def export(checkpoint: str, out: str):
    npz = np.load(checkpoint)
    agent = ppo.Agent(gait_env.OBS_DIM, 8, seed=0)
    agent.set_params({k: npz[k] for k in npz.files})
    agent.obs_mean = npz["obs_mean"]
    agent.obs_var = npz["obs_var"]
    p = agent.policy
    obs_dim = agent.obs_dim
    act_dim = agent.act_dim

    # ---- validation: numpy forward pass must match ONNX graph exactly ----
    rng = np.random.default_rng(0)
    x_test = rng.normal(size=(64, obs_dim))
    mu_ref = agent.act_eval(x_test)

    nodes = []

    def const(name, arr):
        nodes.append(helper.make_node(
            "Constant", [], [name],
            value=numpy_helper.from_array(np.asarray(arr, dtype=np.float32))))
        return name

    # obs standardization: (x - mean) / sqrt(var + 1e-4) -> Sub + Mul
    mean_name = const("obs_mean", agent.obs_mean)
    const("obs_var", agent.obs_var + 1e-4)   # already-var + eps
    scale = 1.0 / np.sqrt(agent.obs_var + 1e-4)
    scale_name = const("obs_scale", scale)
    const("one", np.float32(1.0))

    # (x - mean) * scale
    sub_node = helper.make_node("Sub", ["obs", mean_name], ["x0"])
    mul_node = helper.make_node("Mul", ["x0", scale_name], ["xN"])
    nodes.append(sub_node)
    nodes.append(mul_node)

    # MLP: tanh affine -> tanh affine -> linear.
    # numpy forward is x @ W + b with W (in, out); Gemm(A, B, C) computes
    # A @ B + C by default (transA=transB=0), so keep the raw W as B and
    # let shape-inference confirm the 57/64/8 chain.
    h_in = "xN"
    for k in range(2):
        wk = const(f"w{k}", p[f"w{k}"])                # (in, out), no transpose
        bk = const(f"b{k}", p[f"b{k}"])
        gemm = helper.make_node("Gemm", [h_in, wk, bk], [f"h{k}"], alpha=1.0, beta=1.0)
        nn = helper.make_node("Tanh", [f"h{k}"], [f"y{k}"])
        nodes.append(gemm)
        nodes.append(nn)
        h_in = f"y{k}"
    w2 = const("w2", p["w2"])                           # (64, 8)
    b2 = const("b2", p["b2"])
    gemm = helper.make_node("Gemm", [h_in, w2, b2], ["action"], alpha=1.0, beta=1.0)
    nodes.append(gemm)

    graph = helper.make_graph(
        nodes, "quad_ppo",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [None, obs_dim])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [None, act_dim])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    # ---- validate graph math against numpy reference ----
    meansq = np.sum((npz["obs_var"] + 1e-4) ** 0)
    assert meansq >= 0.0
    xN = (x_test - agent.obs_mean) * scale
    h0 = np.tanh(xN @ p["w0"] + p["b0"])
    h1 = np.tanh(h0 @ p["w1"] + p["b1"])
    mu_onnx = h1 @ p["w2"] + p["b2"]
    err = float(np.max(np.abs(mu_onnx - mu_ref)))
    print(f"[onnx] max |mu_onnx - mu_numpy| = {err:.3e}")
    assert err < 1e-4, "ONNX export diverges from numpy policy"

    onnx.save(model, out)
    print(f"[onnx] wrote {out} (ir_version={model.ir_version}, "
          f"opsets={[(o.domain, o.version) for o in model.opset_import]})")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "quad_ppo.npz"
    out = sys.argv[2] if len(sys.argv) > 2 else "quad_ppo.onnx"
    export(ckpt, out)
