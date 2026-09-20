"""Build a quadruped model by stamping the single-leg XML four times.

Reads ``fivebar_leg.xml`` as the single source of truth for one leg, then
programmatically prefixes all MuJoCo identifiers with a leg tag (FL, FR, RL,
RR) so that four independent legs share the same geometry without name
collisions.

Usage from scripts::

    import quad_model
    model, data = quad_model.build_quad()
"""

from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np

_HERE = Path(__file__).parent

# Leg mount positions on the trunk (x: fwd, y: left, z: up from trunk center).
# The single-leg sagittal plane is the xz plane; all four legs point the same
# way, so the mount positions are pure translations with no rotation.
# A wide stance (y = +-0.085) gives the planar-legged robot enough roll
# authority that a 3-foot tripod is statically stable during a crawl.
_LEG_MOUNTS = {
    "FL": np.array([+0.100, +0.085, -0.02]),
    "FR": np.array([+0.100, -0.085, -0.02]),
    "RL": np.array([-0.100, +0.085, -0.02]),
    "RR": np.array([-0.100, -0.085, -0.02]),
}

# Trot phase offsets (0 = swing starts first; 1 = second)
_TROT_PHASE = {"FL": 0.0, "FR": 1.0, "RL": 1.0, "RR": 0.0}


def _extract_subtree(xml: str, body_name: str) -> str:
    """Extract the <body name="body_name">...</body> subtree by depth counting."""
    marker = f'<body name="{body_name}"'
    idx = xml.index(marker)
    # find the opening tag's closing >
    depth_close = xml.index('>', idx) + 1
    depth = 1
    pos = depth_close
    while depth > 0:
        open_pos = xml.find('<body ', pos)
        close_pos = xml.find('</body>', pos)
        if close_pos == -1:
            raise RuntimeError("Unbalanced body tags")
        if open_pos != -1 and open_pos < close_pos:
            depth += 1
            pos = open_pos + 5
        else:
            depth -= 1
            if depth == 0:
                return xml[idx:close_pos + len('</body>')]
            pos = close_pos + 7
    raise RuntimeError("Unbalanced body tags")


def _rename(xml: str, prefix: str) -> str:
    """Prefix all declared MuJoCo identifiers with *prefix*.

    Identifiers are found from ``name="..."`` declarations.  Every
    attribute value that exactly matches a declared name gets the prefix
    prepended.
    """
    declared = set(re.findall(r'name="([A-Za-z_]\w*)"', xml))
    for name in sorted(declared, key=len, reverse=True):
        xml = xml.replace(f'="{name}"', f'="{prefix}{name}"')
    return xml


def _tendon_xml(prefix: str, stiffness: float = 40.0, damping: float = 0.3,
                knee_dir: float = 1.0) -> str:
    """Fixed SEA tendon per leg.

    spring1 couples the knee rotor r1 to its link j1: tension = k*(r1-j1),
    torque pair equal-and-opposite.  ``knee_dir`` flips which sense the knee
    spring loads (mechanical mirror of the spring's attachment side), the
    lever used to make the elastic thrust push +x instead of +y.
    """
    j1c = -knee_dir
    j2c = -1.0
    return (
        f'<fixed name="{prefix}spring1" stiffness="{stiffness}" damping="{damping}">'
        f'<joint joint="{prefix}r1" coef="1"/><joint joint="{prefix}j1" coef="{j1c}"/>'
        f'</fixed>\n'
        f'<fixed name="{prefix}spring2" stiffness="{stiffness}" damping="{damping}">'
        f'<joint joint="{prefix}r2" coef="1"/><joint joint="{prefix}j2" coef="{j2c}"/>'
        f'</fixed>\n'
    )


def _actuator_xml(prefix: str, forcerange: float = 30.0) -> str:
    return (
        f'<motor name="{prefix}m1" joint="{prefix}r1" '
        f'ctrlrange="-{forcerange} {forcerange}" ctrllimited="true" '
        f'forcerange="-{forcerange} {forcerange}" forcelimited="true"/>\n'
        f'<motor name="{prefix}m2" joint="{prefix}r2" '
        f'ctrlrange="-{forcerange} {forcerange}" ctrllimited="true" '
        f'forcerange="-{forcerange} {forcerange}" forcelimited="true"/>\n'
        f'<motor name="{prefix}yaw" joint="{prefix}yaw" '
        f'ctrlrange="-15 15" ctrllimited="true" '
        f'forcerange="-15 15" forcelimited="true"/>\n'
    )


def _sensor_xml(prefix: str) -> str:
    s = ""
    for j in ("r1", "j1", "r2", "j2", "jc1", "jc2"):
        s += f'<jointpos name="{prefix}sp_{j}" joint="{prefix}{j}"/>\n'
    for j in ("r1", "j1", "r2", "j2"):
        s += f'<jointvel name="{prefix}sv_{j}" joint="{prefix}{j}"/>\n'
    s += f'<jointpos name="{prefix}sp_yaw" joint="{prefix}yaw"/>\n'
    s += f'<jointvel name="{prefix}sv_yaw" joint="{prefix}yaw"/>\n'
    s += f'<framepos name="{prefix}foot_pos" objtype="site" objname="{prefix}s_foot"/>\n'
    s += f'<actuatorfrc name="{prefix}am1" actuator="{prefix}m1"/>\n'
    s += f'<actuatorfrc name="{prefix}am2" actuator="{prefix}m2"/>\n'
    return s


def _equality_xml(prefix: str) -> str:
    return (
        f'<weld body1="{prefix}tipC1" body2="{prefix}tipC2" '
        f'solref="0.001 1" solimp="0.9 0.95 0.001"/>\n'
    )


# Directional-SEA lever (experiment).  +1 = stock.  -1 flips the knee spring
# attachment sense: it fails load-bearing within ~40 ms (the inverted spring
# cannot hold stance weight, so the PD doesn't have a usable equilibrium).
# A true directional spring must keep the SAME load-bearing sense and ease
# only forward-stroke recoil (one-way clutch / position-dependent stiffness),
# which needs mechanism or custom-tendon work beyond a coefficient sign flip.
KNEE_SPRING_DIR = 1.0


def build_quad(k_s: float = 40.0, d_s: float = 0.3) -> tuple:
    """Build and compile the quadruped MuJoCo model.

    Returns (MjModel, MjData).
    """
    leg_xml_raw = (_HERE / "fivebar_leg.xml").read_text(encoding="utf-8")
    hip_xml = _extract_subtree(leg_xml_raw, "hip")

    # Trunk mass and inertia (approx solid box 0.24 x 0.17 x 0.06 m)
    trunk_mass = 2.0
    dx, dy, dz = 0.12, 0.085, 0.03
    Ixx = (1.0/12.0) * trunk_mass * (dy*dy + dz*dz)
    Iyy = (1.0/12.0) * trunk_mass * (dx*dx + dz*dz)
    Izz = (1.0/12.0) * trunk_mass * (dx*dx + dy*dy)

    xml = f'<mujoco model="quad">\n'
    xml += '  <compiler angle="radian" autolimits="false"/>\n'
    xml += '  <option timestep="0.002" iterations="40" tolerance="1e-10"\n'
    xml += '          cone="elliptic" integrator="Euler" gravity="0 0 -9.81"/>\n'
    xml += '  <default>\n'
    xml += '    <geom type="box" condim="3"/>\n'
    xml += '    <joint limited="true" damping="0.02"/>\n'
    xml += '  </default>\n'

    # Worldbody: floor + trunk
    xml += '  <worldbody>\n'
    xml += '    <geom name="floor" type="plane" size="4 4 1" pos="0 0 0" condim="3"/>\n'
    xml += '    <body name="trunk" pos="0 0 0.226">\n'
    xml += f'      <freejoint name="root"/>\n'
    xml += f'      <inertial pos="0 0 0" mass="{trunk_mass}" '
    xml += f'diaginertia="{Ixx:.6f} {Iyy:.6f} {Izz:.6f}"/>\n'
    xml += '      <geom name="trunk_geom" type="box" size="0.12 0.085 0.03" '
    xml += 'contype="0" conaffinity="0"/>\n'
    xml += '      <site name="s_trunk" pos="0 0 0"/>\n'

    # Four legs
    for tag, pos in _LEG_MOUNTS.items():
        hip = _rename(hip_xml, tag)
        # hip sits at the origin of its yaw body; the yaw body carries the
        # mount position on the trunk and a z-axis (vertical) revolute joint.
        hip = re.sub(r'<body name="(.*?)" pos="[^"]*"',
                     '<body name="\\1" pos="0 0 0"',
                     hip, count=1)
        xml += f'      <body name="{tag}hip_yaw" pos="{pos[0]:+.4f} {pos[1]:+.4f} {pos[2]:+.4f}">\n'
        xml += f'        <joint name="{tag}yaw" axis="0 0 1" range="-0.8 0.8" damping="0.05" armature="0.002"/>\n'
        xml += f'        <inertial pos="0 0 0" mass="0.03" diaginertia="0.0002 0.0002 0.0002"/>\n'
        xml += f'        <geom name="{tag}yaw_geom" type="box" size="0.012 0.011 0.014" '
        xml += f'pos="{0.0:+.4f} {0.0:+.4f} {0.0:+.4f}" mass="0.02" contype="0" conaffinity="0"/>\n'
        xml += hip + "\n"
        xml += f'      </body>\n'

    xml += '    </body>\n'   # close trunk
    xml += '  </worldbody>\n'

    # Tendons
    xml += '  <tendon>\n'
    for tag in _LEG_MOUNTS:
        xml += _tendon_xml(tag, stiffness=k_s, damping=d_s,
                           knee_dir=KNEE_SPRING_DIR)
    xml += '  </tendon>\n'

    # Actuators
    xml += '  <actuator>\n'
    for tag in _LEG_MOUNTS:
        xml += _actuator_xml(tag)
    xml += '  </actuator>\n'

    # Sensors (trunk IMU + per-leg)
    xml += '  <sensor>\n'
    xml += '    <framequat name="trunk_quat" objtype="body" objname="trunk"/>\n'
    xml += '    <framelinvel name="trunk_lvel" objtype="body" objname="trunk"/>\n'
    xml += '    <gyro name="gyro" site="s_trunk"/>\n'
    xml += '    <accelerometer name="acc" site="s_trunk"/>\n'
    for tag in _LEG_MOUNTS:
        xml += _sensor_xml(tag)
    xml += '  </sensor>\n'

    # Equalities (weld closure per leg)
    xml += '  <equality>\n'
    for tag in _LEG_MOUNTS:
        xml += _equality_xml(tag)
    xml += '  </equality>\n'

    xml += '</mujoco>\n'

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data
