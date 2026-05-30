from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
URDF_PATH = ROOT / "model" / "Ares" / "urdf" / "Ares.urdf"
MESH_DIR = ROOT / "model" / "Ares" / "meshes"
OUTPUT_PATH = ROOT / "Ares_mujoco_deploy" / "assets" / "Ares.xml"

DEFAULT_STAND = [0.0, -1.35453, 2.54948] * 4


def parse_floats(text: str, count: int | None = None) -> list[float]:
    vals = [float(x.replace("E", "e")) for x in text.replace(",", " ").split()]
    if count is not None and len(vals) != count:
        raise ValueError(f"Expected {count} floats, got {len(vals)} from {text!r}")
    return vals


def rpy_to_quat(rpy: list[float]) -> list[float]:
    roll, pitch, yaw = rpy
    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def fmt(vals: list[float]) -> str:
    return " ".join(f"{v:.8g}" for v in vals)


def main() -> None:
    tree = ET.parse(URDF_PATH)
    root = tree.getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    joints = root.findall("joint")

    parent_joint = {j.find("child").attrib["link"]: j for j in joints}
    children: dict[str, list[ET.Element]] = {}
    for joint in joints:
        children.setdefault(joint.find("parent").attrib["link"], []).append(joint)

    meshdir_rel = os.path.relpath(MESH_DIR, OUTPUT_PATH.parent)
    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<mujoco model="Ares">')
    lines.append(
        f'  <compiler angle="radian" coordinate="local" inertiafromgeom="false" '
        f'balanceinertia="true" meshdir="{meshdir_rel}"/>'
    )
    lines.append('  <option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast"/>')
    lines.append("  <default>")
    lines.append('    <joint damping="0.1" armature="0.01" limited="true"/>')
    lines.append('    <geom friction="1 0.5 0.1" condim="3" rgba="0.8 0.8 0.9 1" density="0"/>')
    lines.append('    <motor ctrllimited="true"/>')
    lines.append("  </default>")
    lines.append("  <asset>")
    lines.append('    <texture name="blue_checker" type="2d" builtin="checker" rgb1="0.05 0.16 0.38" rgb2="0.45 0.68 1" width="512" height="512"/>')
    lines.append('    <material name="blue_checker" texture="blue_checker" texrepeat="8 8" reflectance="0.12"/>')
    for link_name, link in links.items():
        mesh = link.find("./visual/geometry/mesh")
        if mesh is None:
            continue
        lines.append(f'    <mesh name="{link_name}" file="{os.path.basename(mesh.attrib["filename"])}"/>')
    lines.append("  </asset>")
    lines.append("  <worldbody>")
    lines.append('    <geom name="floor" type="plane" size="8 8 0.1" material="blue_checker" contype="1" conaffinity="1" friction="1 0.5 0.1"/>')

    def body_block(link_name: str, indent: int = 4) -> list[str]:
        link = links[link_name]
        pad = " " * indent
        out: list[str] = []
        if link_name == "base_link":
            out.append(f'{pad}<body name="{link_name}" pos="0 0 0">')
            out.append(f"{pad}  <freejoint/>")
            out.append(f'{pad}  <site name="imu_site" pos="0 0 0" size="0.01" rgba="0 0 0 0"/>')
        else:
            joint = parent_joint[link_name]
            origin = joint.find("origin").attrib
            pos = parse_floats(origin.get("xyz", "0 0 0"), 3)
            quat = rpy_to_quat(parse_floats(origin.get("rpy", "0 0 0"), 3))
            axis = joint.find("axis").attrib["xyz"]
            limits = joint.find("limit").attrib
            out.append(f'{pad}<body name="{link_name}" pos="{fmt(pos)}" quat="{fmt(quat)}">')
            out.append(
                f'{pad}  <joint name="{joint.attrib["name"]}" type="hinge" axis="{axis}" '
                f'limited="true" range="{limits["lower"]} {limits["upper"]}"/>'
            )
        inertial = link.find("inertial")
        if inertial is not None:
            origin = inertial.find("origin").attrib
            mass = float(inertial.find("mass").attrib["value"])
            inertia = inertial.find("inertia").attrib
            full = [
                float(inertia["ixx"]),
                float(inertia["iyy"]),
                float(inertia["izz"]),
                float(inertia["ixy"]),
                float(inertia["ixz"]),
                float(inertia["iyz"]),
            ]
            out.append(
                f'{pad}  <inertial pos="{fmt(parse_floats(origin.get("xyz", "0 0 0"), 3))}" '
                f'mass="{mass:.8g}" fullinertia="{fmt(full)}"/>'
            )
        visual = link.find("visual")
        if visual is not None:
            origin = visual.find("origin").attrib
            mesh = visual.find("./geometry/mesh")
            color = visual.find("./material/color")
            rgba = parse_floats(color.attrib.get("rgba", "0.8 0.8 0.9 1"), 4) if color is not None else [0.8, 0.8, 0.9, 1.0]
            out.append(
                f'{pad}  <geom name="{link_name}_geom" type="mesh" mesh="{link_name}" '
                f'pos="{fmt(parse_floats(origin.get("xyz", "0 0 0"), 3))}" '
                f'euler="{fmt(parse_floats(origin.get("rpy", "0 0 0"), 3))}" rgba="{fmt(rgba)}" contype="1" conaffinity="1"/>'
            )
        for child_joint in children.get(link_name, []):
            out.extend(body_block(child_joint.find("child").attrib["link"], indent + 2))
        out.append(f"{pad}</body>")
        return out

    lines.extend(body_block("base_link"))
    lines.append("  </worldbody>")
    lines.append("  <sensor>")
    lines.append('    <gyro site="imu_site"/>')
    lines.append('    <accelerometer site="imu_site"/>')
    lines.append("  </sensor>")
    lines.append("  <actuator>")
    for joint in joints:
        limits = joint.find("limit").attrib
        lines.append(
            f'    <motor name="{joint.attrib["name"]}_motor" joint="{joint.attrib["name"]}" '
            f'gear="1" ctrlrange="-{limits["effort"]} {limits["effort"]}"/>'
        )
    lines.append("  </actuator>")
    lines.append("</mujoco>")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
