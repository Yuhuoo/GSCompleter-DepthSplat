from pathlib import Path

import numpy as np
import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes


def export_ply(
    extrinsics: Float[Tensor, "4 4"],
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    path: Path,
):

    view_rotation = extrinsics[:3, :3].inverse()
    # Apply the rotation to the means (Gaussian positions).
    means = einsum(view_rotation, means, "i j, ... j -> ... i")

    # Apply the rotation to the Gaussian rotations.
    rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    rotations = view_rotation.detach().cpu().numpy() @ rotations
    rotations = R.from_matrix(rotations).as_quat()
    x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
    rotations = np.stack((w, x, y, z), axis=-1)

    # Since our axes are swizzled for the spherical harmonics, we only export the DC band
    harmonics_view_invariant = harmonics[..., 0]

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0)]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = (
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        harmonics_view_invariant.detach().cpu().contiguous().numpy(),
        torch.logit(opacities[..., None]).detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations,
    )
    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
    

# def save_gaussian_ply(gaussians, visualization_dump, example, save_path):

#     v, _, h, w = example["context"]["image"].shape[1:]

#     # Transform means into camera space.
#     means = rearrange(
#         gaussians.means, "() (v h w spp) xyz -> h w spp v xyz", v=v, h=h, w=w
#     )

#     # Create a mask to filter the Gaussians. throw away Gaussians at the
#     # borders, since they're generally of lower quality.
#     mask = torch.zeros_like(means[..., 0], dtype=torch.bool)
#     GAUSSIAN_TRIM = 8
#     mask[GAUSSIAN_TRIM:-GAUSSIAN_TRIM, GAUSSIAN_TRIM:-GAUSSIAN_TRIM, :, :] = 1

#     def trim(element):
#         element = rearrange(
#             element, "() (v h w spp) ... -> h w spp v ...", v=v, h=h, w=w
#         )
#         return element[mask][None]

#     # convert the rotations from camera space to world space as required
#     cam_rotations = trim(visualization_dump["rotations"])[0]
#     c2w_mat = repeat(
#         example["context"]["extrinsics"][0, :, :3, :3],
#         "v a b -> h w spp v a b",
#         h=h,
#         w=w,
#         spp=1,
#     )
#     c2w_mat = c2w_mat[mask]  # apply trim

#     cam_rotations_np = R.from_quat(
#         cam_rotations.detach().cpu().numpy()
#     ).as_matrix()
#     world_mat = c2w_mat.detach().cpu().numpy() @ cam_rotations_np
#     world_rotations = R.from_matrix(world_mat).as_quat()
#     world_rotations = torch.from_numpy(world_rotations).to(
#         visualization_dump["scales"]
#     )

#     export_ply(
#         example["context"]["extrinsics"][0, 0],
#         trim(gaussians.means)[0],
#         trim(visualization_dump["scales"])[0],
#         world_rotations,
#         trim(gaussians.harmonics)[0],
#         trim(gaussians.opacities)[0],
#         save_path,
#     )

def save_gaussian_ply(gaussians, visualization_dump, example, save_path):
    # 确保父目录存在
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 取出 batch 0 的数据 (假设 test batch size 为 1)
    # [N, 3]
    xyz = gaussians.means[0].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    
    # [N, 1] -> [N]
    opacities = gaussians.opacities[0].detach().cpu().numpy()
    
    # [N, 3]
    scale = gaussians.scales[0].detach().cpu().numpy()
    # Log scale for PLY format (standard 3DGS convention)
    scale = np.log(scale + 1e-8) 
    
    # [N, 4]
    rotation = gaussians.rotations[0].detach().cpu().numpy()
    
    # SH (Spherical Harmonics)
    # gaussians.harmonics: [1, N, 3, K] -> [N, 3, K]
    shs = gaussians.harmonics[0].detach().cpu().numpy()
    
    # 处理 SH 格式: f_dc (degree 0) 和 f_rest (degree 1-3)
    # standard 3DGS ply format stores f_dc_0, f_dc_1, f_dc_2, then f_rest...
    # shs shape: [N, 3, (max_sh_degree+1)**2]
    
    # DC component: [N, 3, 1] -> [N, 3]
    f_dc = shs[:, :, 0].flatten(order='C') # flatten to (N*3) is wrong for structured array, keep separate
    f_dc = shs[:, :, 0] # [N, 3]
    
    # Rest components: [N, 3, K-1]
    f_rest = shs[:, :, 1:] 
    # Flatten last two dims: [N, 3 * (K-1)]
    f_rest = f_rest.reshape(f_rest.shape[0], -1)

    # === 构建 PLY 数据结构 ===
    
    # 定义 PLY 属性头
    dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                  ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
    
    # f_dc
    for i in range(3):
        dtype_full.append(('f_dc_{}'.format(i), 'f4'))
    
    # f_rest
    for i in range(f_rest.shape[1]):
        dtype_full.append(('f_rest_{}'.format(i), 'f4'))
        
    # opacity, scale, rot
    dtype_full.append(('opacity', 'f4'))
    
    for i in range(3):
        dtype_full.append(('scale_{}'.format(i), 'f4'))
        
    for i in range(4):
        dtype_full.append(('rot_{}'.format(i), 'f4'))

    # 填充数据
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    
    # 1. Geometry
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['nx'] = normals[:, 0]
    elements['ny'] = normals[:, 1]
    elements['nz'] = normals[:, 2]
    
    # 2. Color (SH DC)
    # f_dc is [N, 3] (R, G, B)
    elements['f_dc_0'] = f_dc[:, 0]
    elements['f_dc_1'] = f_dc[:, 1]
    elements['f_dc_2'] = f_dc[:, 2]
    
    # 3. Color (SH Rest)
    for i in range(f_rest.shape[1]):
        elements['f_rest_{}'.format(i)] = f_rest[:, i]
        
    # 4. Attributes
    # Opacity is usually stored after sigmoid in memory, but PLY usually expects logit?
    # standard 3DGS python convertors usually expect raw opacity (logit) if the renderer applies sigmoid, 
    # OR sigmoid opacity if the renderer expects it.
    # DepthSplat's Gaussians dataclass usually stores raw logits for scale/opacity?
    # Based on Encoder code: opacities = raw_gaussians[..., :1].sigmoid()
    # So 'gaussians.opacities' IS ALREADY SIGMOIDED [0,1].
    # Standard PLY Viewers (like SIBR) expect LOGIT for opacity.
    # Inverse Sigmoid: log(p / (1-p))
    opacities_logit = np.log(opacities / (1 - opacities + 1e-8) + 1e-8)
    elements['opacity'] = opacities_logit
    
    # Scales
    elements['scale_0'] = scale[:, 0]
    elements['scale_1'] = scale[:, 1]
    elements['scale_2'] = scale[:, 2]
    
    # Rotations
    elements['rot_0'] = rotation[:, 0]
    elements['rot_1'] = rotation[:, 1]
    elements['rot_2'] = rotation[:, 2]
    elements['rot_3'] = rotation[:, 3]

    # 保存文件
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(str(save_path))
    
    print(f"Saved fused gaussians ({xyz.shape[0]} points) to {save_path}")
