# terrain_generator.py
import os
import math
import random
from typing import List, Tuple, Optional, Dict

import numpy as np
import pybullet as p

# Optional image backends
PIL_AVAILABLE = False
CV2_AVAILABLE = False
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    try:
        import cv2  # type: ignore
        CV2_AVAILABLE = True
    except Exception:
        pass


def generate_procedural_heightfield(
    rows: int = 100,
    cols: int = 100,
    height_perturbation: float = 0.01,
    seed: int = 10,
) -> np.ndarray:
    """
    Generate a procedural heightfield using random perturbation.

    Args:
        rows: Number of rows in heightfield (Y direction).
        cols: Number of columns in heightfield (X direction).
        height_perturbation: Noise amplitude (raw values before Z meshScale).
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray: 2D heightfield data of shape (rows, cols).
    """
    rng = random.Random(seed)
    heightfield = np.zeros((rows, cols), dtype=np.float32)

    # Simple correlated noise by accumulating from neighbors
    for r in range(rows):
        for c in range(cols):
            n1 = heightfield[r - 1, c] if r > 0 else 0.0
            n2 = heightfield[r, c - 1] if c > 0 else n1
            noise = rng.uniform(-height_perturbation, height_perturbation)
            heightfield[r, c] = 0.5 * (n1 + n2) + noise

    return heightfield


def _read_image_rgb(image_path: str, target_wh: Tuple[int, int]) -> np.ndarray:
    """
    Read an image as RGB uint8 and resize to target (width, height).
    Returns (H, W, 3)
    """
    target_w, target_h = target_wh
    if PIL_AVAILABLE:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((target_w, target_h), resample=Image.NEAREST)
        return np.array(img, dtype=np.uint8)
    elif CV2_AVAILABLE:
        arr_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if arr_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        arr_bgr = cv2.resize(arr_bgr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return arr_bgr[:, :, ::-1].copy()
    else:
        raise ImportError("Please install Pillow (`pip install pillow`) or OpenCV (`pip install opencv-python`).")

def _read_image_gray(image_path: str, target_wh: Tuple[int, int]) -> np.ndarray:
    """
    Read an image as GRAY uint8 and resize to target (width, height).
    Returns (H, W) uint8 in [0,255].
    """
    target_w, target_h = target_wh
    if PIL_AVAILABLE:
        img = Image.open(image_path).convert("L")
        img = img.resize((target_w, target_h), resample=Image.NEAREST)
        return np.array(img, dtype=np.uint8)
    elif CV2_AVAILABLE:
        arr = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return arr
    else:
        raise ImportError("Please install Pillow (`pip install pillow`) or OpenCV (`pip install opencv-python`).")



def _dilate_mask(mask: np.ndarray, radius: int = 1, iters: int = 1) -> np.ndarray:
    """
    Naive binary dilation without external deps. Suitable for small grids.

    Args:
        mask: boolean array (H, W)
        radius: neighborhood radius in pixels
        iters: number of dilation iterations
    """
    assert mask.dtype == np.bool_, "mask must be boolean"
    out = mask.copy()
    H, W = mask.shape
    for _ in range(iters):
        padded = np.pad(out, radius, mode="edge")
        acc = np.zeros_like(out, dtype=np.bool_)
        k = 2 * radius + 1
        for dr in range(k):
            for dc in range(k):
                acc |= padded[dr : dr + H, dc : dc + W]
        out = acc
    return out


def _pick_sparse_points(
    mask: np.ndarray,
    min_dist: int = 6,
    max_points: Optional[int] = None,
    seed: int = 0,
) -> List[Tuple[int, int]]:
    """
    Pick sparse point coordinates (row, col) from a boolean mask using NMS-style suppression.

    Args:
        mask: boolean mask (H, W)
        min_dist: minimum Manhattan distance between chosen points (in pixels)
        max_points: optional cap on number of points
        seed: RNG seed for shuffling candidates

    Returns:
        List of (row, col)
    """
    coords = np.argwhere(mask)
    if coords.size == 0:
        return []
    rng = random.Random(seed)
    idxs = list(range(coords.shape[0]))
    rng.shuffle(idxs)

    chosen: List[Tuple[int, int]] = []
    used = np.zeros_like(mask, dtype=np.uint8)
    H, W = mask.shape

    for idx in idxs:
        r, c = map(int, coords[idx])
        if used[r, c]:
            continue
        # Accept this point
        chosen.append((r, c))
        if max_points is not None and len(chosen) >= max_points:
            break
        # Suppress neighbors within min_dist
        r0 = max(0, r - min_dist)
        r1 = min(H, r + min_dist + 1)
        c0 = max(0, c - min_dist)
        c1 = min(W, c + min_dist + 1)
        used[r0:r1, c0:c1] = 1

    return chosen


def _rc_to_world(
    r: int,
    c: int,
    heightfield: np.ndarray,
    mesh_scale: Tuple[float, float, float],
    base_position: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """
    Convert (row, col) index to world (x, y, z).
    x ~ col * sx, y ~ row * sy, z ~ height * sz
    """
    sx, sy, sz = mesh_scale
    x = c * sx + base_position[0]
    y = r * sy + base_position[1]
    z = float(heightfield[r, c]) * sz + base_position[2]
    return x, y, z


def _carve_disk(
    heightfield: np.ndarray,
    center_rc: Tuple[int, int],
    radius_world: float,
    depth_world: float,
    mesh_scale: Tuple[float, float, float],
    mode: str = "depress",
):
    """
    Carve a circular depression or mound by modifying heights in-place.

    Args:
        heightfield: (H, W)
        center_rc: (row, col)
        radius_world: radius in world meters
        depth_world: positive magnitude in meters; depress lowers, mound raises
        mesh_scale: (sx, sy, sz)
        mode: "depress" or "mound"
    """
    H, W = heightfield.shape
    r0, c0 = center_rc
    sx, sy, sz = mesh_scale
    # Convert world radius to pixel extents (use average spacing)
    rx = max(1, int(round(radius_world / sx)))
    ry = max(1, int(round(radius_world / sy)))
    rmin = max(0, r0 - ry)
    rmax = min(H, r0 + ry + 1)
    cmin = max(0, c0 - rx)
    cmax = min(W, c0 + rx + 1)

    # Convert world depth to heightfield units
    magnitude_hf = depth_world / sz
    for r in range(rmin, rmax):
        for c in range(cmin, cmax):
            # Elliptical distance
            dr = (r - r0) / max(1e-6, ry)
            dc = (c - c0) / max(1e-6, rx)
            d = math.sqrt(dr * dr + dc * dc)
            if d <= 1.0:
                # Smooth profile (cosine falloff)
                w = 0.5 * (1.0 + math.cos(math.pi * d))
                if mode == "depress":
                    heightfield[r, c] -= w * magnitude_hf
                else:
                    heightfield[r, c] += w * magnitude_hf


def _apply_trenches_from_mask(
    heightfield: np.ndarray,
    blue_trench_mask: np.ndarray,
    trench_depth_world: float,
    mesh_scale: Tuple[float, float, float],
    dilate_radius: int = 1,
    dilate_iters: int = 1,
):
    """
    Apply trenches by lowering height where the (dilated) mask is True.
    """
    mask = blue_trench_mask.astype(bool)
    if dilate_iters > 0 and dilate_radius > 0:
        mask = _dilate_mask(mask, radius=dilate_radius, iters=dilate_iters)

    depth_hf = trench_depth_world / max(1e-6, mesh_scale[2])
    heightfield[mask] -= depth_hf


def _extract_feature_masks(
    rows: int,
    cols: int,
    # Option A: single label map where white=trench, black=cylinder
    label_map_path: Optional[str] = None,
    label_trench_white_thresh: int = 200,
    label_cylinder_black_thresh: int = 55,
    # Option B: two separate binary masks
    trench_mask_path: Optional[str] = None,      # foreground: white by default
    cylinder_mask_path: Optional[str] = None,    # foreground: black by default (set flag to False if white)
    trench_mask_white_is_fg: bool = True,
    cylinder_mask_black_is_fg: bool = True,
    binary_threshold: int = 127,
    # Fallback (original color heuristics) if no masks provided
    image_path_for_fallback: Optional[str] = None,
    blue_thresh: int = 160,
    red_thresh: int = 160,
    green_low: int = 120,
    other_low: int = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (blue_mask_for_trench, red_mask_for_cylinder) as booleans of shape (rows, cols).
    Priority:
      1) label_map_path if provided
      2) separate trench/cylinder masks if provided
      3) fallback color detection from image_path_for_fallback
    """
    target_wh = (cols, rows)

    # 1) Single label map (white=trench, black=cylinder)
    if label_map_path is not None and os.path.exists(label_map_path):
        g = _read_image_gray(label_map_path, target_wh)
        trench_mask = (g >= label_trench_white_thresh)
        cylinder_mask = (g <= label_cylinder_black_thresh)
        return trench_mask, cylinder_mask

    # 2) Separate masks
    trench_mask = None
    cylinder_mask = None

    if trench_mask_path is not None and os.path.exists(trench_mask_path):
        gt = _read_image_gray(trench_mask_path, target_wh)
        if trench_mask_white_is_fg:
            trench_mask = (gt > binary_threshold)
        else:
            trench_mask = (gt <= binary_threshold)

    if cylinder_mask_path is not None and os.path.exists(cylinder_mask_path):
        gc = _read_image_gray(cylinder_mask_path, target_wh)
        if cylinder_mask_black_is_fg:
            cylinder_mask = (gc <= binary_threshold)
        else:
            cylinder_mask = (gc > binary_threshold)

    if trench_mask is not None or cylinder_mask is not None:
        if trench_mask is None:
            trench_mask = np.zeros((rows, cols), dtype=bool)
        if cylinder_mask is None:
            cylinder_mask = np.zeros((rows, cols), dtype=bool)
        return trench_mask, cylinder_mask

    # 3) Fallback to color-based heuristics (original behavior)
    if image_path_for_fallback is None or not os.path.exists(image_path_for_fallback):
        raise FileNotFoundError("No valid mask provided and fallback image is missing.")
    rgb = _read_image_rgb(image_path_for_fallback, target_wh)  # (H,W,3)
    R = rgb[:, :, 0].astype(np.int32)
    G = rgb[:, :, 1].astype(np.int32)
    B = rgb[:, :, 2].astype(np.int32)
    blue_mask = (B >= blue_thresh) & (R <= other_low) & (G <= green_low)
    red_mask = (R >= red_thresh) & (G <= other_low) & (B <= other_low)
    return blue_mask, red_mask


def create_pybullet_terrain(
    heightfield: np.ndarray,
    mesh_scale: Tuple[float, float, float] = (1.0, 1.0, 10.0),
    base_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    texture_path: Optional[str] = None,
    terrain_lateral_friction: Optional[float] = None,
    terrain_rolling_friction: Optional[float] = None,
    terrain_spinning_friction: Optional[float] = None,
    terrain_contact_stiffness: Optional[float] = None,
    terrain_contact_damping: Optional[float] = None,
) -> int:
    """
    Create a PyBullet heightfield terrain from a 2D height array.

    Args:
        heightfield: shape (rows, cols)
        mesh_scale: (sx, sy, sz) spacing & vertical scale
        base_position: base xyz placement
        texture_path: optional path to a texture image to apply
        terrain_*: optional dynamics for the terrain body

    Returns:
        terrain body unique id
    """
    rows, cols = heightfield.shape
    heightfield_flat = np.array(heightfield, dtype=np.float32).reshape(-1)

    terrain_collision = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        heightfieldData=heightfield_flat,
        meshScale=list(mesh_scale),
        numHeightfieldRows=rows,
        numHeightfieldColumns=cols,
    )

    # For heightfields, the collision shape renders in GUI; no explicit visual shape needed
    terrain = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=terrain_collision)
    p.resetBasePositionAndOrientation(terrain, base_position, [0, 0, 0, 1])

    # Terrain dynamics (optional)
    dynamics_kwargs: Dict[str, float] = {}
    if terrain_lateral_friction is not None:
        dynamics_kwargs["lateralFriction"] = float(terrain_lateral_friction)
    if terrain_rolling_friction is not None:
        dynamics_kwargs["rollingFriction"] = float(terrain_rolling_friction)
    if terrain_spinning_friction is not None:
        dynamics_kwargs["spinningFriction"] = float(terrain_spinning_friction)
    if terrain_contact_stiffness is not None:
        dynamics_kwargs["contactStiffness"] = float(terrain_contact_stiffness)
    if terrain_contact_damping is not None:
        dynamics_kwargs["contactDamping"] = float(terrain_contact_damping)
    if dynamics_kwargs:
        p.changeDynamics(terrain, -1, **dynamics_kwargs)

    # Apply texture if provided
    if texture_path is not None:
        try:
            if os.path.exists(texture_path):
                texture_id = p.loadTexture(texture_path)
                p.changeVisualShape(terrain, -1, textureUniqueId=texture_id)
                print(f"Successfully loaded texture: {texture_path}")
            else:
                print(f"Warning: Texture file not found: {texture_path}")
        except Exception as e:
            print(f"Warning: Failed to load texture {texture_path}: {e}")
            print("Continuing without texture...")

    return terrain


def _create_cylinder(
    radius: float,
    height: float,
    mass: float = 0.0,
    rgba: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
) -> Tuple[int, int]:
    """
    Create cylinder collision + visual shapes and return ids for a multibody create.

    Returns:
        (collision_shape_id, visual_shape_id)
    """
    col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=rgba)
    return col_id, vis_id


def _create_box_patch(
    size_xyz: Tuple[float, float, float],
    rgba: Tuple[float, float, float, float] = (0.4, 0.25, 0.1, 0.6),
) -> Tuple[int, int]:
    """
    Create a thin box collision & visual shape for patches (e.g., mud surface).
    """
    hx, hy, hz = (s / 2.0 for s in size_xyz)
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=rgba)
    return col_id, vis_id


def _set_dynamics(
    body_id: int,
    lateralFriction: Optional[float] = None,
    rollingFriction: Optional[float] = None,
    spinningFriction: Optional[float] = None,
    contactStiffness: Optional[float] = None,
    contactDamping: Optional[float] = None,
):
    kwargs: Dict[str, float] = {}
    if lateralFriction is not None:
        kwargs["lateralFriction"] = float(lateralFriction)
    if rollingFriction is not None:
        kwargs["rollingFriction"] = float(rollingFriction)
    if spinningFriction is not None:
        kwargs["spinningFriction"] = float(spinningFriction)
    if contactStiffness is not None:
        kwargs["contactStiffness"] = float(contactStiffness)
    if contactDamping is not None:
        kwargs["contactDamping"] = float(contactDamping)
    if kwargs:
        p.changeDynamics(body_id, -1, **kwargs)


def _place_cylinders_from_points(
    points_rc: List[Tuple[int, int]],
    heightfield: np.ndarray,
    mesh_scale: Tuple[float, float, float],
    base_position: Tuple[float, float, float],
    radius: float = 0.15,
    height: float = 1.2,
    mass: float = 0.0,
    rgba: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
    dynamics: Optional[Dict[str, float]] = None,
) -> List[int]:
    """
    Place cylinders centered at the selected (row, col) points, sitting on terrain surface.

    Returns:
        List of body ids.
    """
    body_ids: List[int] = []
    col_id, vis_id = _create_cylinder(radius=radius, height=height, mass=mass, rgba=rgba)
    for (r, c) in points_rc:
        x, y, z = _rc_to_world(r, c, heightfield, mesh_scale, base_position)
        # Cylinder base is at center; adjust z to place base on terrain
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[x, y, z + height / 2.0],
            baseOrientation=[0, 0, 0, 1],
        )
        if dynamics:
            _set_dynamics(body_id, **dynamics)
        body_ids.append(body_id)
    return body_ids


def _place_patch_box(
    center_rc: Tuple[int, int],
    size_xy: Tuple[float, float],
    thickness: float,
    heightfield: np.ndarray,
    mesh_scale: Tuple[float, float, float],
    base_position: Tuple[float, float, float],
    rgba: Tuple[float, float, float, float],
    dynamics: Optional[Dict[str, float]] = None,
) -> int:
    """
    Place a thin box patch aligned to world XY axes at the local terrain height.
    """
    col_id, vis_id = _create_box_patch((size_xy[0], size_xy[1], thickness), rgba=rgba)
    r, c = center_rc
    x, y, z = _rc_to_world(r, c, heightfield, mesh_scale, base_position)
    # Slight lift to avoid z-fighting with terrain
    body_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=[x, y, z + thickness / 2.0 + 1e-3],
        baseOrientation=[0, 0, 0, 1],
    )
    if dynamics:
        _set_dynamics(body_id, **dynamics)
    return body_id


def setup_terrain_from_image(
    image_path: str = "terrain/6ha.png",
    rows: int = 256,
    cols: int = 256,
    mesh_scale: Tuple[float, float, float] = (0.25, 0.25, 3.0),
    base_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    # base noise
    height_perturbation: float = 0.003,
    seed: int = 10,
    # trenches
    trench_depth: float = 0.18,
    trench_dilate_radius: int = 1,
    trench_dilate_iters: int = 2,
    # mask options (new)
    label_map_path: Optional[str] = None,
    label_trench_white_thresh: int = 200,
    label_cylinder_black_thresh: int = 55,
    trench_mask_path: Optional[str] = None,
    cylinder_mask_path: Optional[str] = None,
    trench_mask_white_is_fg: bool = True,
    cylinder_mask_black_is_fg: bool = True,
    binary_threshold: int = 127,
    # red cylinders (extracted from mask or fallback)
    red_min_distance_cells: int = 8,
    red_max_points: Optional[int] = None,
    cylinder_radius: float = 0.2,
    cylinder_height: float = 1.5,
    cylinder_mass: float = 0.0,
    cylinder_rgba: Tuple[float, float, float, float] = (0.85, 0.1, 0.1, 1.0),
    cylinder_dynamics: Optional[Dict[str, float]] = None,
    # background features
    pile_prob: float = 0.0015,
    pothole_prob: float = 0.0015,
    pile_radius_range: Tuple[float, float] = (0.12, 0.28),
    pile_height_range: Tuple[float, float] = (0.08, 0.22),
    pothole_radius_range: Tuple[float, float] = (0.15, 0.35),
    pothole_depth_range: Tuple[float, float] = (0.05, 0.14),
    mud_patch_size_xy: Tuple[float, float] = (0.35, 0.35),
    mud_patch_thickness: float = 0.015,
    pile_rgba: Tuple[float, float, float, float] = (0.35, 0.25, 0.18, 1.0),
    mud_patch_rgba: Tuple[float, float, float, float] = (0.25, 0.18, 0.12, 0.7),
    # dynamics presets
    terrain_dynamics: Optional[Dict[str, float]] = None,
    pile_dynamics: Optional[Dict[str, float]] = None,
    pothole_patch_dynamics: Optional[Dict[str, float]] = None,
    # visuals
    terrain_texture_path: Optional[str] = None,
) -> dict:
    """
    Build a terrain by projecting features from masks:
      - White in label_map or trench_mask -> trenches
      - Black in label_map or cylinder_mask (if black-is-fg) -> cylinders
      - Elsewhere -> random piles/potholes
    """
    rng = random.Random(seed)

    # 1) Base terrain
    heightfield = generate_procedural_heightfield(
        rows=rows, cols=cols, height_perturbation=height_perturbation, seed=seed
    )

    # 2) Feature masks (mask-first; fallback to color detection from image_path)
    blue_mask, red_mask = _extract_feature_masks(
        rows=rows,
        cols=cols,
        label_map_path=label_map_path,
        label_trench_white_thresh=label_trench_white_thresh,
        label_cylinder_black_thresh=label_cylinder_black_thresh,
        trench_mask_path=trench_mask_path,
        cylinder_mask_path=cylinder_mask_path,
        trench_mask_white_is_fg=trench_mask_white_is_fg,
        cylinder_mask_black_is_fg=cylinder_mask_black_is_fg,
        binary_threshold=binary_threshold,
        image_path_for_fallback=image_path,
    )

    # 3) Apply trenches from mask
    _apply_trenches_from_mask(
        heightfield,
        blue_trench_mask=blue_mask,
        trench_depth_world=trench_depth,
        mesh_scale=mesh_scale,
        dilate_radius=trench_dilate_radius,
        dilate_iters=trench_dilate_iters,
    )

    # 4) Background mask for piles/potholes
    background_mask = ~(blue_mask | red_mask)

    # 5) Random potholes: depress terrain and add thin "mud" patch with soft dynamics
    mud_patch_ids: List[int] = []
    if pothole_patch_dynamics is None:
        pothole_patch_dynamics = dict(
            lateralFriction=0.55,
            rollingFriction=0.03,
            spinningFriction=0.005,
            contactStiffness=2.0e3,
            contactDamping=40.0,
        )

    # 6) Random piles: small cylinders with adjusted dynamics
    mud_pile_ids: List[int] = []
    if pile_dynamics is None:
        pile_dynamics = dict(
            lateralFriction=0.9,
            rollingFriction=0.25,   # increases rolling resistance
            spinningFriction=0.02,
            contactStiffness=1.0e4,
            contactDamping=10.0,
        )

    # Iterate cells sparsely to avoid too many features
    for r in range(rows):
        for c in range(cols):
            if not background_mask[r, c]:
                continue
            roll = rng.random()
            # Decide pothole
            if roll < pothole_prob:
                radius = rng.uniform(*pothole_radius_range)
                depth = rng.uniform(*pothole_depth_range)
                _carve_disk(heightfield, (r, c), radius_world=radius, depth_world=depth, mesh_scale=mesh_scale, mode="depress")
                # Add thin soft patch (mud) at the pothole center
                patch_id = _place_patch_box(
                    (r, c),
                    size_xy=mud_patch_size_xy,
                    thickness=mud_patch_thickness,
                    heightfield=heightfield,
                    mesh_scale=mesh_scale,
                    base_position=base_position,
                    rgba=mud_patch_rgba,
                    dynamics=pothole_patch_dynamics,
                )
                mud_patch_ids.append(patch_id)
            # Else small pile
            elif roll < pothole_prob + pile_prob:
                radius = rng.uniform(*pile_radius_range)
                height_m = rng.uniform(*pile_height_range)
                # Build a small cylinder pile
                col_id, vis_id = _create_cylinder(radius=radius, height=height_m, mass=0.0, rgba=pile_rgba)
                x, y, z = _rc_to_world(r, c, heightfield, mesh_scale, base_position)
                body_id = p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=col_id,
                    baseVisualShapeIndex=vis_id,
                    basePosition=[x, y, z + height_m / 2.0],
                    baseOrientation=[0, 0, 0, 1],
                )
                _set_dynamics(body_id, **pile_dynamics)
                mud_pile_ids.append(body_id)

    # 7) Create the PyBullet terrain with optional terrain dynamics and texture
    terrain_id = create_pybullet_terrain(
        heightfield=heightfield,
        mesh_scale=mesh_scale,
        base_position=base_position,
        texture_path=terrain_texture_path,
        **(terrain_dynamics or {}),
    )

    # 8) Place tall cylinders at red dots (sparse centroids)
    cylinder_ids: List[int] = []
    red_points = _pick_sparse_points(
        red_mask, min_dist=red_min_distance_cells, max_points=red_max_points, seed=seed
    )
    if cylinder_dynamics is None:
        cylinder_dynamics = dict(
            lateralFriction=0.8,
            rollingFriction=0.01,
            spinningFriction=0.01,
            contactStiffness=5.0e4,
            contactDamping=200.0,
        )
    cylinder_ids = _place_cylinders_from_points(
        red_points,
        heightfield=heightfield,
        mesh_scale=mesh_scale,
        base_position=base_position,
        radius=cylinder_radius,
        height=cylinder_height,
        mass=cylinder_mass,
        rgba=cylinder_rgba,
        dynamics=cylinder_dynamics,
    )

    return {
        "terrain_id": terrain_id,
        "heightfield": heightfield,
        "cylinder_ids": cylinder_ids,
        "mud_pile_ids": mud_pile_ids,
        "mud_patch_ids": mud_patch_ids,
    }


# Backwards-compatible simple wrapper if you still want only terrain generation
def create_basic_terrain_with_texture(
    rows: int = 100,
    cols: int = 100,
    height_perturbation: float = 0.01,
    seed: int = 10,
    mesh_scale: Tuple[float, float, float] = (1.0, 1.0, 10.0),
    base_position: Tuple[float, float, float] = (0, 0, 0),
    texture_path: Optional[str] = None,
) -> Tuple[int, np.ndarray]:
    """
    Legacy/simple call to create only a procedural terrain and optional texture.
    """
    hf = generate_procedural_heightfield(rows, cols, height_perturbation, seed)
    terrain = create_pybullet_terrain(hf, mesh_scale=mesh_scale, base_position=base_position, texture_path=texture_path)
    return terrain, hf


# Example usage (uncomment to test manually):
# if __name__ == "__main__":
#     p.connect(p.GUI)
#     p.setGravity(0, 0, -9.81)
#     env = setup_terrain_from_image(
#         image_path="terrain/6ha.png",
#         rows=256,
#         cols=256,
#         mesh_scale=(0.25, 0.25, 3.0),
#         base_position=(0, 0, 0),
#         terrain_texture_path="terrain/6ha.png",  # optional: also use image as texture
#     )
#     print("Environment created:", {k: (v if not isinstance(v, np.ndarray) else v.shape) for k, v in env.items()})
#     while p.isConnected():
#         p.stepSimulation()