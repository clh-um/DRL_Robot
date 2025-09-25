# terrain_generator.py
import numpy as np
import random
import pybullet as p


def generate_procedural_heightfield(
    rows=100,
    cols=100,
    height_perturbation=0.01,
    seed=10
):
    """
    Generate a procedural heightfield using random perturbation.

    Args:
        rows (int): Number of rows in heightfield.
        cols (int): Number of columns in heightfield.
        height_perturbation (float): Noise amplitude (raw values).
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: 2D heightfield data.
    """
    random.seed(seed)
    heightfield = np.zeros((cols, rows), dtype=np.float32)

    # Generate the full terrain procedurally (no mirroring)
    for i in range(cols):
        for j in range(rows):
            n1 = heightfield[i, j - 1] if j > 0 else 0
            n2 = heightfield[i - 1, j] if i > 0 else n1
            noise = random.uniform(-height_perturbation, height_perturbation)
            heightfield[i, j] = (n1 + n2) / 2 + noise

    return heightfield

def create_pybullet_terrain(
    heightfield,
    mesh_scale=(1.0, 1.0, 10.0),
    base_position=(0, 0, 0),
    texture_path=None
):
    rows, cols = heightfield.shape
    heightfield_flat = heightfield.reshape(-1)

    # Collision shape (also renders in GUI)
    terrain_collision = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        heightfieldData=heightfield_flat,
        meshScale=list(mesh_scale),
        numHeightfieldRows=rows,
        numHeightfieldColumns=cols
    )

    # Create terrain body
    terrain = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=terrain_collision
    )

    p.resetBasePositionAndOrientation(terrain, base_position, [0, 0, 0, 1])

    # Apply texture to the collision shape if provided
    if texture_path is not None:
        try:
            import os
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



