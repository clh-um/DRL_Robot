import coacd
import trimesh
import numpy as np

input_file = "terrain5.obj"
output_file = "terrain5_convex.obj"

# Load your mesh using trimesh
mesh = trimesh.load(input_file, force="mesh")

# Convert to coacd.Mesh
coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)

# Run convex decomposition (no extra keyword arguments)
parts = coacd.run_coacd(coacd_mesh)

# Combine all convex parts into a single mesh
all_vertices = []
all_faces = []
vertex_offset = 0

for part in parts:
    vertices, faces = part  # unpack tuple
    vertices = np.array(vertices)
    faces = np.array(faces)

    all_vertices.append(vertices)
    all_faces.append(faces + vertex_offset)
    vertex_offset += len(vertices)

all_vertices = np.vstack(all_vertices)
all_faces = np.vstack(all_faces)

# Export the final convex OBJ
convex_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
convex_mesh.export(output_file)

print(f"Convex decomposition saved to {output_file}")
