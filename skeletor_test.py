import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import skeletor as sk
import trimesh


# reading a mesh to trimesh
mesh_name = 'solid.stl'
mesh = trimesh.load(mesh_name)
mesh_o3d = o3d.io.read_triangle_mesh(mesh_name)

# Contract the mesh
cont = sk.contract(mesh, iter_lim=15)
# Extract the skeleton from the contracted mesh
swc = sk.skeletonize(cont, method='vertex_clusters', sampling_dist=1, output='swc')
# swc = sk.skeletonize(cont, method='edge_collapse', output='swc')
# Clean up the skeleton
swc = sk.clean(swc, mesh)
# Add/update radii
swc['radius'] = sk.radii(swc, mesh, method='knn', n=5, aggregate='mean')
swc.head()

# visualizing scattered points
max_val = max(max(swc.x), max(swc.y), max(swc.z))
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.axis('off')
plt.xlim([-max_val, max_val])
plt.ylim([-max_val, max_val])
ax.set_zlim(-max_val, max_val)
ax.scatter(swc.x, swc.y, swc.z, s=10)
plt.show()

# visualizing using open3d
xyz = np.zeros((swc.x.size, 3))
xyz[:, 0] = swc.x
xyz[:, 1] = swc.y
xyz[:, 2] = swc.z
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
mesh_o3d.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_o3d, pcd])
o3d.visualization.draw_geometries([pcd])

# exporting csv file
export_file_path = mesh_name[:-4] + '.csv'
swc.to_csv(export_file_path, index=False, header=True)
