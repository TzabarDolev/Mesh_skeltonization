import matplotlib.pyplot as plt
import skeletor as sk
import trimesh


# reading a mesh to trimesh
mesh = trimesh.load('solid2.stl')

# Contract the mesh
cont = sk.contract(mesh, iter_lim=1)
# Extract the skeleton from the contracted mesh
swc = sk.skeletonize(cont, method='vertex_clusters', sampling_dist=1, output='swc')
# Clean up the skeleton
swc = sk.clean(swc, mesh)
# Add/update radii
swc['radius'] = sk.radii(swc, mesh, method='knn', n=5, aggregate='mean')
swc.head()

# visualizing
max_val = max(max(swc.x), max(swc.y), max(swc.z))
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.axis('off')
plt.xlim([-max_val, max_val])
plt.ylim([-max_val, max_val])
ax.set_zlim(-max_val, max_val)
ax.scatter(swc.x, swc.y, swc.z, s=10)

plt.show()

