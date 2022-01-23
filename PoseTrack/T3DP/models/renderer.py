import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh

def create_raymond_lights():
    import pyrender
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
    nodes = []
    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)
        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
          light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
          matrix=matrix
        ))
    return nodes

class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length=5000, img_res=224, faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res,
                                       viewport_height=img_res,
                                       point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = [img_res // 2, img_res // 2]
        self.faces = faces
    
    def visualize_all(self, vertices, camera_translation, color, images, baseColorFactors=(1.0, 1.0, 0.9, 1.0)):

        baseColorFactors = np.hstack([color[:, [2,1,0]], np.ones((color.shape[0], 1))])
        
        images_np = images #np.transpose(images, (1,2,0))
        fl = self.focal_length
        verts = vertices.copy()
        cam_trans = camera_translation.copy()
        verts = verts + cam_trans#[:,None]
        
        color = self.__call__(verts, image=images_np, focal_length=fl, baseColorFactors=baseColorFactors)
        
        valid_mask = color.sum(axis=2, keepdims=True)>0
        output_img = color[:, :, :3] * valid_mask + (1 - valid_mask) * images_np
        return output_img
    
    def __call__(self, vertices, image, focal_length=5000, baseColorFactors=[(1.0, 1.0, 0.9, 1.0)]):
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        
        for i_, verts in enumerate(vertices):
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode='OPAQUE',
                baseColorFactor=baseColorFactors[i_])
            
            mesh = trimesh.Trimesh(verts.copy(), self.faces.copy())
            rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
            mesh.apply_transform(rot)
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            scene.add(mesh, 'mesh')
            
        camera_pose = np.eye(4)
        # camera_pose[:3, 3] = camera_translation
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length, cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        
        output_img = color[:, :, :3]

        return output_img
