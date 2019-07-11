from mayavi import mlab
from .body_models import create
import torch
import os
import numpy as np

@mlab.animate(delay=10)
def anim(verts,s):
    for i in range(verts.shape[0]):
        s.mlab_source.x = verts[i,:,0].data.numpy()
        s.mlab_source.y = verts[i,:,1].data.numpy()
        s.mlab_source.z = verts[i,:,2].data.numpy()
        yield

def smpl_anim(pose,orient=None,trans=None,shape=None):
    pose = torch.tensor(pose,dtype=torch.float32)
    if trans is not None:
        trans = torch.tensor(trans,dtype=torch.float32)
    else:
        trans = torch.zeros([pose.shape[0],3],dtype=torch.float32)

    if orient is not None:
        orient = torch.tensor(orient,dtype=torch.float32)
    else:
        orient = torch.zeros([pose.shape[0],3],dtype=torch.float32)
    if shape is not None:
        shape = torch.tensor(shape,dtype=torch.float32)
        if shape.ndimension() == 1:
            shape = shape[None,:]
    smpl = create('smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    smpl_out = smpl.forward(body_pose=pose,global_orient=orient,transl=trans,betas=shape,get_skin=True)
    msh = smpl_out[0]
    s = mlab.triangular_mesh(msh[0,:,0].data.numpy(),msh[0,:,1].data.numpy(),msh[0,:,2].data.numpy(),smpl.faces)
    anim(msh,s)
    mlab.show()


@mlab.animate(delay=10)
def anim2(verts,s):
    for i in range(verts[0].shape[0]):
        for j in range(len(s)):
            s[j].mlab_source.x = verts[j][i,:,0].data.numpy()
            s[j].mlab_source.y = verts[j][i,:,1].data.numpy()
            s[j].mlab_source.z = verts[j][i,:,2].data.numpy()
        yield

def smpl_anim2(pose,trans=None,shape=None):
    smpl = create('smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')

    msh = []
    s = []
    for i in range(len(pose)):
        pose_tens = torch.tensor(pose[i],dtype=torch.float32)
        if trans is not None:
            trans = torch.tensor(trans,dtype=torch.float32)
        else:
            trans = torch.zeros([pose.shape[0],3],dtype=torch.float32)
        if shape is not None:
            shape_tens = torch.tensor(shape[i],dtype=torch.float32)
            if shape_tens.ndimension() == 1:
                shape_tens = shape_tens[None,:]
        else:
            shape_tens = None
        m = smpl.forward(pose=pose_tens,trans=trans_tens,betas=shape_tens,get_skin=True)
        msh.append(m[0])
        s.append(mlab.triangular_mesh(msh[i][0,:,0].data.numpy(),msh[i][0,:,1].data.numpy(),msh[i][0,:,2].data.numpy(),smpl.faces))

    anim2(msh,s)
    mlab.show()



@mlab.animate(delay=10)
def anim3(verts,s,save_img,dir_name):
    for i in range(verts[0].shape[0]):
        for j in range(len(s)):
            s[j].mlab_source.x = verts[j][i,:,0].data.numpy()
            s[j].mlab_source.y = verts[j][i,:,1].data.numpy()
            s[j].mlab_source.z = verts[j][i,:,2].data.numpy()
        if save_img:
            mlab.savefig(filename=os.path.join(dir_name,'{:04d}'.format(i)+'.png'))
        yield

def smpl_anim3(pose,trans,shape=None,save_img=False,dir_name=None):

    if save_img:
        assert dir_name is not None

    os.mkdir(dir_name)

    smpl = torchSMPL.create('smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')

    mcolors = np.random.rand(len(pose),3)

    msh = []
    s = []
    for i in range(len(pose)):
        pose_tens = torch.tensor(pose[i],dtype=torch.float32) 
        if trans is not None:
            trans = torch.tensor(trans,dtype=torch.float32)
        else:
            trans = torch.zeros([pose.shape[0],3],dtype=torch.float32)
        if shape is not None:
            shape_tens = torch.tensor(shape[i],dtype=torch.float32)
            if shape_tens.ndimension() == 1:
                shape_tens = shape_tens[None,:]
        else:
            shape_tens = None
        m = smpl.forward(pose=pose_tens,trans=trans_tens,betas=shape_tens,get_skin=True)
        msh.append(m[0])
        s.append(mlab.triangular_mesh(msh[i][0,:,0].data.numpy(),msh[i][0,:,1].data.numpy(),msh[i][0,:,2].data.numpy(),smpl.faces,color=tuple(mcolors[i])))

    anim3(msh,s,save_img,dir_name)
    mlab.show()
