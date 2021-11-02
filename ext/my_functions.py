import os
import numpy as np
import nibabel as nib
import cv2
import torch
import scipy.ndimage

###############################3

# Crop label volume
def cropLabelVol(V,
                 margin=10,
                 threshold=0):

    # Make sure it's 3D
    margin = np.array(margin)
    if len(margin.shape) < 2:
        margin = [margin, margin, margin]

    if len(V.shape) < 2:
        V = V[..., np.newaxis]
    if len(V.shape) < 3:
        V = V[..., np.newaxis]

    # Now
    idx = np.where(V > threshold)
    i1 = np.max([0, np.min(idx[0]) - margin[0]]).astype('int')
    j1 = np.max([0, np.min(idx[1]) - margin[1]]).astype('int')
    k1 = np.max([0, np.min(idx[2]) - margin[2]]).astype('int')
    i2 = np.min([V.shape[0], np.max(idx[0]) + margin[0] + 1]).astype('int')
    j2 = np.min([V.shape[1], np.max(idx[1]) + margin[1] + 1]).astype('int')
    k2 = np.min([V.shape[2], np.max(idx[2]) + margin[2] + 1]).astype('int')

    cropping = [i1, j1, k1, i2, j2, k2]
    cropped = V[i1:i2, j1:j2, k1:k2]

    return cropped, cropping

###############################3

def applyCropping(V, cropping):
    i1 = cropping[0]
    j1 = cropping[1]
    k1 = cropping[2]
    i2 = cropping[3]
    j2 = cropping[4]
    k2 = cropping[5]

    if len(V.shape)>2:
        Vcropped = V[i1:i2, j1: j2, k1: k2, ...]
    else:
        Vcropped = V[i1:i2, j1: j2]

    return Vcropped

###############################3

def viewVolume(x, aff=None):

    if aff is None:
        aff = np.eye(4)
    else:
        if type(aff) == torch.Tensor:
            aff = aff.cpu().detach().numpy()

    if type(x) is not list:
        x = [x]

    cmd = 'source /usr/local/freesurfer/nmr-dev-env-bash && freeview '

    for n in np.arange(len(x)):
        vol = x[n]
        if type(vol) == torch.Tensor:
            vol = vol.cpu().detach().numpy()
        vol = np.squeeze(np.array(vol))
        name = '/tmp/' + str(n) + '.nii.gz'
        MRIwrite(vol, aff, name)
        cmd = cmd + ' ' + name

    os.system(cmd + ' &')

###############################3

def MRIwrite(volume, aff, filename, dtype=None):

    if dtype is not None:
        volume = volume.astype(dtype=dtype)

    if aff is None:
        aff = np.eye(4)
    header = nib.Nifti1Header()
    nifty = nib.Nifti1Image(volume, aff, header)

    nib.save(nifty, filename)

###############################3

def MRIread(filename, dtype=None, im_only=False):

    assert filename.endswith(('.nii', '.nii.gz', '.mgz')), 'Unknown data file: %s' % filename

    x = nib.load(filename)
    volume = x.get_data()
    aff = x.affine

    if dtype is not None:
        volume = volume.astype(dtype=dtype)

    if im_only:
        return volume
    else:
        return volume, aff

###############################3

def downsampleMRI2d(X, aff, shape, factors, mode='image'):

    assert False, 'Function not debugged/tested yet...'

    assert mode=='image' or mode=='labels', 'Mode must be image or labels'
    assert (shape is None) or (factors is None), 'Either shape or factors must be None'
    assert (shape is not None) or (factors is not None), 'Either shape or factors must be not None'

    if shape is not None:
        factors = np.array(shape) / X.shape[0:2]
    else:
        factors = np.array(factors)
        shape = np.round(X.shape[0:2] * factors).astype('int')

    if mode == 'image':
        if np.mean(factors) < 1: # shrink
            Y = cv2.resize(X, shape, interpolation=cv2.INTER_AREA)
        else:  # expan
            Y = cv2.resize(X, shape, interpolation=cv2.INTER_LINEAR)
    else:
        Y = cv2.resize(X, shape, interpolation=cv2.INTER_NEAREST)

    aff2 = aff
    aff2[:, 0] = aff2[:, 0] * factors[0]
    aff2[:, 1] = aff2[:, 1] * factors[1]
    aff2[0:3, 3] = aff2[0:3, 3] + aff[0:3, 0:3] * (0.5*np.array([[factors[0]], [factors[1]], [1]])-0.5)

    return Y, aff2

###############################3

def vox2ras(vox, vox2ras):

    vox2 = np.concatenate([vox, np.ones(shape=[1, vox.shape[1]])], axis=0)

    ras = np.matmul(vox2ras, vox2)[:-1, :]

    return ras

###############################

def ras2vox(ras, vox2ras):

    ras2 = np.concatenate([ras, np.ones(shape=[1, ras.shape[1]])], axis=0)

    vox = np.matmul(np.linalg.inv(vox2ras), ras2)[:-1, :]

    return vox


###############################3

def prepBiasFieldBase2d(siz, max_order):
    x = np.linspace(-1, 1, siz[0])
    y = np.linspace(-1, 1, siz[1])
    xx, yy = np.meshgrid(x, y, indexing='ij')
    PSI = []
    for o in range(max_order + 1):
        for ox in range(o + 1):
            for oy in range(o + 1):
                if (ox + oy) == o:
                    psi = np.ones(siz)
                    for i in range(1, ox + 1):
                        psi = psi * xx
                    for j in range(1, oy + 1):
                        psi = psi * yy
                    PSI.append(psi)

    PSI = np.stack(PSI, axis=-1)

    return PSI

###############################3

def grad3d(X, provide_gradients=False):
    h = np.array([-1, 0, 1])
    Gx = scipy.ndimage.convolve(X, np.reshape(h, [3, 1, 1]))
    Gy = scipy.ndimage.convolve(X, np.reshape(h, [1, 3, 1]))
    Gz = scipy.ndimage.convolve(X, np.reshape(h, [1, 1, 3]))
    Gmodule = np.sqrt(Gx * Gx + Gy * Gy + Gz * Gz)

    if provide_gradients:
        return Gmodule, Gx, Gy, Gz
    else:
        return Gmodule

###############################3

def grad2d(X, provide_gradients=False):
    h = np.array([-1, 0, 1])
    Gx = scipy.ndimage.convolve(X, np.reshape(h, [3, 1]))
    Gy = scipy.ndimage.convolve(X, np.reshape(h, [1, 3]))
    Gmodule = np.sqrt(Gx * Gx + Gy * Gy)

    if provide_gradients:
        return Gmodule, Gx, Gy
    else:
        return Gmodule
