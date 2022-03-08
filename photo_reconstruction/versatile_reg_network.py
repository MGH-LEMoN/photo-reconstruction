import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as nnf


class PhotoAligner(nn.Module):
    """
    Main class to perform alignment
    """

    def __init__(self,
                 photo_vol,
                 photo_mask_vol,
                 photo_aff,
                 mri_vol,
                 mri_aff,
                 ref_surf,
                 device='cpu',
                 allow_scaling_and_shear=False,
                 allow_sz=False,
                 k_dice_mri=0.8,
                 k_ncc_intermodality=0.8,
                 k_surface_term=0.8,
                 k_dice_slices=0.1,
                 k_ncc_slices=0.1,
                 k_regularizer=0.01,
                 pad_ignore=None,
                 t_ini=None,
                 theta_ini=None,
                 shear_ini=None,
                 scaling_ini=None,
                 sz_ini=None,
                 ref_type='mask',
                 allow_s_reference=False,
                 t_reference_ini=None,
                 theta_reference_ini=None,
                 s_reference_ini=None,
                 allow_nonlin=False,
                 field_ini=None,
                 k_nonlinear=0.1):

        super().__init__()

        self.device = device
        self.ref_type = ref_type
        self.photo_vol = torch.Tensor(photo_vol).to(self.device)
        self.photo_rearranged = torch.unsqueeze(self.photo_vol.permute(
            3, 0, 1, 2),
                                                dim=0).to(self.device)
        self.photo_mask_vol = torch.Tensor(photo_mask_vol).to(self.device)
        self.mask_rearranged = torch.unsqueeze(torch.unsqueeze(
            self.photo_mask_vol, dim=0),
                                               dim=0).to(self.device)
        self.photo_aff = torch.Tensor(photo_aff).to(self.device)
        if ref_type == 'surface':
            self.ref_surf = torch.Tensor(ref_surf).to(self.device)
        else:
            self.ref_surf = None
        self.mri_vol = torch.Tensor(mri_vol).to(self.device)
        self.mri_rearranged = torch.unsqueeze(torch.unsqueeze(self.mri_vol,
                                                              dim=0),
                                              dim=0).to(self.device)
        self.mri_aff = torch.Tensor(mri_aff).to(self.device)
        self.allow_scaling_and_shear = allow_scaling_and_shear
        self.photo_siz = photo_mask_vol.shape[:-1]
        self.Nslices = photo_mask_vol.shape[-1]
        self.k_dice_mri = k_dice_mri
        self.k_dice_slices = k_dice_slices
        self.k_ncc_slices = k_ncc_slices
        self.k_regularizer = k_regularizer
        self.k_ncc_intermodality = k_ncc_intermodality
        self.k_surface_term = k_surface_term
        self.allow_nonlin = allow_nonlin
        self.k_nonlinear = k_nonlinear

        if pad_ignore is not None:
            self.pad_ignore = [pad_ignore, pad_ignore]
        else:
            # Discover automatically
            idx = np.argwhere(
                np.sum(np.sum(photo_mask_vol, axis=0), axis=0) > 0)
            self.pad_ignore = [
                np.min(idx), photo_mask_vol.shape[-1] - 1 - np.max(idx)
            ]

        if t_ini is not None:
            self.t = torch.nn.Parameter(torch.tensor(t_ini).to(self.device))
        else:
            self.t = torch.nn.Parameter(
                torch.zeros(2, self.Nslices).to(self.device))
        self.t.requires_grad = True

        if theta_ini is not None:
            self.theta = torch.nn.Parameter(
                torch.tensor(theta_ini).to(self.device))
        else:
            self.theta = torch.nn.Parameter(
                torch.zeros(self.Nslices).to(self.device))
        self.theta.requires_grad = True

        if allow_scaling_and_shear:
            if shear_ini is not None:
                self.shear = torch.nn.Parameter(
                    torch.tensor(shear_ini).to(self.device))
            else:
                self.shear = torch.nn.Parameter(
                    torch.zeros(2, self.Nslices).to(self.device))
            self.shear.requires_grad = True
            if scaling_ini is not None:
                self.scaling = torch.nn.Parameter(
                    torch.tensor(scaling_ini).to(self.device))
            else:
                self.scaling = torch.nn.Parameter(
                    torch.zeros(2, self.Nslices).to(self.device))
            self.scaling.requires_grad = True
        else:
            if shear_ini is not None:
                self.shear = torch.tensor(shear_ini).to(self.device)
            else:
                self.shear = torch.zeros(2, self.Nslices).to(self.device)
            if scaling_ini is not None:
                self.scaling = torch.tensor(scaling_ini).to(self.device)
            else:
                self.scaling = torch.zeros(2, self.Nslices).to(self.device)

        if allow_sz:
            if sz_ini is not None:
                self.sz = torch.nn.Parameter(
                    torch.tensor(sz_ini).to(self.device))
            else:
                self.sz = torch.nn.Parameter(torch.zeros(1).to(self.device))
            self.sz.requires_grad = True
        else:
            if sz_ini is not None:
                self.sz = torch.tensor(sz_ini).to(self.device)
            else:
                self.sz = torch.zeros(1).to(self.device)

        if allow_s_reference:
            if s_reference_ini is not None:
                self.s_reference = torch.nn.Parameter(
                    torch.tensor(s_reference_ini).to(self.device))
            else:
                self.s_reference = torch.nn.Parameter(
                    torch.zeros(1).to(self.device))
            self.s_reference.requires_grad = True
        else:
            if s_reference_ini is not None:
                self.s_reference = torch.tensor(s_reference_ini).to(
                    self.device)
            else:
                self.s_reference = torch.zeros(1).to(self.device)

        if t_reference_ini is not None:
            self.t_reference = torch.nn.Parameter(
                torch.tensor(t_reference_ini).to(self.device))
        else:
            self.t_reference = torch.nn.Parameter(
                torch.zeros(3).to(self.device))
        self.t_reference.requires_grad = True

        if theta_reference_ini is not None:
            self.theta_reference = torch.nn.Parameter(
                torch.tensor(theta_reference_ini).to(self.device))
        else:
            self.theta_reference = torch.nn.Parameter(
                torch.zeros(3).to(self.device))
        self.theta_reference.requires_grad = True

        if allow_nonlin:
            if field_ini is not None:
                self.field = torch.nn.Parameter(
                    torch.tensor(field_ini).to(self.device))
            else:
                self.field = torch.nn.Parameter(
                    torch.zeros(5, 5, 2, self.Nslices).to(self.device))
            self.field.requires_grad = True
        else:
            if field_ini is not None:
                self.field = torch.tensor(field_ini).to(self.device)
            else:
                self.field = None

        # create sampling grid
        vectors = [torch.arange(0, s) for s in self.photo_mask_vol.shape]
        self.grids = torch.stack(torch.meshgrid(vectors)).to(self.device)

    def forward(self):

        # We scale angles / shearings / scalings as a simple form of preconditioning (which shouldn't be needed with bfgs, but whatever...)

        # Parameters of photos
        theta_f = self.theta / 180 * torch.tensor(np.pi)  # degrees -> radians
        shear_f = self.shear / 100  # percentages
        scaling_f = torch.exp(
            self.scaling /
            20)  # ensures positive and symmetry around 1 in log scale
        t_f = self.t  # no scaling
        sz_f = torch.exp(
            self.sz /
            20)  # ensures positive and symmetry around 1 in log scale

        # Parameters of reference volume, if it exists
        theta_reference_f = self.theta_reference / 180 * torch.tensor(
            np.pi)  # degrees -> radians
        t_reference_f = self.t_reference  # no scaling
        s_reference_f = torch.exp(
            self.s_reference /
            20)  # ensures positive and symmetry around 1 in log scale
        if self.field is not None:
            # make it a % of the dimensions
            field_x = self.field[0, :, :, :] * torch.Tensor(
                np.array(self.photo_siz[0] / 100.0))
            field_y = self.field[1, :, :, :] * torch.Tensor(
                np.array(self.photo_siz[1] / 100.0))
            field_pixels = torch.stack([field_x, field_y])
        else:
            field_pixels = None

        if torch.any(torch.isnan(theta_f)) or torch.any(torch.isnan(shear_f)) or torch.any(torch.isnan(scaling_f))  \
                or torch.any(torch.isnan(t_f)) or torch.any(torch.isnan(sz_f))    or torch.any(torch.isnan(theta_reference_f)) \
                or torch.any(torch.isnan(t_reference_f))  or torch.any(torch.isnan(s_reference_f)) :
            kk = 1

        # Prepare 2D matrices for the photos
        M = torch.zeros([4, 4, self.Nslices]).to(self.device)
        M[0, 0, :] = scaling_f[0, :] * (torch.cos(theta_f) -
                                        shear_f[0, :] * torch.sin(theta_f))
        M[0, 2, :] = scaling_f[0, :] * (
            shear_f[1, :] * torch.cos(theta_f) -
            (1 + shear_f[0, :] * shear_f[1, :]) * torch.sin(theta_f))
        M[0, 3, :] = t_f[0, :]
        M[1, 1, :] = 1
        M[2, 0, :] = scaling_f[1, :] * (torch.sin(theta_f) +
                                        shear_f[0, :] * torch.cos(theta_f))
        M[2, 2, :] = scaling_f[1, :] * (
            shear_f[1, :] * torch.sin(theta_f) +
            (1 + shear_f[0, :] * shear_f[1, :]) * torch.cos(theta_f))
        M[2, 3, :] = t_f[1, :]
        M[3, 3, :] = 1

        # update mesh grids for photos
        # First, upscale field
        if self.field is not None:

            field_fullsiz = torch.nn.Upsample(size=self.photo_vol.shape[:-2],
                                              align_corners=True,
                                              mode='bilinear')(field_pixels)
            field_fullsiz_rearranged = field_fullsiz.permute(0, 2, 3, 1)

            # cost_field = torch.mean(torch.sqrt(torch.clamp(field_pixels * field_pixels, min=1e-5)))
            grad_xx = (field_fullsiz[0, 2:, 1:-1, :] -
                       field_fullsiz[0, :-2, 1:-1, :]) / 2.0
            grad_xy = (field_fullsiz[0, 1:-1, 2:, :] -
                       field_fullsiz[0, 1:-1, :-2, :]) / 2.0
            grad_x = torch.sqrt(
                torch.clamp(grad_xx * grad_xx + grad_xy * grad_xy, min=1e-5))
            grad_yx = (field_fullsiz[1, 2:, 1:-1, :] -
                       field_fullsiz[1, :-2, 1:-1, :]) / 2.0
            grad_yy = (field_fullsiz[1, 1:-1, 2:, :] -
                       field_fullsiz[1, 1:-1, :-2, :]) / 2.0
            grad_y = torch.sqrt(
                torch.clamp(grad_yx * grad_yx + grad_yy * grad_yy, min=1e-5))
            cost_field = torch.mean(grad_x) + torch.mean(grad_y)

        else:
            field_fullsiz = None
            field_fullsiz_rearranged = None
            cost_field = torch.zeros(1).to(self.device)

        # update mesh grids for photos
        grids_new = torch.zeros(self.grids.shape).to(self.device)
        photo_aff = torch.zeros(4, 4).to(self.device)
        photo_aff[:, :] = self.photo_aff
        photo_aff[1, 2] = self.photo_aff[1, 2] * sz_f
        T = torch.zeros([4, 4, self.Nslices]).to(self.device)
        for z in range(self.Nslices):
            T[:, :, z] = torch.matmul(
                torch.matmul(torch.inverse(photo_aff), M[:, :, z]), photo_aff)
            if field_fullsiz_rearranged is None:
                for d in range(3):
                    grids_new[d, :, :, z] = T[d, 0, z] * self.grids[
                        0, :, :, z] + T[d, 1, z] * self.grids[1, :, :, z] + T[
                            d, 2, z] * self.grids[2, :, :, z] + T[d, 3, z]
            else:
                for d in range(2):
                    grids_new[d, :, :, z] = T[d, 0, z] * self.grids[
                        0, :, :, z] + T[d, 1, z] * self.grids[1, :, :, z] + T[
                            d, 2, z] * self.grids[2, :, :, z] + T[
                                d, 3, z] + field_fullsiz_rearranged[d, :, :, z]
                grids_new[2, :, :, z] = T[2, 0, z] * self.grids[
                    0, :, :, z] + T[2, 1, z] * self.grids[1, :, :, z] + T[
                        2, 2, z] * self.grids[2, :, :, z] + T[2, 3, z]

        # Resample photos and masks
        # We need to make the new grid compatible with grid_resample...
        grids_new = torch.unsqueeze(grids_new, 0)
        grids_new = grids_new.permute(0, 2, 3, 4, 1)
        for i in range(3):
            grids_new[:, :, :, :,
                      i] = 2 * (grids_new[:, :, :, :, i] /
                                (self.photo_mask_vol.shape[i] - 1) - 0.5)
        # Not sure why, but channels need to be reversed
        grids_new = grids_new[..., [2, 1, 0]]
        photo_resampled = nnf.grid_sample(self.photo_rearranged,
                                          grids_new,
                                          align_corners=True,
                                          mode='bilinear',
                                          padding_mode='zeros')
        photo_resampled = torch.squeeze(photo_resampled.permute(2, 3, 4, 1, 0))
        mask_resampled = nnf.grid_sample(self.mask_rearranged,
                                         grids_new,
                                         align_corners=True,
                                         mode='bilinear',
                                         padding_mode='zeros')
        mask_resampled = torch.squeeze(mask_resampled)

        # Now work on the reference
        Rx = torch.zeros([4, 4]).to(self.device)
        Rx[0, 0] = 1
        Rx[1, 1] = torch.cos(theta_reference_f[0])
        Rx[1, 2] = -torch.sin(theta_reference_f[0])
        Rx[2, 1] = torch.sin(theta_reference_f[0])
        Rx[2, 2] = torch.cos(theta_reference_f[0])
        Rx[3, 3] = 1

        Ry = torch.zeros([4, 4]).to(self.device)
        Ry[0, 0] = torch.cos(theta_reference_f[1])
        Ry[0, 2] = torch.sin(theta_reference_f[1])
        Ry[1, 1] = 1
        Ry[2, 0] = -torch.sin(theta_reference_f[1])
        Ry[2, 2] = torch.cos(theta_reference_f[1])
        Ry[3, 3] = 1

        Rz = torch.zeros([4, 4]).to(self.device)
        Rz[0, 0] = torch.cos(theta_reference_f[2])
        Rz[0, 1] = -torch.sin(theta_reference_f[2])
        Rz[1, 0] = torch.sin(theta_reference_f[2])
        Rz[1, 1] = torch.cos(theta_reference_f[2])
        Rz[2, 2] = 1
        Rz[3, 3] = 1

        trans_and_scale = torch.eye(4).to(self.device)
        trans_and_scale[:-1, -1] = t_reference_f
        trans_and_scale[0, 0] = s_reference_f
        trans_and_scale[1, 1] = s_reference_f
        trans_and_scale[2, 2] = s_reference_f

        Rt = torch.matmul(trans_and_scale,
                          torch.matmul(torch.matmul(Rx, Ry), Rz))

        if self.ref_type == 'surface':

            # Combine transformation of surface with ras2vox
            combo = torch.matmul(torch.inverse(photo_aff), Rt)
            xx = combo[0, 0] * self.ref_surf[:, 0] + combo[
                0, 1] * self.ref_surf[:, 1] + combo[
                    0, 2] * self.ref_surf[:, 2] + combo[0, 3]
            yy = combo[1, 0] * self.ref_surf[:, 0] + combo[
                1, 1] * self.ref_surf[:, 1] + combo[
                    1, 2] * self.ref_surf[:, 2] + combo[1, 3]
            zz = combo[2, 0] * self.ref_surf[:, 0] + combo[
                2, 1] * self.ref_surf[:, 1] + combo[
                    2, 2] * self.ref_surf[:, 2] + combo[2, 3]

            ok = (xx > 0) & (yy > 0) & (zz > 0) & (
                xx < photo_resampled.shape[0] - 1) & (
                    yy < photo_resampled.shape[1] - 1) & (
                        zz < photo_resampled.shape[2] - 1)

            x = xx[ok]
            y = yy[ok]
            z = zz[ok]

            # Interpolate in a nice, differentiable way. We follow https://en.wikipedia.org/wiki/Trilinear_interpolation
            xf = torch.floor(x).long()
            yf = torch.floor(y).long()
            zf = torch.floor(z).long()
            xd = (x - xf).unsqueeze(1)
            yd = (y - yf).unsqueeze(1)
            zd = (z - zf).unsqueeze(1)

            c000 = photo_resampled[xf, yf, zf, :]
            c001 = photo_resampled[xf, yf, zf + 1, :]
            c010 = photo_resampled[xf, yf + 1, zf, :]
            c011 = photo_resampled[xf, yf + 1, zf + 1, :]
            c100 = photo_resampled[xf + 1, yf, zf, :]
            c101 = photo_resampled[xf + 1, yf, zf + 1, :]
            c110 = photo_resampled[xf + 1, yf + 1, zf, :]
            c111 = photo_resampled[xf + 1, yf + 1, zf + 1, :]

            c00 = c000 * (1 - xd) + c100 * xd
            c01 = c001 * (1 - xd) + c101 * xd
            c10 = c010 * (1 - xd) + c110 * xd
            c11 = c011 * (1 - xd) + c111 * xd

            c0 = c00 * (1 - yd) + c10 * yd
            c1 = c01 * (1 - yd) + c11 * yd

            c = c0 * (1 - zd) + c1 * zd

            av_grad = torch.sum(c) / torch.numel(ok) / 3.0

        # Now let's work on the deformed volume
        grids_new_mri = torch.zeros(self.grids.shape).to(self.device)
        mri_aff_combined = torch.matmul(Rt, self.mri_aff)
        D = torch.matmul(torch.inverse(mri_aff_combined), photo_aff)

        for d in range(3):
            grids_new_mri[d, :, :, :] = D[d, 0] * self.grids[0, :, :, :] + D[
                d, 1] * self.grids[1, :, :, :] + D[d, 2] * self.grids[
                    2, :, :, :] + grids_new_mri[d, :, :, :] + D[d, 3]

        grids_new_mri = torch.unsqueeze(grids_new_mri, 0)
        grids_new_mri = grids_new_mri.permute(0, 2, 3, 4, 1)
        for i in range(3):
            grids_new_mri[:, :, :, :,
                          i] = 2 * (grids_new_mri[:, :, :, :, i] /
                                    (self.mri_vol.shape[i] - 1) - 0.5)
        # Not sure why, but channels need to be reversed
        grids_new_mri = grids_new_mri[..., [2, 1, 0]]
        mri_resampled = nnf.grid_sample(self.mri_rearranged,
                                        grids_new_mri,
                                        align_corners=True,
                                        mode='bilinear',
                                        padding_mode='zeros')
        mri_resampled = torch.squeeze(mri_resampled)

        if self.ref_type == 'image':

            # I now do slice-wise ncc, rather than slice-wise lncc

            # ncc_mri = slice_wise_LNCC(mri_resampled[:, :, self.pad_ignore[0]:-self.pad_ignore[1]], photo_resampled[:, :, self.pad_ignore[0]:-self.pad_ignore[1], 0]) / 3.0 + \
            #           slice_wise_LNCC(mri_resampled[:, :, self.pad_ignore[0]:-self.pad_ignore[1]], photo_resampled[:, :, self.pad_ignore[0]:-self.pad_ignore[1], 1]) / 3.0 + \
            #           slice_wise_LNCC(mri_resampled[:, :, self.pad_ignore[0]:-self.pad_ignore[1]], photo_resampled[:, :, self.pad_ignore[0]:-self.pad_ignore[1], 2]) / 3.0

            nccs = torch.zeros(
                3, self.Nslices - self.pad_ignore[0] - self.pad_ignore[1]).to(
                    self.device)
            for z in range(self.Nslices - self.pad_ignore[0] -
                           self.pad_ignore[1]):
                x = mri_resampled[:, :, z + self.pad_ignore[0]]
                mx = torch.mean(x)
                vx = x - mx
                for c in range(3):
                    y = photo_resampled[:, :, z + self.pad_ignore[0], c]
                    my = torch.mean(y)
                    vy = y - my
                    nccs[c, z] = torch.mean(vx * vy) / torch.sqrt(
                        torch.clamp(torch.mean(vx**2), min=1e-5)) / torch.sqrt(
                            torch.clamp(torch.mean(vy**2), min=1e-5))

            ncc_mri = torch.mean(nccs)

        else:
            num = torch.sum(2 * (mri_resampled * mask_resampled))
            den = torch.clamp(torch.sum(mri_resampled + mask_resampled),
                              min=1e-5)
            dice_mri = num / den

        dices_slices = torch.zeros(self.Nslices - 1 - self.pad_ignore[0] -
                                   self.pad_ignore[1]).to(self.device)
        for z in range(self.Nslices - 1 - self.pad_ignore[0] -
                       self.pad_ignore[1]):
            dices_slices[z] = torch.sum(2 * (mask_resampled[:, :, z + self.pad_ignore[0]] * mask_resampled[:, :, z + self.pad_ignore[0] + 1]))\
                / torch.clamp(torch.sum(mask_resampled[:, :, z + self.pad_ignore[0]] + mask_resampled[:, :, z + self.pad_ignore[0] + 1]), min=1e-5)
        dice_slices = torch.mean(dices_slices)

        nccs_slices = torch.zeros(self.Nslices - 1 - self.pad_ignore[0] -
                                  self.pad_ignore[1]).to(self.device)
        for z in range(self.Nslices - 1 - self.pad_ignore[0] -
                       self.pad_ignore[1]):
            x = photo_resampled[:, :, z + self.pad_ignore[0], ...]
            y = photo_resampled[:, :, z + self.pad_ignore[0] + 1, ...]
            m = mask_resampled[:, :, z + self.pad_ignore[
                0]] * mask_resampled[:, :, z + self.pad_ignore[0] + 1]
            if len(x.shape) == 2:  # grayscale
                m3 = m
            else:  # rgb
                m3 = m.repeat([3, 1, 1]).permute([1, 2, 0])
            bottom = torch.clamp(torch.sum(m3), min=1e-5)
            mx = torch.sum(x * m3) / bottom
            my = torch.sum(y * m3) / bottom
            vx = x - mx
            vy = y - my
            nccs_slices[z] = torch.sum(vx * vy * m3) / torch.clamp(torch.sqrt(
                torch.sum(m3 * (vx**2))) * torch.sqrt(torch.sum(m3 * (vy**2))),
                                                                   min=1e-5)
        ncc_slices = torch.mean(nccs_slices)

        aff_regularizers = torch.abs(torch.sum(self.scaling, dim=0) / 20)
        regularizer = torch.mean(aff_regularizers)

        loss_photos = -self.k_dice_slices * dice_slices - self.k_ncc_slices * ncc_slices + self.k_regularizer * regularizer + self.k_nonlinear * cost_field

        if self.ref_type == 'mask' or self.ref_type == 'soft_mask':
            loss = loss_photos - self.k_dice_mri * dice_mri

        elif self.ref_type == 'image':
            loss = loss_photos - self.k_ncc_intermodality * ncc_mri

        else:

            loss = loss_photos - self.k_surface_term * av_grad - self.k_dice_mri * dice_mri

        if torch.isnan(loss):
            kk = 1

        return loss, photo_resampled, photo_aff, mri_aff_combined, Rt, M
