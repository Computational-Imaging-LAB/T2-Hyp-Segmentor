import numpy as np
import os
import torch
#import matplotlib.pyplot as plt
from torchvision.utils import save_image
import nipype.interfaces.fsl as fsl
#import nibabel as nb
import shutil
import glob
import cc3d

class preGlioma():
    """This class is for preprocessing. Mainly wrote for gliomas but it is used in meningiomas too
    """    
    def __init__(self):
        """Initialize the preprocessing class
        """        
        self.counter = np.zeros(3).astype('int')
        self.image_size = np.zeros(3).astype('int')
        self.dims = []
        self.patch_size = [0, 0]
        self.trim=[0,0,0,0]

    def regionprops3d(self,image,third_dim=-1):
        labels_out = cc3d.connected_components(image) 
        stats = cc3d.statistics(labels_out)
        try:
            value=np.argmax(stats['voxel_counts'][1::])+1
        except:
            value=0
        return (labels_out==value).astype('int')

    def normalize(self, img,typx='unit-variance',masked=False):
        """Normalization step of the image

        Args:
            img (np.array): image to be normalized
            typx (str, optional): Type of normalization [min-max, unit-variance]. Defaults to 'unit-variance'.
            masked (bool, optional): Mask is for eliminate the zeroish voxels. Defaults to False.

        Returns:
            [np.array]: Normalized image
        """      
        try:
            img_mask=(img>np.mean(img)*.10)
        except TypeError:
            img=img.numpy()
            img_mask=(img>np.mean(img)*.10)
        if typx == 'unit-variance':
            img=img-np.mean(img)
            img=img/np.std(img)
            if masked:
                return img*img_mask
            else:
                return img
        elif typx == 'min-max':
            min_val = np.min(img)
            max_val = np.max(img)
            if (max_val-min_val) == 0:
                return img*img_mask
            else:
                img = (img-min_val) / (max_val-min_val)
            if masked:
                return img*img_mask
            else:
                return img
    def trim_black_ends(self,img,seg=None):
        """Trim the black ends of the image

        Args:
            img (np.array): Image to be trimmed

        Returns:
            [np.array]: Trimmed image
        """        
        mean_val=np.mean(img)*0.1
        self.trim[2],self.trim[3]=np.min(np.where(img.mean(axis=0)>mean_val)),np.max(np.where(img.mean(axis=0)>mean_val))
        self.trim[0],self.trim[1]=np.min(np.where(img.mean(axis=1)>mean_val)),np.max(np.where(img.mean(axis=1)>mean_val))

        if seg is not None:
            return img[self.trim[0]:self.trim[1],self.trim[2]:self.trim[3]], seg[self.trim[0]:self.trim[1],self.trim[2]:self.trim[3]]
        else:
            return img[self.trim[0]:self.trim[1],self.trim[2]:self.trim[3]]      

    def slice_chopper(self, img, seg=None, slices=5, dim=-1, phase='test'):
        """This function is used to chop the image into slices (Not implemented in s100 project it is needed for the patch-wise projects)

        Args:
            img (torch.tensor): Image to be chopped
            seg (torch.tensor, optional): Segmentation mask . Defaults to None.
            slices (int, optional): Number of the slices. Defaults to 5.
            dim (int, optional): Dimension of the splitting occurs. Defaults to -1.
            phase (str, optional): Defaults to 'test'.

        Returns:
            [list]: List of torch tensors
        """        


        self.slice_chop = slices
        if phase == 'train':
            self.slc1, self.slc2 = self.cut_tumor_image(seg)
            img = img[:, :, self.slc1:self.slc2+1]
        tms = img.shape[dim]//slices
        tms_mod = img.shape[dim] % slices
        self.counter[dim] = np.ceil(tms)
        self.image_size[dim] = img.shape[dim]

        images = []
        if tms == 0:
            images.append(img)
            return images
        for idx in range(0, tms):

            images.append(img[:, :, idx*slices:(idx+1)*slices])

        imageLast = images[-1].copy()
        for idc in range(tms_mod, 0, -1):
            imageLast[:, :, -idc] = img[:, :, (idx+1)*slices+idc-1]

        images.append(imageLast)

        return images

    def betfsl(self, segs, root='/cta/users/abas/Desktop/Meningiom/MeningiomData/preprocessed/'):
        """ Performing fsl-bet using python

        Args:
            segs (torch.tensor): segmentation paths. It is needed for finding the corresponding image
            root (str, optional): Root path of the images. Defaults to '/cta/users/abas/Desktop/Meningiom/MeningiomData/preprocessed/'.

        Returns:
            [str]: Result situation
        """
        

        T2_HYP_segs = glob.glob(segs)
        for t2_hyp in T2_HYP_segs:
            t2_root = ('/').join(t2_hyp.split('/')[:-2])
            print(f'Processing:{t2_root}')
            try:
                t2 = glob.glob(t2_root+'/Anatomic/T2_TSE_TRA*/*.nii')[0]
            except:
                continue

            #image = nib.load(t2).get_fdata().astype('float32')
            #seg = nib.load(t2_hyp).get_fdata().astype('float32')
            # image=self.prg.betfsl(image)
            mybet = fsl.BET()
            name = t2_root.split('/')[-1]
            mybet.inputs.in_file = t2
            mybet.inputs.out_file = root+name+'/T2_BET_TRA.nii'
            if os.path.exists(root+name):
                continue
            os.mkdir(root+name)
            shutil.copyfile(t2_hyp, root+name+'/T2_SEG_TRA.nii')
            print(f'Copy from: {t2_hyp} , To: {root+name}')
            result = mybet.run()
        return result



    def patch_chopper(self, imgs, patch_size=256, dim=0):
        """This function is used to chop the image into patches

        Args:
            imgs (list): input images
            patch_size (int, optional): Defaults to 256.
            dim (int, optional):  Defaults to 0.

        Returns:
            [list]: list chopped images tensors
        """        
        
        images = []
        self.dims.append(dim)
        self.patch_size[dim] = patch_size
        for img in imgs:

            tms = img.shape[dim]/patch_size
            tms_mod = img.shape[dim] % patch_size

            self.counter[dim] = np.ceil(tms)
            tms_mod2 = (patch_size-tms_mod)//2
            if tms_mod != 0:
                if dim == 0:
                    vals = ((tms_mod2, (patch_size-tms_mod)-tms_mod2),
                            (0, 0), (0, 0))
                elif dim == 1:
                    vals = (
                        (0, 0), (tms_mod2, (patch_size-tms_mod)-tms_mod2), (0, 0))

                img = np.pad(img, vals, 'constant', constant_values=(0, 0))
            # print(img.shape)
            self.image_size[dim] = img.shape[dim]

            for idx in range(0, self.counter[dim]):
                if dim == 0:
                    images.append(img[idx*patch_size:(idx+1)*patch_size, :, :])
                elif dim == 1:
                    images.append(img[:, idx*patch_size:(idx+1)*patch_size, :])
        return images

    def save_image(self, images, segs, image, seg, name, root='/cta/users/abas/Desktop/Meningiom/MeningiomData/preprocessed/'):
        """Saving the tensors of images

        Args:
            images (list): Chopped images
            segs (list): Chopped segmentations
            image (np.array): Original image
            seg (np.array): Original segmentation
            name (str): Name of the image
            root (str, optional): Save path. Defaults to '/cta/users/abas/Desktop/Meningiom/MeningiomData/preprocessed/'.
        """        
        if not os.path.exists(root+name):
            os.mkdir(root+name)

        for idx, (img, segs) in enumerate(zip(images, segs)):
            if np.max(segs) < 0.01:
                continue
            save_image(torch.tensor(img).permute(2, 0, 1),
                       root+name+'/img_patch_'+str(idx)+'.png')
            save_image(torch.tensor(segs).permute(2, 0, 1),
                       root+name+'/seg_patch_'+str(idx)+'.png')

        torch.save(torch.tensor(image), root+name+'/orig.pt')
        torch.save(torch.tensor(seg), root+name+'/seg.pt')

        return

    def reverse_pad(self, image, org_size):
        """Reverse padding of initialized class

        Args:
            image (torch.tensor): Reconstructed image with padding
            org_size (org_size): Original size of the image

        Returns:
            torch.tensor: Reverse padded image
        """        
        shapes = image.shape

        difs = np.array(shapes)-np.array(org_size)

        image = image[difs[0]//2:shapes[0]-difs[0]//2, difs[1] //
                      2:shapes[1]-difs[1]//2, difs[2]//2:shapes[2]-difs[2]//2]

        return image

    def reconstruct(self, images, org_size,labels=None):
        """Reconstructing the image from the patches

        Args:
            images (list): List of patches
            org_size (shape): Not used in this version. Deprecated
            labels (list ,optional): Defaults to None. If it is not none it will reconstruct image with the labels gathered from the patches using classifier network

        Returns:
            torch.tensor: Reconstructed image
        """        

        #print(self.counter)
        image = np.zeros(self.image_size)
        labels_gen=np.zeros(self.image_size)
        for idx in range(0, self.counter[2]):
            for idc in range(0, self.counter[1]):
                for idx2 in range(0, self.counter[0]):
                    self.indices = (
                        (self.counter[0]*self.counter[1]*idx)+(self.counter[0]*idc)+idx2)
                    print(self.indices)
                    self.idc = idc
                    self.idx2 = idx2
                    image[idx2*self.patch_size[0]:(idx2+1)*self.patch_size[0], self.patch_size[1]*idc:(
                        idc+1)*self.patch_size[1], idx*self.slice_chop:(idx+1)*self.slice_chop] = images[self.indices]
                    if labels is not None:
                        labels_gen[idx2*self.patch_size[0]:(idx2+1)*self.patch_size[0], self.patch_size[1]*idc:(
                        idc+1)*self.patch_size[1], idx*self.slice_chop:(idx+1)*self.slice_chop] = labels[self.indices]    
                    # images[idx][idc][idx2]=self.reverse_pad(images[idx][idc][idx2],org_size)
        if labels is not None:
            return image, labels_gen
        else:
            return image

    def cut_tumor_image(self, seg,nonzero=False):
        """Cutting the tumor image from the segmentation

        Args:
            seg (segmentatiom): [description]

        Returns:
            [type]: [description]
        """        
        if nonzero:
            nonzers = torch.nonzero(torch.tensor(seg))[:, 2]
            return torch.min(nonzers), torch.max(nonzers)
        else:
            nonzers=np.where(seg==1)[2]
            return nonzers.min(), nonzers.max()
        
