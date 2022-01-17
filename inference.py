import argparse
import nibabel as nib
import numpy as np
import src.subplotter as subplotter
import torch.nn as nn
import torch
import src.preprocess as preprocess
import models.model as model
import imageio
import scipy.ndimage as ndimage
import cv2
from skimage.morphology import dilation
import time
from matplotlib import pyplot as plt
from efficientnet_pytorch import EfficientNet
def load_model(model_path,model):
    model.encoder1.enc1conv1.in_channels = 1
    model.encoder1.enc1conv1.weight = torch.nn.Parameter(
    torch.mean(model.encoder1.enc1conv1.weight, dim=1).unsqueeze(1))

    model.convClassifier = nn.Conv2d(
        in_channels=256, out_channels=128, kernel_size=3, stride=1)
    model.convClassifier2 = nn.Conv2d(
        in_channels=128, out_channels=64, kernel_size=5, stride=1)
    model.convClassifier3 = nn.Conv2d(
        in_channels=64, out_channels=32, kernel_size=3, stride=1)    
    model.classifier = nn.Linear(in_features=2048, out_features=1024)
    model.classifier2 = nn.Linear(in_features=1024, out_features=256)
    model.classifier3 = nn.Linear(in_features=256, out_features=1)
    model.load_state_dict(torch.load(model_path))

    return model

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--model', type=str, default='/cta/users/abas/Desktop/segmentation/t2_hyp_segment/checkpoints/INF_model_256_0.191_best.pt', help='model path')
    parser.add_argument('--input', type=str,default='/cta/users/abas/Desktop/segmentation/t2_hyp_segment/data/test/G0015/T0015/Anatomic/T2_TSE_TRA_448_5MM_YENI_0003/T2_TSE_TRA_448_5MM_YENI_0003.nii',help='input image path')
    parser.add_argument('--seg',type=str,default='/cta/users/abas/Desktop/segmentation/t2_hyp_segment/data/test/G0015/T0015/Segmentations/T0015_T2_HYP.nii',help='it is used for testing')
    parser.add_argument('--output', type=str, default='/cta/users/abas/Desktop/segmentation/t2_hyp_segment/images/output.nii', help='output image path')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--bet',type=bool,default=False,help='bet')
    args=parser.parse_args()
    start=time.time()
    prg=preprocess.preGlioma()
    model_u=model.UNet()
    model_u=load_model(args.model,model_u)
    model_u=model_u.eval()
   
    if args.gpu>=0:
        try:
            device=f'cuda:{args.gpu}'
            
            model_u.to(device)
        except:
            print('GPU not available')
            exit()
    print(device)
    img_nii=nib.load(args.input)
    img=img_nii.get_data().astype('float32')
    if args.seg!='None':
        seg=nib.load(args.seg).get_data().astype('float32')
    else:
        seg=np.zeros(img.shape)

    if args.bet:
        img=prg.bet(img)
    
    img=prg.normalize(img,typx='min-max')
    images=prg.slice_chopper(img,phase='test',slices=5)
    images=prg.patch_chopper(images, dim=1, patch_size=512)
    images=prg.patch_chopper(images, dim=0, patch_size=512)

    segs=prg.slice_chopper(seg,phase='test',slices=5)
    segs=prg.patch_chopper(segs, dim=1, patch_size=512)
    segs=prg.patch_chopper(segs, dim=0, patch_size=512)
    seg_rec=prg.reconstruct(segs,512)
    outputs=[]
    labels=np.array(0)
    with torch.no_grad():
        for imgx in images:
            imgx=torch.tensor(imgx).unsqueeze(0).to(device)
            #model_efficient(torch.tensor(imgx).squeeze(-1))
            output=model_u(torch.tensor(imgx).permute(3,0,1,2))

            #output=output.permute(0,2,3,1).squeeze(0)
            output=output.squeeze(1).permute(1,2,0)
            outputs.append(output.cpu().detach().numpy())
    out=prg.reconstruct(outputs,img.shape,labels=None)  
    out=prg.reverse_pad(out,img.shape)
    print('*'*100)
    masked_img=prg.regionprops3d((out)>0.5,third_dim=-1)
    masked_img2=dilation(masked_img,np.ones((5,5,2)))
    sbtplotter=subplotter.plotter([img,out,masked_img,seg,'min-max'])
    image_lst=[]
    for i in range(img.shape[-1]):
        con_image=np.dstack((img[:,:,i],masked_img[:,:,i],seg[:,:,i]))
        seg_image=np.dstack((masked_img[:,:,i],masked_img[:,:,i],seg[:,:,i]))
        #print(seg_image.shape)
        #cv2.imshow('image',img[:,:,i])
        #cv2.imshow('segs',seg_image)
        im_to_show=np.hstack((np.dstack((img[:,:,i],img[:,:,i],img[:,:,i])),seg_image))
        cv2.imshow('frame',im_to_show)
        image_lst.append(im_to_show)
        cv2.waitKey(200)

        
        
    output_path=args.output
    print(f'Saving to: {output_path}')
    nib.save( nib.Nifti1Image(prg.normalize(img,'unit-variance'),affine=img_nii.affine),args.output.replace('nii','3d_fs.nii'))
    #nib.save( nib.Nifti1Image(prg.normalize(label_im,'unit-variance'),affine=img_nii.affine),args.output.replace('nii','label.nii'))

    nib.save( nib.Nifti1Image(out,affine=img_nii.affine),args.output)
    est=time.time()-start
    print(f'Save Completed. Total time: {est}')
    imageio.mimsave(args.output+'.gif', image_lst, fps=4)