# T2-Hyp-Segmentor
*This repo is for segmentation of T2 hyp regions in gliomas.*


[![Maintaner](https://img.shields.io/badge/maintainer-CIL-blue)](https://cil.boun.edu.tr)
[![Website monip.org](https://img.shields.io/website-up-down-green-red/http/monip.org.svg)](https://www.python.org/)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-sphinx-doc](https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg)](https://www.sphinx-doc.org/)
[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](https://www.python.org/)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Computational-Imaging-LAB/https://computational-imaging-lab.github.io/Identification-of-S100-using-T2-w-images-deep-learning-/blob/master/LICENSE)
![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://github.com/abdullahbas)


By downloading the model from [here](https://drive.google.com/file/d/1vfF-IE7fyxN1_Ld98NDFBCP6yt_p_Web/view?usp=sharing) you can use it to segment your T2w images of gliomas. Our model is independent from the input image size. 

We reached **89.88%** of dice score on the traning set and **82.9%** dice score on the validation set. The results we show in the following are from the test set. 

Red channel represents the dilated output. We used (5,5,2) kernel to dilate the image. Hence, it is possible to see more slices segmented as t2 hyper regions. 
Green channel is the output of the model after post-processing. 
Blue channel is the ground-truth segmentation.

Briefly we can conclude that white regions on right hand side in the following GIFs are the places that the model hit the ground truth. We added 1 very good, 1 average and 1 bad examples.

# Very good one!
![.](https://github.com/Computational-Imaging-LAB/T2-Hyp-Segmentor/blob/main/images/mG0012.gif)
# Average
![.](https://github.com/Computational-Imaging-LAB/T2-Hyp-Segmentor/blob/main/images/G0027.nii.gif)
# Bad one
![.](https://github.com/Computational-Imaging-LAB/T2-Hyp-Segmentor/blob/main/images/mG0023.gif)


For inference;

`git clone https://github.com/Computational-Imaging-LAB/T2-Hyp-Segmentor/`

`cd T2-Hyp-Segmentor`

`pip install -r requirements.txt`

If you want to test the output then use,

`python inference.py --model <model_path> --input <input_nii_path> --output <output_path> --seg <seg_nii_path>`

If you want to use the model to segment

`python inference.py --model <model_path> --input <input_nii_path> --output <output_path> `



