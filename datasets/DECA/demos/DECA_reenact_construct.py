# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

################### Kuoyuan Comment ###################
# This file is used to run DECA to build meshes to reenact the given video
# first extract all parameters of Kuoyuan
# then extract the pose and expression parameter of the images to be reenacted
# substitute the pose and expression parameters of Kuoyuan with the parameters of the images to be reenacted
# then render the mesh and save pose and expression parameters
################### Kuoyuan Comment ###################


import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points

def main(args):
    # if args.rasterizer_type != 'standard':
    #     args.render_orig = False
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # load test images 
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector, sample_step=args.sample_step)
    # load reenactment images
    reenact_data = datasets.TestData(args.reenactpath, iscrop=args.iscrop, face_detector=args.detector)
    
    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config = deca_cfg, device=device)
    
    # identity (testdata) renference ("Kuoyuan")
    test_i = 0
    name = testdata[test_i]['imagename']
    images = testdata[test_i]['image'].to(device)[None,...]
    with torch.no_grad():
        id_codedict = deca.encode(images)
        
        id_opdict, id_visdict = deca.decode(id_codedict)
        id_visdict = {x:id_visdict[x] for x in ['inputs', 'shape_detail_images']}   
        if args.render_orig: # true
            tform = testdata[test_i]['tform'][None, ...] # 2D similarity transform
            tform = torch.inverse(tform).transpose(1,2).to(device)
            original_image = testdata[test_i]['original_image'][None, ...].to(device)
            orig_id_opdict, orig_id_visdict = deca.decode(id_codedict, render_orig=True, original_image=original_image, tform=tform)    
            orig_id_visdict['inputs'] = original_image
            orig_id_visdict = {x:orig_id_visdict[x] for x in ['inputs', 'shape_detail_images']}          

    # expression and pose transfer
    exp_pose_dict = {}
    for i in tqdm(range(len(reenact_data))):
        name = reenact_data[i]['imagename']
        images = reenact_data[i]['image'].to(device)[None,...]
        with torch.no_grad():
            codedict = deca.encode(images)
            
            exp_pose_dict[name] = {'exp': codedict['exp'].cpu().numpy().tolist(), 'pose': codedict['pose'].cpu().numpy().tolist()}
            
            codedict_with_identity = id_codedict.copy()
            codedict_with_identity['exp'] = codedict['exp']
            codedict_with_identity['pose'] = codedict['pose']
            
            transfer_opdict, transfer_visdict = deca.decode(codedict_with_identity) # decode cropped image
            
            id_visdict['transferred_shape'] = transfer_visdict['shape_detail_images']
            transfer_opdict['uv_texture_gt'] = id_opdict['uv_texture_gt']
            
            if args.render_orig: # true
                tform = reenact_data[i]['tform'][None, ...] # 2D similarity transform
                tform = torch.inverse(tform).transpose(1,2).to(device)
                original_image = reenact_data[i]['original_image'][None, ...].to(device)
                orig_transfer_opdict, orig_transfer_visdict= deca.decode(codedict_with_identity, render_orig=True, original_image=original_image, tform=tform)    
                orig_id_visdict['transferred_shape'] = orig_transfer_visdict['shape_detail_images']
                orig_transfer_opdict['uv_texture_gt'] = orig_id_opdict['uv_texture_gt']
                orig_transfer_visdict['inputs'] = original_image            

        visdict = transfer_visdict; opdict = transfer_opdict
        if args.render_orig:
            orig_visdict = orig_transfer_visdict; orig_opdict = orig_transfer_opdict

        if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages or args.saveNeededOnly:
            os.makedirs(os.path.join(savefolder, name), exist_ok=True)
        # -- save results
        if args.saveNeededOnly:
            deca.save_obj(os.path.join(savefolder, name, name + '.obj'), opdict)
        if args.saveDepth: # no need
            depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1,3,1,1)
            visdict['depth_images'] = depth_image
            cv2.imwrite(os.path.join(savefolder, name, name + '_depth.jpg'), util.tensor2image(depth_image[0]))
        if args.saveKpt: # save kpt of the original image
            np.savetxt(os.path.join(savefolder, name, name + '_kpt2d.txt'), orig_opdict['landmarks2d'][0].cpu().numpy())
            np.savetxt(os.path.join(savefolder, name, name + '_kpt3d.txt'), orig_opdict['landmarks3d'][0].cpu().numpy())
        if args.saveObj: # save object and detail object
            deca.save_obj(os.path.join(savefolder, name, name + '.obj'), opdict)
        if args.saveMat: # save cropped image opdict
            opdict = util.dict_tensor2npy(opdict)
            savemat(os.path.join(savefolder, name, name + '.mat'), opdict)
        if args.saveVis: # save both cropped and original image visulization
            cv2.imwrite(os.path.join(savefolder, name + '_vis.jpg'), deca.visualize(visdict))
            if args.render_orig:
                cv2.imwrite(os.path.join(savefolder, name, name + '_vis_original_size.jpg'), deca.visualize(orig_visdict))
        if args.saveOrigVis:
            if args.render_orig:
                cv2.imwrite(os.path.join(savefolder, name, name + '_vis_original_size.jpg'), deca.visualize(orig_visdict))
        if args.saveImages:
            for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images', 'landmarks2d']:
                if vis_name not in visdict.keys():
                    continue
                image = util.tensor2image(visdict[vis_name][0])
                cv2.imwrite(os.path.join(savefolder, name, name + '_' + vis_name +'.jpg'), util.tensor2image(visdict[vis_name][0]))
                if args.render_orig:
                    image = util.tensor2image(orig_visdict[vis_name][0])
                    cv2.imwrite(os.path.join(savefolder, name, 'orig_' + name + '_' + vis_name +'.jpg'), util.tensor2image(orig_visdict[vis_name][0]))
        if args.saveOrigImages:
            for vis_name in ['inputs', 'rendered_images', 'shape_images', 'shape_detail_images', 'landmarks2d']:
                if vis_name not in orig_visdict.keys():
                    continue
                if args.render_orig:
                    image = util.tensor2image(orig_visdict[vis_name][0])
                    cv2.imwrite(os.path.join(savefolder, name, 'orig_' + name + '_' + vis_name +'.jpg'), util.tensor2image(orig_visdict[vis_name][0]))
    # Store the dictionary in a json file, these parameters are used in the NeRF MLP (second MLP)
    with open(os.path.join(savefolder, 'exp_pose_dict_test.json'), 'w') as f:
        json.dump(exp_pose_dict, f)
    print(f'-- please check the results in {savefolder}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-r', '--reenactpath', default='TestSamples/examples', type=str,
                        help='path to the reenact data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='TestSamples/examples/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--sample_step', default=10, type=int,
                        help='sample images from video data for every step' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='pytorch3d', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    parser.add_argument('--render_orig', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to render results in original image size, currently only works when rasterizer_type=standard')
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode' )
    parser.add_argument('--saveVis', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints' )
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image' )
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow' )
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat' )
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    parser.add_argument('--saveNeededOnly',default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save only the needed files for RigNeRF')
    parser.add_argument('--saveOrigVis', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output in original image size')
    parser.add_argument('--saveOrigImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images in original image size')
    main(parser.parse_args())