#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import time
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    fps = []
    rendering_times = []
    nums_renderered = []
    psnrs = []

    means = []
    maxs = []

    print("warming up")
    for _ in tqdm(range(5)):
        rendering = render(views[0], gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        gt = views[0].original_image.cuda()
        l1 = l1_loss(rendering, gt)
        ssim_value = ssim(rendering, gt)
        loss = l1 + ssim_value
        loss.backward()

    views = views[3:4]
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()
        t0 = time.time()
        rendering_pack = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = rendering_pack["render"]
        gt = view.original_image.cuda()
        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]
        l1 = l1_loss(rendering, gt)
        ssim_value = ssim(rendering, gt)
        loss = l1 + ssim_value
        loss.backward()
        torch.cuda.synchronize()
        t = time.time() - t0
        rendering_times.append(t)
        psnrs.append(psnr(rendering, gt).mean())
        fps.append(1.0 / t)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        print(os.path.join(render_path, '{0:05d}'.format(idx) + ".png"));
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


    assert(isinstance(gaussians, GaussianModel))
    grads = gaussians.export_grad()
    # # torch.save(grads, 'grads_ref.pth')
    ref_grads = torch.load('grads_ref.pth', weights_only=True)
    # for key, value in grads.items():
    #     if key == "features_rest":
    #         continue
    #     print(key)
    #     for value_item in value:
    #         print(value_item)
            # print(key)
        # if True:
    #         print(torch.mean(torch.abs(grads[key] - value) / torch.max(abs(grads[key]), abs(value))))
    #         print(torch.max(torch.abs(grads[key] - value) / torch.max(abs(grads[key]), abs(value))))
    #         print(torch.mean(grads[key]), torch.mean(value))
    #         print(torch.max(grads[key]), torch.max(value))
    
    for grad, ref_grad in zip(grads["xyz"][:10000], ref_grads["xyz"][:10000]):
        print(grad, ref_grad)
    
    print("time = ", torch.tensor(rendering_times).mean())
    print("fps = ", torch.tensor(fps).mean())
    #print("num_rendered=", torch.tensor(nums_renderered).float().mean())
    print("psnrs=", torch.tensor(psnrs).mean())

    print("means=", means)
    print("maxs=", maxs)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    # with torch.no_grad():
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

    if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    args.depths = ""
    args.train_test_exp = False
    args.data_device = 'cuda'
    args.images = "images_4"

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, False)