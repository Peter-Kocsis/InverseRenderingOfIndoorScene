import torch
import numpy as np
from torch.autograd import Variable
import argparse
import random
import os

from torch.utils.data import DataLoader

import models
import utils
import glob
import os.path as osp
import cv2
import BilateralLayer as bs
import torch.nn.functional as F
import scipy.io as io
import utils
from irois import dataLoader

parser = argparse.ArgumentParser()
# The locationi of testing set
parser.add_argument('--dataRoot', help='path to real images')
parser.add_argument('--imList', help='path to image list')

parser.add_argument('--experiment0', default=None, help='the path to the model of first cascade' )
parser.add_argument('--experimentLight0', default=None, help='the path to the model of first cascade' )
parser.add_argument('--experimentBS0', default=None, help='the path to the model of bilateral solver')
parser.add_argument('--experiment1', default=None, help='the path to the model of second cascade' )
parser.add_argument('--experimentLight1', default=None, help='the path to the model of second cascade')
parser.add_argument('--experimentBS1', default=None, help='the path to the model of second bilateral solver')

parser.add_argument('--testRoot', help='the path to save the testing errors' )

# The basic testing setting
parser.add_argument('--nepoch0', type=int, default=14, help='the number of epoch for testing')
parser.add_argument('--nepochLight0', type=int, default=10, help='the number of epoch for testing')
parser.add_argument('--nepochBS0', type=int, default=15, help='the number of epoch for bilateral solver')
parser.add_argument('--niterBS0', type=int, default=1000, help='the number of iterations for testing')

parser.add_argument('--nepoch1', type=int, default=7, help='the number of epoch for testing')
parser.add_argument('--nepochLight1', type=int, default=10, help='the number of epoch for testing')
parser.add_argument('--nepochBS1', type=int, default=8, help='the number of epoch for bilateral solver')
parser.add_argument('--niterBS1', type=int, default=4500, help='the number of iterations for testing')

parser.add_argument('--imHeight', type=int, default=240, help='the height / width of the input image to network' )
parser.add_argument('--imWidth', type=int, default=320, help='the height / width of the input image to network' )
parser.add_argument('--imHeight0', type=int, default=240, help='the height / width of the input image to network' )
parser.add_argument('--imWidth0', type=int, default=320, help='the height / width of the input image to network' )
parser.add_argument('--imHeight1', type=int, default=240, help='the height / width of the input image to network' )
parser.add_argument('--imWidth1', type=int, default=320, help='the height / width of the input image to network' )

parser.add_argument('--envRow', type=int, default=120, help='the height /width of the envmap predictions')
parser.add_argument('--envCol', type=int, default=160, help='the height /width of the envmap predictions')
parser.add_argument('--envHeight', type=int, default=32, help='the height /width of the envmap predictions')
parser.add_argument('--envWidth', type=int, default=64, help='the height /width of the envmap predictions')

parser.add_argument('--SGNum', type=int, default=12, help='the number of spherical Gaussian lobes')
parser.add_argument('--offset', type=float, default=1, help='the offset when train the lighting network')

parser.add_argument('--cuda', action = 'store_true', help='enables cuda')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for testing network')

parser.add_argument('--level', type=int, default=2, help='the cascade level')
parser.add_argument('--isLight', action='store_true', help='whether to predict lightig')
parser.add_argument('--isBS', action='store_true', help='whether to use bilateral solver')

# Image Picking
opt = parser.parse_args()
print(opt)

opt.gpuId = opt.deviceIds[0]

if opt.experiment0 is None:
    opt.experiment0 = 'check_cascade0_w%d_h%d' % (opt.imWidth0, opt.imHeight0 )

if opt.experiment1 is None:
    opt.experiment1 = 'check_cascade1_w%d_h%d' % (opt.imWidth1, opt.imHeight1 )

if opt.experimentLight0 is None:
    opt.experimentLight0 = 'check_cascadeLight0_sg%d_offset%.1f' % \
            (opt.SGNum, opt.offset )

if opt.experimentLight1 is None:
    opt.experimentLight1 = 'check_cascadeLight1_sg%d_offset%.1f' % \
            (opt.SGNum, opt.offset )

if opt.experimentBS0 is None:
    opt.experimentBS0 = 'checkBs_cascade0_w%d_h%d' % (opt.imWidth0, opt.imHeight0 )

if opt.experimentBS1 is None:
    opt.experimentBS1 = 'checkBs_cascade1_w%d_h%d' % (opt.imWidth1, opt.imHeight1 )

experiments = [opt.experiment0, opt.experiment1 ]
experimentsLight = [opt.experimentLight0, opt.experimentLight1 ]
experimentsBS = [opt.experimentBS0, opt.experimentBS1 ]
nepochs = [opt.nepoch0, opt.nepoch1 ]
nepochsLight = [opt.nepochLight0, opt.nepochLight1 ]
nepochsBS = [opt.nepochBS0, opt.nepochBS1 ]
nitersBS = [opt.niterBS0, opt.niterBS1 ]

imHeights = [opt.imHeight0, opt.imHeight1 ]
imWidths = [opt.imWidth0, opt.imWidth1 ]

os.system('mkdir {0}'.format(opt.testRoot ) )
os.system('cp *.py %s' % opt.testRoot )

opt.seed = 0
print("Random Seed: ", opt.seed )
random.seed(opt.seed )
torch.manual_seed(opt.seed )

opt.batchSize = 1
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


imBatchSmall = Variable(torch.FloatTensor(opt.batchSize, 3, opt.envRow, opt.envCol ) )

####################################
outfilename = opt.testRoot + '/results'
for n in range(0, opt.level ):
    outfilename = outfilename + '_brdf%d' % nepochs[n]
    if opt.isLight:
        outfilename += '_light%d' % nepochsLight[n]
os.system('mkdir -p {0}'.format(outfilename ) )

####################################
brdfDataset = dataLoader.BatchLoader( opt.dataRoot, isAllLight = True,
        imWidth = opt.imWidth, imHeight = opt.imHeight, isLight = True,
        cascadeLevel = 0, SGNum = opt.SGNum,
                                      # envHeight = opt.envHeight, envWidth = opt.envWidth,
                                      )
brdfLoader = DataLoader(brdfDataset, batch_size = 1, num_workers =
        0, shuffle = True )

j = 0
for i, dataBatch in enumerate(brdfLoader):
    j += 1
    imId = dataBatch['name'][0]

    imOutputNames = []
    imId = imId.split('/')[-1]
    print(imId)
    imOutputNames.append(osp.join(outfilename, imId))

    imBatches = []

    albedoNames, albedoImNames = [], []
    normalNames, normalImNames = [], []
    roughNames, roughImNames = [], []
    depthNames, depthImNames = [], []
    envmapPredNames, envmapPredImNames = [], []
    renderedNames, renderedImNames = [], []
    cLightNames = []
    shadingNames, envmapsPredSGNames = [], []

    n = 0
    albedoNames.append(osp.join(outfilename, imId.replace('.hdr', '_albedo%d.npy' % n) ) )
    albedoImNames.append(osp.join(outfilename, imId.replace('.hdr', '_albedo%d.png' % n ) ) )
    normalNames.append(osp.join(outfilename, imId.replace('.hdr', '_normal%d.npy' % n ) ) )
    normalImNames.append(osp.join(outfilename, imId.replace('.hdr', '_normal%d.png' % n) ) )
    roughNames.append(osp.join(outfilename, imId.replace('.hdr', '_rough%d.npy' % n) ) )
    roughImNames.append(osp.join(outfilename, imId.replace('.hdr', '_rough%d.png' % n) ) )
    depthNames.append(osp.join(outfilename, imId.replace('.hdr', '_depth%d.npy' % n) ) )
    depthImNames.append(osp.join(outfilename, imId.replace('.hdr', '_depth%d.png' % n) ) )

    albedoBSNames = albedoNames[n].replace('albedo', 'albedoBs')
    albedoImBSNames = albedoImNames[n].replace('albedo', 'albedoBs')
    roughBSNames = roughNames[n].replace('rough', 'roughBs')
    roughImBSNames = roughImNames[n].replace('rough', 'roughBs')
    depthBSNames = depthNames[n].replace('depth', 'depthBs')
    depthImBSNames = depthImNames[n].replace('depth', 'depthBs')

    envmapsPredSGNames.append(osp.join(outfilename, imId.replace('.hdr', '_envmapSG%d.npy' % n) ) )
    shadingNames.append(osp.join(outfilename, imId.replace('.hdr', '_shading%d.png' % n) ) )
    envmapPredNames.append(osp.join(outfilename, imId.replace('.hdr', '_envmap%d.npz' % n) ) )
    envmapPredImNames.append(osp.join(outfilename, imId.replace('.hdr', '_envmap%d.png' % n) ) )
    renderedNames.append(osp.join(outfilename, imId.replace('.hdr', '_rendered%d.npy' % n) ) )
    renderedImNames.append(osp.join(outfilename, imId.replace('.hdr', '_rendered%d.png' % n) ) )

    cLightNames.append(osp.join(outfilename, imId.replace('.hdr', '_cLight%d.mat' % n) ) )

    # Load the
    im_cpu = dataBatch['im'][0].cpu().permute(1, 2, 0).numpy()
    nh, nw = im_cpu.shape[0], im_cpu.shape[1]

    # Resize Input Images
    newImWidth = []
    newImHeight = []
    for n in range(0, 1):
        if nh < nw:
            newW = imWidths[n]
            newH = int(float(imWidths[n] ) / float(nw) * nh )
        else:
            newH = imHeights[n]
            newW = int(float(imHeights[n] ) / float(nh) * nw )

        if nh < newH:
            im = cv2.resize(im_cpu, (newW, newH), interpolation = cv2.INTER_AREA )
        else:
            im = cv2.resize(im_cpu, (newW, newH), interpolation = cv2.INTER_LINEAR )

        newImWidth.append(newW )
        newImHeight.append(newH )

        im = (np.transpose(im, [2, 0, 1] ).astype(np.float32 ) / 255.0 )[np.newaxis, :, :, :]
        im = im / im.max()
        imBatches.append( Variable(torch.from_numpy(im**(2.2) ) ).cuda() )

    nh, nw = newImHeight[-1], newImWidth[-1]

    newEnvWidth, newEnvHeight, fov = 0, 0, 0
    if nh < nw:
        fov = 57
        newW = opt.envCol
        newH = int(float(opt.envCol ) / float(nw) * nh )
    else:
        fov = 42.75
        newH = opt.envRow
        newW = int(float(opt.envRow ) / float(nh) * nw )

    if nh < newH:
        im = cv2.resize(im_cpu, (newW, newH), interpolation = cv2.INTER_AREA )
    else:
        im = cv2.resize(im_cpu, (newW, newH), interpolation = cv2.INTER_LINEAR )

    newEnvWidth = newW
    newEnvHeight = newH

    im = (np.transpose(im, [2, 0, 1] ).astype(np.float32 ) / 255.0 )[np.newaxis, :, :, :]
    im = im / im.max()
    imBatchSmall = Variable(torch.from_numpy(im**(2.2) ) ).cuda()
    renderLayer = models.renderingLayer(isCuda = opt.cuda,
            imWidth=newEnvWidth, imHeight=newEnvHeight, fov = fov,
            envWidth = opt.envWidth, envHeight = opt.envHeight)

    output2env = models.output2env(isCuda = opt.cuda,
            envWidth = opt.envWidth, envHeight = opt.envHeight, SGNum = opt.SGNum )

    ########################################################
    # Build the cascade network architecture #
    albedoPreds, normalPreds, roughPreds, depthPreds = [], [], [], []
    albedoBSPreds, roughBSPreds, depthBSPreds = [], [], []
    envmapsPreds, envmapsPredImages, renderedPreds = [], [], []
    cAlbedos = []
    cLights = []

    ################# BRDF Prediction ######################
    # inputBatch = imBatches[0]
    # x1, x2, x3, x4, x5, x6 = encoders[0](inputBatch )

    # albedoPred = 0.5 * (albedoDecoders[0](imBatches[0], x1, x2, x3, x4, x5, x6) + 1)
    # normalPred = normalDecoders[0](imBatches[0], x1, x2, x3, x4, x5, x6)
    # roughPred = roughDecoders[0](imBatches[0], x1, x2, x3, x4, x5, x6 )
    # depthPred = 0.5 * (depthDecoders[0](imBatches[0], x1, x2, x3, x4, x5, x6) + 1)
    albedoPred = dataBatch['albedo'].cuda()
    normalPred = dataBatch['normal'].cuda()
    roughPred = dataBatch['rough'].cuda()
    depthPred = dataBatch['depth'].cuda()

    # # Normalize Albedo and depth
    # bn, ch, nrow, ncol = albedoPred.size()
    # albedoPred = albedoPred.view(bn, -1)
    # albedoPred = albedoPred / torch.clamp(torch.mean(albedoPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
    # albedoPred = albedoPred.view(bn, ch, nrow, ncol)
    #
    # bn, ch, nrow, ncol = depthPred.size()
    # depthPred = depthPred.view(bn, -1)
    # depthPred = depthPred / torch.clamp(torch.mean(depthPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
    # depthPred = depthPred.view(bn, ch, nrow, ncol)

    albedoPreds.append(albedoPred )
    normalPreds.append(normalPred )
    roughPreds.append(roughPred )
    depthPreds.append(depthPred )

    ################# Lighting Prediction ###################
    # Interpolation
    imBatchLarge = F.interpolate(imBatches[0], [imBatchSmall.size(2) *
        4, imBatchSmall.size(3) * 4], mode='bilinear')
    albedoPredLarge = F.interpolate(albedoPreds[0], [imBatchSmall.size(2)*
        4, imBatchSmall.size(3) * 4], mode='bilinear')
    normalPredLarge = F.interpolate(normalPreds[0], [imBatchSmall.size(2) *
        4, imBatchSmall.size(3) * 4], mode='bilinear')
    roughPredLarge = F.interpolate(roughPreds[0], [imBatchSmall.size(2) *
        4, imBatchSmall.size(3) * 4], mode='bilinear')
    depthPredLarge = F.interpolate(depthPreds[0], [imBatchSmall.size(2) *
        4, imBatchSmall.size(3) * 4], mode='bilinear')

    inputBatch = torch.cat([imBatchLarge, albedoPredLarge,
        0.5*(normalPredLarge+1), 0.5*(roughPredLarge+1), depthPredLarge ], dim=1 )

    # x1, x2, x3, x4, x5, x6 = lightEncoders[0](inputBatch )

    # Prediction
    # axisPred = axisDecoders[0](x1, x2, x3, x4, x5, x6, imBatchSmall )
    # lambPred = lambDecoders[0](x1, x2, x3, x4, x5, x6, imBatchSmall )
    # weightPred = weightDecoders[0](x1, x2, x3, x4, x5, x6, imBatchSmall )
    lighting = dataBatch['sgenv'].cuda()
    SGNum = opt.SGNum
    def axis_2_euclidean(axis):
        bn, _, envRow, envCol = axis.size()
        axisOrig = axis.view(bn, SGNum, 2, envRow, envCol)
        # thetaOrig, phiOrig = axisOrig.split(1, dim=2)
        theta, phi = axisOrig.split(1, dim=2)
        # theta = thetaOrig * torch.pi / 2.0
        # phi = (2 * phiOrig - 1) * torch.pi
        '''
        theta = (2 * thetaOrig-1) * self.thetaRange + self.thetaCenter.expand_as(thetaOrig )
        phi =  (2 * phiOrig -1) * self.phiRange + self.phiCenter.expand_as(phiOrig )
        '''
        axis_x = torch.sin(theta) * torch.cos(phi)
        axis_y = torch.sin(theta) * torch.sin(phi)
        axis_z = torch.cos(theta)
        axis = torch.cat([axis_x, axis_y, axis_z], dim=2)
        axis = axis
        return axis

    bn, envRow, envCol, _ = lighting.size()
    axisPred, lambPred, weightPred = torch.split(
        lighting.view(bn, envRow, envCol, SGNum, 6), [2, 1, 3], dim=-1)
    axisPred = axisPred.reshape(1, envRow, envCol, 2 * SGNum).permute(0, 3, 1, 2)
    lambPred = lambPred.reshape(1, envRow, envCol, SGNum).permute(0, 3, 1, 2)
    weightPred = weightPred.reshape(1, envRow, envCol, 3 * SGNum).permute(0, 3, 1, 2)

    axisPred = axis_2_euclidean(axisPred)

    bn, SGNum, _, envRow, envCol = axisPred.size()
    envmapsPred = torch.cat([axisPred.view(bn, SGNum*3, envRow, envCol ), lambPred, weightPred], dim=1)
    envmapsPreds.append(envmapsPred )

    # Use SG
    envmapsPredImage, axisPred, lambPred, weightPred = output2env.output2env(axisPred, lambPred, weightPred, useGT=True )
    envmapsPredImages.append(envmapsPredImage )
    # # Use EnvMap
    # envmapsPredImage = dataBatch['envmaps'].cuda()
    # envmapsPredImages.append(envmapsPredImage)

    diffusePred, specularPred = renderLayer.forwardEnv(albedoPreds[0], normalPreds[0],
            roughPreds[0], envmapsPredImages[0] )

    diffusePredNew, specularPredNew = models.LSregressDiffSpec(
            diffusePred,
            specularPred,
            imBatchSmall,
            diffusePred, specularPred )
    # diffusePredNew, specularPredNew = diffusePred, specularPred
    renderedPred = diffusePredNew + specularPredNew
    renderedPreds.append(renderedPred )

    cDiff, cSpec = (torch.sum(diffusePredNew) / torch.sum(diffusePred )).data.item(), ((torch.sum(specularPredNew) ) / (torch.sum(specularPred) ) ).data.item()
    if cSpec < 1e-3:
        cAlbedo = 1/ albedoPreds[-1].max().data.item()
        cLight = cDiff / cAlbedo
    else:
        cLight = cSpec
        cAlbedo = cDiff / cLight
        cAlbedo = np.clip(cAlbedo, 1e-3, 1 / albedoPreds[-1].max().data.item() )
        cLight = cDiff / cAlbedo
    envmapsPredImages[0] = envmapsPredImages[0] * cLight
    cAlbedos.append(cAlbedo )
    cLights.append(cLight )

    diffusePred = diffusePredNew
    specularPred = specularPredNew

    #################### Output Results #######################
    # Save the albedo
    for n in range(0, len(albedoPreds ) ):
        if n < len(cAlbedos ):
            albedoPred = (albedoPreds[n] * cAlbedos[n]).data.cpu().numpy().squeeze()
        else:
            albedoPred = albedoPreds[n].data.cpu().numpy().squeeze()

        albedoPred = albedoPred.transpose([1, 2, 0] )
        albedoPred = (albedoPred ) ** (1.0/2.2 )
        albedoPred = cv2.resize(albedoPred, (nw, nh), interpolation = cv2.INTER_LINEAR )

        albedoPredIm = (np.clip(255 * albedoPred, 0, 255) ).astype(np.uint8)

        cv2.imwrite(albedoImNames[n], albedoPredIm[:, :, ::-1] )

    # Save the normal
    for n in range(0, len(normalPreds ) ):
        normalPred = normalPreds[n].data.cpu().numpy().squeeze()
        normalPred = normalPred.transpose([1, 2, 0] )
        normalPred = cv2.resize(normalPred, (nw, nh), interpolation = cv2.INTER_LINEAR )

        np.save(normalNames[n], normalPred )

        normalPredIm = (255 * 0.5*(normalPred+1) ).astype(np.uint8)
        cv2.imwrite(normalImNames[n], normalPredIm[:, :, ::-1] )

    # Save the rough
    for n in range(0, len(roughPreds ) ):
        roughPred = roughPreds[n].data.cpu().numpy().squeeze()
        roughPred = cv2.resize(roughPred, (nw, nh), interpolation = cv2.INTER_LINEAR )

        roughPredIm = (255 * 0.5*(roughPred+1) ).astype(np.uint8)
        cv2.imwrite(roughImNames[n], roughPredIm )

    # Save the depth
    for n in range(0, len(depthPreds ) ):
        depthPred = depthPreds[n].data.cpu().numpy().squeeze()
        np.save(depthNames[n], depthPred )

        depthPred = depthPred / np.maximum(depthPred.mean(), 1e-10) * 3
        depthPred = cv2.resize(depthPred, (nw, nh), interpolation = cv2.INTER_LINEAR )

        depthOut = 1 / np.clip(depthPred+1, 1e-6, 10)
        depthPredIm = (255 * depthOut ).astype(np.uint8)
        cv2.imwrite(depthImNames[n], depthPredIm )

    if opt.isBS:
        # Save the albedo bs
        for n in range(0, len(albedoBSPreds ) ):
            if n < len(cAlbedos ):
                albedoBSPred = (albedoBSPreds[n] * cAlbedos[n]).data.cpu().numpy().squeeze()
            else:
                albedoBSPred = albedoBSPreds[n].data.cpu().numpy().squeeze()
            albedoBSPred = albedoBSPred.transpose([1, 2, 0] )
            albedoBSPred = (albedoBSPred ) ** (1.0/2.2 )
            albedoBSPred = cv2.resize(albedoBSPred, (nw, nh), interpolation = cv2.INTER_LINEAR )

            albedoBSPredIm = ( np.clip(255 * albedoBSPred, 0, 255) ).astype(np.uint8)
            cv2.imwrite(albedoImNames[n].replace('albedo', 'albedoBS'), albedoBSPredIm[:, :, ::-1] )

        # Save the rough bs
        for n in range(0, len(roughBSPreds ) ):
            roughBSPred = roughBSPreds[n].data.cpu().numpy().squeeze()
            roughBSPred = cv2.resize(roughBSPred, (nw, nh), interpolation = cv2.INTER_LINEAR )

            roughBSPredIm = (255 * 0.5*(roughBSPred+1) ).astype(np.uint8)
            cv2.imwrite(roughImNames[n].replace('rough', 'roughBS'), roughBSPredIm )


        for n in range(0, len(depthBSPreds) ):
            depthBSPred = depthBSPreds[n].data.cpu().numpy().squeeze()
            np.save(depthNames[n].replace('depth', 'depthBS'), depthBSPred )

            depthBSPred = depthBSPred / np.maximum(depthBSPred.mean(), 1e-10) * 3
            depthBSPred = cv2.resize(depthBSPred, (nw, nh), interpolation = cv2.INTER_LINEAR )

            depthOut = 1 / np.clip(depthBSPred+1, 1e-6, 10)
            depthBSPredIm = (255 * depthOut ).astype(np.uint8)
            cv2.imwrite(depthImNames[n].replace('depth', 'depthBS'), depthBSPredIm )

    if opt.isLight:
        # Save the envmapImages
        for n in range(0, len(envmapsPredImages ) ):
            envmapsPredImage = envmapsPredImages[n].data.cpu().numpy().squeeze()
            envmapsPredImage = envmapsPredImage.transpose([1, 2, 3, 4, 0] )

            # Flip to be conincide with our dataset
            np.savez_compressed(envmapPredImNames[n],
                    env = np.ascontiguousarray(envmapsPredImage[:, :, :, :, ::-1] ) )

            utils.writeEnvToFile(envmapsPredImages[n], 0, envmapPredImNames[n], nrows=24, ncols=16, envHeight=opt.envHeight, envWidth=opt.envWidth)

        for n in range(0, len(envmapsPreds ) ):
            envmapsPred = envmapsPreds[n].data.cpu().numpy()
            np.save(envmapsPredSGNames[n], envmapsPred )
            shading = utils.predToShading(envmapsPred, SGNum = opt.SGNum )
            shading = shading.transpose([1, 2, 0] )
            shading = shading / np.mean(shading ) / 3.0
            shading = np.clip(shading, 0, 1)
            shading = (255 * shading ** (1.0/2.2) ).astype(np.uint8 )
            cv2.imwrite(shadingNames[n], shading[:, :, ::-1] )

        for n in range(0, len(cLights) ):
            io.savemat(cLightNames[n], {'cLight': cLights[n] } )

        # Save the rendered image
        for n in range(0, len(renderedPreds ) ):
            renderedPred = renderedPreds[n].data.cpu().numpy().squeeze()
            renderedPred = renderedPred.transpose([1, 2, 0] )
            # renderedPred = renderedPred / renderedPred.max()
            renderedPred = renderedPred  ** (1.0/2.2)
            renderedPred = cv2.resize(renderedPred, (nw, nh), interpolation = cv2.INTER_LINEAR )
            #np.save(renderedNames[n], renderedPred )

            renderedPred = (np.clip(renderedPred, 0, 1) * 255).astype(np.uint8 )
            cv2.imwrite(renderedImNames[n], renderedPred[:, :, ::-1] )

    # Save the image
    cv2.imwrite(imOutputNames[0], im_cpu[:,:, ::-1] )

    exit(0)
