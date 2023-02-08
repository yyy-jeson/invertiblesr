import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import numpy as np
import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)

####################################################
print('model testing')
count = 1
readNum = 10
readDir = '/workspace/cpfs-data/ffmpeg_tools/iir_lol_LR_compress'
saveDir = '/workspace/cpfs-data/ffmpeg_tools/iir_lol_LR_compress_SR'
        
while True:
    lrpath = '{}/{}.png'.format(readDir, ("%05d" % count))
    #lrpath = '{}/{}_LR.png'.format(readDir, ("%05d" % count))
    lrtensor = util.img2tensor(lrpath)
    if lrtensor == None:
        print('lrtensor == None')
        #break
    else:
        #model.print(lrtensor)
        #hrtensor = model.downscale(lrtensor)
        hrtensor = model.upscale(lrtensor, 2)
        sr_img = util.tensor2img(hrtensor)

        hrpath = '{}/{}.png'.format(saveDir, ("%05d" % count))
        util.save_img(sr_img, hrpath)

    if count > readNum:
            break
    count = count + 1

print('exit')
exit()
##############################################################