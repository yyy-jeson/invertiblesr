import os.path as osp
import logging, os, argparse,time
from datetime import datetime
from collections import OrderedDict
import cv2,subprocess,av

from sklearn import semi_supervised
import numpy as np
import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model

logger = logging.getLogger('base')

#### options
def make_tmp_dir(tmpdir):
    dtstr = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
    if not osp.exists(tmpdir):
        os.makedirs(tmpdir)
    lrdir =  osp.join(tmpdir, "LR-"+dtstr)
    if not osp.exists( lrdir):
        os.makedirs(lrdir)
    lrvfile = osp.join(tmpdir, "LR-"+dtstr+".mp4")
    hrdir = osp.join(tmpdir, "HR-"+dtstr)
    if not osp.exists( hrdir):
        os.makedirs(hrdir)
    hrvfile = osp.join(tmpdir, "HR-"+dtstr+".mp4")
    return lrdir, lrvfile, hrdir, hrvfile

class HYInvSR:
    def __init__(self, opt, ffmpeg_cmd="/workspace/cpfs-data/data/srdata/tools/ffmpeg") -> None:
        self.model = create_model(opt)
        self.ffmpeg_cmd = ffmpeg_cmd
        pass

    def _hr2lr(self, img):
        img_tensor = util.npimg2tensor(img)    
        lr_img_tensor = self.model.downscale(img_tensor)
        return util.tensor2img(lr_img_tensor)
    
    def _lr2hr(self, img):
        img_tensor = util.npimg2tensor(img)    
        lr_img_tensor = self.model.upscale(img_tensor)
        return util.tensor2img(lr_img_tensor) 
    
    def hr2hrp(self, imgfile, savedir):
        if not osp.exists(imgfile):
            logger.info(f"Failed to locate: {imgfile}")
            return
        frame = cv2.imread(imgfile)
        lr_img = self._hr2lr(frame)
        img_index = 1;
        img_file_path = "{}/{}.png".format(savedir, ("%05d"%img_index))
        util.save_img(lr_img, img_file_path)

    def hrv2lr(self, videofile, savedir, img_limit=5000):
        img_count = 0
        (vname, fext) = osp.splitext( osp.basename(videofile) )
        if fext != ".mp4":
           logger.error(f"error: {videofile}")
           return img_count
        if not osp.exists(savedir):
           os.makedirs(savedir)
    
        try:
            v_cont = cv2.VideoCapture(videofile)
            img_index = 0
            while True:
                ret, frame = v_cont.read()
                if ret == False:
                    break
                img_index += 1
                img_file_path = "{}/{}.png".format(savedir, ("%05d"%img_index))
                #lr_img = self._hr2lr(frame)
                #util.save_img(lr_img, img_file_path)
                util.save_img(frame, img_file_path)
                break
        except Exception as e:
            print(e)
        return img_count        

    def lrdir2hr(self, lrdir, lrvfile, hrdir, hrvfile):
        logger.info(f"transcode {lrdir} to {lrvfile}")
        util.ysp_ffmpeg_transcode_video_file(lrdir, lrvfile, self.ffmpeg_cmd)

        try:
            v_cont = cv2.VideoCapture(lrvfile)
            img_index = 0
            while True:
                ret, frame = v_cont.read()
                if ret == False:
                    break
                img_index += 1
                img_file_path = "{}/{}.png".format(hrdir, ("%05d"%img_index))
                lr_img = self._lr2hr(frame)
                util.save_img(lr_img, img_file_path)
        except Exception as e:
            print(f"{lrvfile} error:{e}") 
        
        logger.info(f"transcode {hrdir} to {hrvfile}")
        util.ysp_ffmpeg_transcode_video_file(hrdir, hrvfile, self.ffmpeg_cmd)


#python3 pipeline.py -opt ./options/test/test_IRN_x2_down.yml --vtool ./ffmpeg --vfile ./v_10002.mp4
#python3 pipeline.py -opt ./options/test/test_IRN_x2_down.yml --vtool ./ffmpeg -m 0 -f ./source.png
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
    parser.add_argument('--vtool', default='./ffmpeg')
    parser.add_argument("--vfile", default="./loleval.mp4" )
    parser.add_argument("-f", default="./c.png")
    parser.add_argument("-m", type=int, default=0, help="Generate Low Resolution video.")
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    util.mkdirs( (path for key, path in opt['path'].items()  if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
    util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO, \
        screen=True, tofile=True)
    logger.info(option.dict2str(opt))

    tmpdir = "./tmp"
    lrdir, lrvfile, hrdir, hrvfile = make_tmp_dir(tmpdir)
    invsr = HYInvSR(opt, args.vtool)
    if args.m == 0:
        invsr.hr2hrp( args.f, lrdir)

    #invsr.hrv2lr(args.vfile, lrdir, img_limit=5000)
    #invsr.lrdir2hr(lrdir, lrvfile, hrdir, hrvfile)
    

