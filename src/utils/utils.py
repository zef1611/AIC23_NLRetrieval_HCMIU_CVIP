import cv2 as cv
import os
from typing import List
import torch
from datetime import date, datetime
import os
import numpy as np
import random
import platform
import logging
import os
import  torch
import sys
import os.path as osp
import random
import numpy as np
import refile
import io
import logging
import os
import colorlog
import os.path as osp
import sys
import errno
import numpy as np
import random
import torch
from torchmetrics import RetrievalMRR
import torch


def showInMovedWindow(winname, img, x, y):
    cv.namedWindow(winname)  # Create a named window
    cv.moveWindow(winname, x, y)  # Move it to (x,y)
    cv.imshow(winname, img)


def getCamCapture(data):
    """Returns the camera capture from parsing or a pre-existing video.

    Args:
      isParse: A boolean value denoting whether to parse or not.

    Returns:
      A video capture object to collect video sequences.
      Total video frames

    """
    total_frames = None
    if os.path.isdir(data):
        cap = cv.VideoCapture(data + "/input/in%06d.jpg")
        total_frames = len(os.listdir(os.path.join(data, "input")))
    else:
        cap = cv.VideoCapture(data)
    return cap, total_frames


def deep_update(mapping: dict, *updating_mappings: dict()) -> dict():
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


def get_device(choose_device):
    if torch.cuda.is_available():
        device = "cuda:0"
    if choose_device == "cpu" or device == "cpu":
        return "cpu"
    return device


def get_dict_infor(_dict: dict) -> List[str]:
    res_list = []
    for k, v in _dict.items():
        if isinstance(v, dict):
            get_list = get_dict_infor(v)
            for val in get_list:
                res_list.append(str(k) + "." + str(val))
        else:
            res_list.append(str(k))
    return res_list

def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        if if_train:
            fh = logging.FileHandler(os.path.join(save_dir, "train_log.txt"), mode='w')
        else:
            fh = logging.FileHandler(os.path.join(save_dir, "test_log.txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def update_cfg(cfg: List, value):

    if len(cfg) == 1:
        return {cfg[0]: value}
    return {cfg[0]: update_cfg(cfg[1:], value)}

class WriteLog(object):
    def __init__(self, cfg, save_path, isFirstTime,_type):
        self.type = _type
        today = date.today()
        self.current_date = today.strftime("_%d_%m_%Y")
        self.log_file = os.path.join(save_path, self.type + self.current_date + ".txt")
        self.current_date = today.strftime("%d-%m-%Y")

        self.current_time = datetime.now()
        self.current_time = self.current_time.strftime("%H:%M:%S")

       
        if isFirstTime == True:
            f = open(self.log_file, 'a')
            f.write('\n\n\n')
            f.write(self.current_date + " " + self.current_time + " " + "STARTING TO " + self.type + "\n")
            f.write(str(cfg) +  '\n')
            f.close()
           
        return
    def update(self, infor):
        f = open(self.log_file, 'a')
        f.write(str(infor) + '\n')
        f.close()
        return


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_device_name():

    return platform.node()


def config2object(config):
    """
    Convert dictionary into instance allowing access to dictionary keys using
    dot notation (attributes).
    """
    class ConfigObject(dict):
        """
        Represents configuration options' group, works like a dict
        """

        def __init__(self, *args, **kwargs):
            dict.__init__(self, *args, **kwargs)

        def __getattr__(self, name):
            return self[name]

        def __setattr__(self, name, val):
            self[name] = val

    if isinstance(config, dict):
        result = ConfigObject()
        for key in config:
            result[key] = config2object(config[key])
        return result
    else:
        return config

class TqdmToLogger(io.StringIO):
    logger = None
    level = None
    buf = ''

    def __init__(self):
        super(TqdmToLogger, self).__init__()
        self.logger = get_logger('tqdm')

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.info(self.buf)


def get_logger(logger_name='default', debug=False, save_to_dir=None):
    if debug:
        log_format = (
            '%(asctime)s - '
            '%(levelname)s : '
            '%(name)s - '
            '%(pathname)s[%(lineno)d]:'
            '%(funcName)s - '
            '%(message)s'
        )
    else:
        log_format = (
            '%(asctime)s - '
            '%(levelname)s : '
            '%(name)s - '
            '%(message)s'
        )
    bold_seq = '\033[1m'
    colorlog_format = f'{bold_seq} %(log_color)s {log_format}'
    colorlog.basicConfig(format=colorlog_format, datefmt='%y-%m-%d %H:%M:%S')
    logger = logging.getLogger(logger_name)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if save_to_dir is not None:
        fh = logging.FileHandler(os.path.join(save_to_dir, 'log', 'debug.log'))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        fh = logging.FileHandler(
            os.path.join(save_to_dir, 'log', 'warning.log'))
        fh.setLevel(logging.WARNING)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        fh = logging.FileHandler(os.path.join(save_to_dir, 'log', 'error.log'))
        fh.setLevel(logging.ERROR)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", writeLog = None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.writeLog = writeLog

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        self.writeLog.update('\t'.join(entries))
        return 
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_mrr(sim_mat):
    mrr = RetrievalMRR()
    return mrr(
        sim_mat.flatten(),
        torch.eye(len(sim_mat), device=sim_mat.device).long().bool().flatten(),
        torch.arange(len(sim_mat), device=sim_mat.device)[:, None].expand(len(sim_mat), len(sim_mat)).flatten(),
    )
    pass


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # pred(correct.shape)
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def getAvg(self):
        return self.avg


def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class MgvSaveHelper(object):
    def __init__(self, save_oss=False, oss_path='', echo=True):
        self.oss_path = oss_path
        self.save_oss = save_oss
        self.echo = echo

    def set_stauts(self, save_oss=False, oss_path='', echo=True):
        self.oss_path = oss_path
        self.save_oss = save_oss
        self.echo = echo

    def get_s3_path(self, path):
        if self.check_s3_path(path):
            return path
        return self.oss_path + path

    def check_s3_path(self, path):
        return path.startswith('s3:')

    def load_ckpt(self, path):
        if self.check_s3_path(path):
            with refile.smart_open(path, "rb") as f:
                ckpt = torch.load(f)
        else:
            ckpt = torch.load(path)
        if self.echo:
            print(f"====> load checkpoint from {path}")
        return ckpt

    def save_ckpt(self, path, epoch, model, optimizer=None):
        if self.save_oss:
            if not self.check_s3_path(path):
                path = self.get_s3_path(path)
            with refile.smart_open(path, "wb") as f:
                torch.save(
                    {"epoch": epoch,
                     "state_dict": model.state_dict(),
                     "optimizer": optimizer.state_dict()}, f)
        else:
            torch.save(
                {"epoch": epoch,
                 "state_dict": model.state_dict(),
                 "optimizer": optimizer.state_dict()}, path)

        if self.echo:
            print(f"====> save checkpoint to {path}")

    def save_pth(self, path, file):
        if self.save_oss:
            if not self.check_s3_path(path):
                path = self.get_s3_path(path)
            with refile.smart_open(path, "wb") as f:
                torch.save(file, f)
        else:
            torch.save(file, path)

        if self.echo:
            print(f"====> save pth to {path}")

    def load_pth(self, path):
        if self.check_s3_path(path):
            with refile.smart_open(path, "rb") as f:
                ckpt = torch.load(f)
        else:
            ckpt = torch.load(path)
        if self.echo:
            print(f"====> load pth from {path}")
        return ckpt
