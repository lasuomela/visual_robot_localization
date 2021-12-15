from pathlib import Path
import torch
import cv2
import numpy as np

from hloc import extractors, matchers
from hloc.utils.base_model import dynamic_load

class FeatureExtractor:

    def __init__(self, conf):
        self.prep_conf = conf['preprocessing']

        if not 'resize_force' in self.prep_conf:
            self.prep_conf['resize_force'] = False

        if not 'grayscale' in self.prep_conf:
            self.prep_conf['grayscale'] = False

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Model = dynamic_load(extractors, conf['model']['name'])
        self.model = Model(conf['model'])
        if hasattr(self.model, 'eval'):
            self.model.eval()
        self.model = self.model.to(self.device)

    def _preprocess_img(self, img, disable_grayscale=False):
        img = img.astype(np.float32)
        size = img.shape[:2][::-1]
        w, h = size

        if (self.prep_conf['grayscale']) & (not disable_grayscale):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if self.prep_conf['resize_max'] and (self.prep_conf['resize_force']
                                     or max(w, h) > self.prep_conf['resize_max']):
            scale = self.prep_conf['resize_max'] / max(h, w)
            h_new, w_new = int(round(h*scale)), int(round(w*scale))
            img = cv2.resize(
                img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

        if self.prep_conf['grayscale']:
            img = np.expand_dims(img, axis=(0,1))
        else:
            # Switch HxWxC -> CxHxW
            img = img.transpose((2, 0, 1))
            img = np.expand_dims(img, axis=0)

        img = torch.tensor(img/255., dtype=torch.float32).to(self.device)
        return img, size

    @torch.no_grad()
    def __call__(self, img, disable_grayscale=False):
        img, size = self._preprocess_img(img, disable_grayscale)
        data_dict = {'image': img, 'original_size': size}
        
        pred = self.model(data_dict)
        pred['image_size'] = original_size = data_dict['original_size']
        
        if 'keypoints' in pred:
            size = np.array(data_dict['image'].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)

            # D2net returns keypoint tensor on cpu, should be fixed in hloc code (extractors/d2net.py)
            pred['keypoints'] = (pred['keypoints'][0].to(self.device) + .5) * torch.tensor(scales[None]).to(self.device) - .5

        if 'scores' in pred:
            pred['scores'] = pred['scores'][0]

        if 'descriptors' in pred:
            pred['descriptors'] = pred['descriptors'][0]

        if 'global_descriptor' in pred:
            if pred['global_descriptor'].is_cuda:
                pred['global_descriptor'] = pred['global_descriptor'].cpu()

        return pred

class FeatureMatcher:

    def __init__(self, conf):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Model = dynamic_load(matchers, conf['model']['name'])
        self.model = Model(conf['model'])
        if hasattr(self.model, 'eval'):
            self.model.eval()
        self.model = self.model.to(self.device)

    def prepare_data(self, desc1, desc2, img_shape):
        data = {}
        data['descriptors0'] = desc1['descriptors'].unsqueeze(0).to(self.device) ## FIX THE DEVICE TRANSFERS
        data['descriptors1'] = desc2['descriptors'].unsqueeze(0).to(self.device)
        data['keypoints0'] = desc1['keypoints']
        data['keypoints1'] = desc2['keypoints']
        data['scores0'] = desc1['scores'].unsqueeze(0)
        data['scores1'] = desc2['scores'].unsqueeze(0)
        data['image0'] = torch.empty((1, 1, img_shape[0], img_shape[1]))
        data['image1'] = torch.empty((1, 1, img_shape[0], img_shape[1]))
        return data

    @torch.no_grad()
    def __call__(self, data):
        pred = self.model(data)
        matches = pred['matches0'][0].cpu().short().numpy()
        scores = pred['matching_scores0'][0].cpu().half().numpy()
        return matches, scores
