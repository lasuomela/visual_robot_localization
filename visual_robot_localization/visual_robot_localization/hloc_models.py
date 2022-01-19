from pathlib import Path
import torch
import cv2
import numpy as np

from torch.nn.utils.rnn import pad_sequence

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

    def prepare_data(self, query_feature, gallery_features, img_shape):

        data = {}

        gallery_descriptors = [ desc['descriptors'].transpose(0,1) for desc in gallery_features ]
        gallery_keypoints = [ desc['keypoints'] for desc in gallery_features ]
        gallery_scores = [ desc['scores'] for desc in gallery_features ]

        # Zero-pad the gallery descriptors/keypoints/scores according to the feature with most keypoints
        data['descriptors1'] = pad_sequence(gallery_descriptors, batch_first=True).transpose(1,2)
        data['keypoints1'] = pad_sequence(gallery_keypoints, batch_first=True)
        data['scores1'] = pad_sequence(gallery_scores, batch_first=True)
        batch_size = data['descriptors1'].shape[0]

        # Repeate query descriptor N times in batch dimension to match number of gallery descriptors
        data['descriptors0'] = query_feature['descriptors'].unsqueeze(0).repeat( batch_size, 1, 1)
        data['keypoints0'] = query_feature['keypoints'].unsqueeze(0).repeat( batch_size, 1, 1)
        data['scores0'] = query_feature['scores'].unsqueeze(0).repeat(batch_size,1)
        data['image0'] = torch.empty((batch_size, 1, img_shape[0], img_shape[1]))
        data['image1'] = torch.empty((batch_size, 1, img_shape[0], img_shape[1]))
        return data


    @torch.no_grad()
    def __call__(self, data):
        pred = self.model(data)
        matches = pred['matches0'].cpu().short().numpy()
        scores = pred['matching_scores0'].cpu().half().numpy()
        return matches, scores