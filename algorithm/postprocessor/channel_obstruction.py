import cv2
import numpy as np

import gv
from logger import LOGGER
from postprocessor import Postprocessor as BasePostprocessor
from window.ratio_window import RatioWindow
from .utils import json_utils
from .utils.cv_utils.color_utils import rgb_reverse
from .utils.cv_utils.crop_utils import crop_poly
from .utils.image_utils import base64_to_opencv, opencv_to_base64
from .utils.unique_id_utils import get_object_id


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.model_name = 'feature'
        self.index = None
        self.group_type = None
        self.similarity = None
        self.timeout = None
        self.reinfer_result = {}
        # 目标最小像素值
        self.min_len = 10
        self.pre_gray = None
        self.threshold = None
        self.length = None
        self.window = None

    def __reinfer(self, roi):
        draw_image = base64_to_opencv(self.draw_image)
        if roi:
            cropped_image = crop_poly(draw_image, roi)
        else:
            cropped_image = draw_image
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        cropped_image = rgb_reverse(cropped_image)
        source_data = {
            'source_id': self.source_id,
            'time': self.time * 1000000,
            'infer_image': opencv_to_base64(cropped_image),
            'draw_image': None,
            'reserved_data': {
                'specified_model': [self.model_name],
                'unsort': True
            }
        }
        self.rq_source.put(json_utils.dumps(source_data))
        self.reinfer_result[self.time] = {
            'count': 1,
            'draw_image': self.draw_image,
            'gray_image': gray_image,
            'result': []
        }
        return True

    def __check_expire(self):
        for time in list(self.reinfer_result.keys()):
            if time < self.time - self.timeout:
                LOGGER.warning('Reinfer result expired, source_id={}, alg_name={}, time={}, timeout={}'.format(
                    self.source_id, self.alg_name, time, self.timeout))
                del self.reinfer_result[time]
        return True

    def __process_blacklist(self, feature):
        id_, score = self.index.search(feature, self.similarity) if self.index is not None else (None, None)
        if id_ is not None:
            base_info = self.index.query(id_)
            if base_info:
                return True, base_info['name']
        return False, '正常'

    def __process_whitelist(self, feature):
        id_, score = self.index.search(feature, self.similarity) if self.index is not None else (None, None)
        if id_ is not None:
            base_info = self.index.query(id_)
            if base_info:
                return False, base_info['name']
        return True, '通道堵塞'

    def __check_move(self, gray_image):
        if self.pre_gray is None:
            self.pre_gray = gray_image.copy()
            return True
        fg_mask = cv2.absdiff(gray_image, self.pre_gray)
        self.pre_gray = gray_image.copy()
        ret, fg_mask = cv2.threshold(fg_mask, 30, 255, cv2.THRESH_BINARY)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        fg_mask = cv2.dilate(fg_mask, element, 2)
        contours, _ = cv2.findContours(fg_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 前景目标
        for cont in contours:
            xywh = cv2.boundingRect(cont)
            xyxy = [xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]]
            if xyxy[2] - xyxy[0] < self.min_len or xyxy[3] - xyxy[1] < self.min_len:
                continue
            return True
        return False

    def _process(self, result, filter_result):
        hit = False
        if self.index is None:
            self.index = gv.index_dic.get(self.reserved_args['group_id'])
        if self.group_type is None:
            self.group_type = self.reserved_args['group_type']
        if self.similarity is None:
            self.similarity = self.reserved_args['similarity']
        if self.length is None:
            self.length = self.reserved_args['length']
        if self.threshold is None:
            if self.length != 0:
                self.threshold = self.reserved_args['threshold'] / self.length
                self.threshold = 0 if self.threshold < 0 else self.threshold
                self.threshold = 1 if self.threshold > 1 else self.threshold
            else:
                self.threshold = 1
            LOGGER.info('source_id={}, alg_name={}, length={}, threshold={}'.format(
                self.source_id, self.alg_name, self.length, self.threshold))
        if self.timeout is None:
            self.timeout = (self.frame_interval / 1000) * 2
            LOGGER.info('source_id={}, alg_name={}, timeout={}'.format(self.source_id, self.alg_name, self.timeout))
        if self.window is None:
            self.window = RatioWindow(self.length, self.threshold)
        roi = self.reserved_args.get('roi')
        if not self.reserved_data:
            self.__reinfer(roi)
            return False
        self.__check_expire()
        polygons = []
        if roi is not None:
            data = {
                'polygons': [{
                    'id': get_object_id(),
                    'name': None,
                    'polygon': roi
                }]
            }
            polygons = self._gen_polygons(data)
        model_name, targets = next(iter(filter_result.items()))
        if self.reinfer_result.get(self.time) is None:
            LOGGER.warning('Not found reinfer result, time={}'.format(self.time))
            return False
        self.reinfer_result[self.time]['result'].append(targets)
        if len(self.reinfer_result[self.time]['result']) < self.reinfer_result[self.time]['count']:
            return False
        reinfer_result_ = self.reinfer_result.pop(self.time)
        self.draw_image = reinfer_result_['draw_image']
        gray_image = reinfer_result_['gray_image']
        move_flag = self.__check_move(gray_image)
        if not move_flag:
            for targets in reinfer_result_['result']:
                for target in targets:
                    feature = target.pop('feature', None)
                    if feature is not None:
                        np_feature = np.array(feature, dtype=np.float32)
                        if 'blacklist' == self.group_type:
                            hit_, label = self.__process_blacklist(np_feature)
                        elif 'whitelist' == self.group_type:
                            hit_, label = self.__process_whitelist(np_feature)
                        else:
                            LOGGER.error('Unknown group_type: {}'.format(self.group_type))
                            continue
                        if self.window.insert({'time': self.time, 'data': {'hit': hit_}}):
                            hit = hit_
                            for _, value in polygons.items():
                                value['color'] = self.alert_color
        result['hit'] = hit
        result['data']['bbox']['polygons'].update(polygons)
        result['data']['group'] = {
            'id': self.index.group_id if self.index is not None else None,
            'name': self.index.group_name if self.index is not None else None
        }
        return True

    def _filter(self, model_name, model_data):
        targets = []
        if not self.reserved_data:
            return targets
        engine_result = model_data['engine_result']
        targets.append({
            'feature': engine_result
        })
        return targets
