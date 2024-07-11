import numpy as np

from logger import LOGGER
from postprocessor import Postprocessor as BasePostprocessor


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.model_name = 'lpr'

    def _process(self, result, filter_result):
        hit = False
        polygons = self._gen_polygons()
        lpr_rectangles = filter_result.get(self.model_name)
        if lpr_rectangles is None:
            LOGGER.error('Lpr model result is None!')
            return result
        for lpr_rectangle in lpr_rectangles:
            hit = True
            label = '{} {}'.format(
                self._get_ext(lpr_rectangle, 'plate_code'), self._get_ext(lpr_rectangle, 'plate_type'))
            lpr_rectangle['label'] = label
            lpr_rectangle['color'] = self.non_alert_color
        result['hit'] = hit
        result['data']['bbox']['rectangles'].extend(lpr_rectangles)
        result['data']['bbox']['polygons'].update(polygons)
        return result

    def _filter(self, model_name, model_data):
        targets = []
        model_conf = model_data['model_conf']
        engine_result = model_data['engine_result']
        for engine_result_ in engine_result:
            # 过滤掉置信度低于阈值的目标
            rec_conf = engine_result_['rec_conf']
            min_conf = np.min(np.array(rec_conf))
            if not self._filter_by_conf(model_conf, min_conf):
                continue
            # 坐标缩放
            xyxy = self._scale(engine_result_['xyxy'])
            # 过滤掉不在多边形内的目标
            if not self._filter_by_roi(xyxy):
                continue
            # 生成矩形框
            targets.append(self._gen_rectangle(
                xyxy, self.non_alert_color, None, engine_result_['det_conf'],
                plate_code=engine_result_['plate_code'], plate_type=engine_result_['plate_type']))
        return targets
