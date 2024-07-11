from logger import LOGGER
from postprocessor import Postprocessor as BasePostprocessor
from .utils.cv_utils.geo_utils import calc_iou
from .utils.image_utils import base64_to_opencv


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.fall_down_model_name = 'fall_down'
        self.person_model_name = 'person'
        self.alert_label = '跌倒'
        self.iou = None
        self.image_shape = None
        self.distance = 10

    def _process(self, result, filter_result):
        hit = False
        if self.iou is None:
            self.iou = self.reserved_args['iou']
        if self.image_shape is None:
            draw_image = base64_to_opencv(self.draw_image)
            self.image_shape = draw_image.shape
        polygons = self._gen_polygons()
        fall_down_rectangles = filter_result.get(self.fall_down_model_name)
        if fall_down_rectangles is None:
            LOGGER.error('Fall down model result is None!')
            return False
        person_rectangles = filter_result.get(self.person_model_name)
        if person_rectangles is None:
            LOGGER.error('Person model result is None!')
            return False
        for person_rectangle in person_rectangles:
            if person_rectangle['xyxy'][3] > self.image_shape[0] - self.distance:
                continue
            for fall_down_rectangle in fall_down_rectangles:
                if calc_iou(person_rectangle['xyxy'], fall_down_rectangle['xyxy']) > self.iou:
                    hit = True
                    person_rectangle['color'] = self.alert_color
                    person_rectangle['label'] = self.alert_label
                    break
        result['hit'] = hit
        result['data']['bbox']['rectangles'].extend(person_rectangles)
        result['data']['bbox']['polygons'].update(polygons)
        return True
