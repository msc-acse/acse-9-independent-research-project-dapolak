# -*- coding: utf-8 -*-
"""
Part of slugdetection package

@author: Deirdree A Polak
github: dapolak
"""

name = "slugdetection"

__all__ = ["Data_Engineering", "confusion_mat", "Slug_Labelling",
           "Flow_Recognition", "Slug_Detection", "Slug_Forecasting"]

from slugdetection.Data_Engineering import Data_Engineering
from slugdetection.Data_Engineering import confusion_mat
from slugdetection.Slug_Labelling import Slug_Labelling
from slugdetection.Flow_Recognition import Flow_Recognition
from slugdetection.Slug_Detection import Slug_Detection
from slugdetection.Slug_Forecasting import Slug_Forecasting