#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 17:17:57 2019

@author: ai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 17:54:39 2019

@author: NGUYEN HOANG PHUONG
"""




from utils import exact_file, split_train_valid_set

from parser import parse_args

args = parse_args()


exact_file(args)
#process_test_data(args)
split_train_valid_set(args)
