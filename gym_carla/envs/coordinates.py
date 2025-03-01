#!/usr/bin/env python

# x y z pitch(y) row(x) yaw(z)
def train_coordinates(task_mode):
    starts = {
        'Straight':
            [[322.09, 129.70, 1.5, 0.0, 0.0, 180.0],
             [88.13, 4.32, 1.5, 0.0, 0.0, 90.0],
             [392.47, 87.41, 1.5, 0.0, 0.0, 90.0],
            [383.18, -2.20, 1.5, 0.0, 0.0, 180.0],
            [283.67, 129.48, 1.5, 0.0, 0.0, 180.0]],
         'One_curve':
            [[88.62, 285.39, 1.5, 0.0, 0.0, 90.00],
         ]}
    dests = {
        'Straight':
            [[119.47, 129.75, 1.5, 0.0, 0.0, 180.0],
             [88.13, 299.92, 1.5, 0.0, 0.0, 90.0],
             [392.47, 308.21, 1.5, 0.0, 0.0, 90.0],
             [185.55, -1.95, 1.5, 0.0, 0.0, 180.0],
             [128.94, 129.75, 1.5, 0.0, 0.0, 180.0]],
        'One_curve':
            [[105.57, 330.85, 1.5, 0.0, 0.0, 0.0]

            ]
    }
    return starts[task_mode], dests[task_mode]