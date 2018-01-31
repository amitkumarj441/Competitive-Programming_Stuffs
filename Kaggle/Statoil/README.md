# Statoil/C-CORE Iceberg Classifier Challenge
_Ship or iceberg, can you decide from space?_

## Description

Drifting icebergs present threats to navigation and activities in areas such as offshore of the East Coast of Canada.

Currently, many institutions and companies use aerial reconnaissance and shore-based support to monitor environmental conditions and assess risks from icebergs. However, in remote areas with particularly harsh weather, these methods are not feasible, and the only viable monitoring option is via satellite.

[Statoil](https://www.statoil.com/), an international energy company operating worldwide, has worked closely with companies like C-CORE. [C-CORE](https://www.c-core.ca/) have been using satellite data for over 30 years and have built a computer vision based surveillance system. To keep operations safe and efficient, Statoil is interested in getting a fresh new perspective on how to use machine learning to more accurately detect and discriminate against threatening icebergs as early as possible.

In this competition, you’re challenged to build an algorithm that automatically identifies if a remotely sensed target is a ship or iceberg. Improvements made will help drive the costs down for maintaining safe working conditions.

## My Solution

This solution implements a CNN that handles two inputs; One is the image angle, the other one is an image with two channels: HH (transmit/recieve horizontally), and HV(transmit horizontally and recieve vertically).
A very important step before training is the data augmentation. Data augmentation was done with the help of the keras ImageDataGenerator function. The data augmentation consisted of horizontal flipping, vertical flipping, zooming and rotation.
In addition to, a "Min-Max Stacking" step was implemented to even decrease the loss of the model. This should get a score of ~ 0.1478.

