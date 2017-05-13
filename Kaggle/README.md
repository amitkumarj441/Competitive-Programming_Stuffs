# Kaggle Challenge

## NOAA Fisheries Stellar Sea Lion Population Count

Description : Steller sea lions in the western Aleutian Islands have declined 94 percent in the last 30 years. The endangered western population, found in the North Pacific, are the focus of conservation efforts which require annual population counts. Specially trained scientists at NOAA Fisheries Alaska Fisheries Science Center conduct these surveys using airplanes and unoccupied aircraft systems to collect aerial images. Having accurate population estimates enables us to better understand factors that may be contributing to lack of recovery of Stellers in this area.

Currently, it takes biologists up to four months to count sea lions from the thousands of images NOAA Fisheries collects each year. Once individual counts are conducted, the tallies must be reconciled to confirm their reliability. The results of these counts are time-sensitive.

In this competition, Kagglers are invited to develop algorithms which accurately count the number of sea lions in aerial photographs. Automating the annual population count will free up critical resources allowing NOAA Fisheries to focus on ensuring we hear the sea lion’s roar for many years to come. Plus, advancements in computer vision applied to aerial population counts may also greatly benefit other endangered species.

## Objective
The purpose of this competition is to develop a model that can identify various species of fish from images captured from elevated cameras on board fishing vessels.

## Challenge and Data
The details of this competition can be found on the [Kaggle Competition](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring) page. The data is available to [download](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data).

## Exploring Dataset
A detailed analysis of the data set and label classes can be found in the [Fishery Data Exploration and Preprocessing.ipynb](https://github.com/amitkumarj441/Programming-Stuffs/commit/99c37f5b540128dcd45f61466446d677b46888ae) file in this repository. Important aspects of the data include :

- 8 mutually exclusive fish classes
- 3777 training set images - stored in separate directories for each class
- 1000 stage 1 test set images (no label provided)
- 12153 stage 2 test set images (no label provided) - available only during the last 7 days of the competition.

All training set images were medium resolution, with varying dimensions and aspect ratios. In general, images capture the entire deck that may or may not contain a fish, fisherman, equipment, etc, and were taken during both night and day.

## Strategy
While seemingly a straight forward computer-vision object classification task, there were several complicating factors that made this competition interesting. The major issue was the complicated composition of each image (with people moving, different boats, equipment, partial views of fish, etc). As such, just 3777 images, of which the set was severely imbalanced with some fish classes representing less than 5% of the total, was to be considered an undersized dataset. Moreover, it was clear that in order for the model to generalize in the future, the model would need to accommodate images from unseen boats and camera angles and heights.

With this in mind, my main concern was to keep the model from exhibiting high variance, even at the expense of predictive bias in the training phases.



