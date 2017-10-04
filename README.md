# Found in Time

### Introduction ###
Found in Time is a recommend system for discovering the perfect wristwatch. Given a user uploaded image, Found in Time will recommend visually similar watches. If he or she likes a watch, a user can click "More Like This" in order to get new recommendations based on that image. The banner image is updated based on the characteristics of the watch in order to provide a more personalized user experience. Please refer to the high level [technical presentation](https://github.com/leem99/found_in_time/blob/master/FoundInTime_MitchellLee.pdf), and [demo](https://www.youtube.com/watch?v=B1PFtzsGLWk) to learn more.

### Technical Overview ###
Convolutional Neural Networks (CNN) were used for two tasks:
1) Make wristwatch recommendations based on visual similarity.
2) Classify specific features of the watch (such as material, and intended gender). Based on those features, generate personalized banners to individualize and improve the customer experiance.

### Contents ###
* [data_gathering](https://github.com/leem99/found_in_time/tree/master/data_gathering): Scripts used to scrape watch images and characteristics from a [luxury retailer](https://www.prestigetime.com/). Folder also contains scripts for consoldating watch images into csv files. csv files are included as well.
* [image_processing](https://github.com/leem99/found_in_time/tree/master/image_processing): Scripts used to prepare images for modeling.
* [modeling](https://github.com/leem99/found_in_time/tree/master/modeling): Contains scripts for creating the recommendations (VGG16, Xception, and Inception-v3 model were all tested). Folder also contains scripts for transfer learning of Inception-v3 and Xception for classification tasks. 
* [model_testing](https://github.com/leem99/found_in_time/tree/master/model_testing): Scripts for testing the recommendation and classifcation models. Other than the accuracy reported during CNN training, results are currently evaluated using visual inspection when providing the models with new images. 
* [flask_app](https://github.com/leem99/found_in_time/tree/master/flask_app): HTML and flask code used to generate a [prototype website](https://www.youtube.com/watch?v=B1PFtzsGLWk).
