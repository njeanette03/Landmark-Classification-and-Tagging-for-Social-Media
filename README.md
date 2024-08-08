# Landmark-Classification-and-Tagging-for-Social-Media
Construct and train Convolutional Neural Networks (CNN) to predict the location of the image based on any landmarks depicted in the image.

## Project Overview

Photo sharing and photo storage services like to have location data for each photo that is uploaded. With the location data, these services can build advanced features, such as automatic suggestion of relevant tags or automatic photo organization, which help provide a compelling user experience. Although a photo's location can often be obtained by looking at the photo's metadata, many photos uploaded to these services will not have location metadata available. This can happen when, for example, the camera capturing the picture does not have GPS or if a photo's metadata is scrubbed due to privacy concerns.

If no location metadata for an image is available, one way to infer the location is to detect and classify a discernible landmark in the image. Given the large number of landmarks across the world and the immense volume of images that are uploaded to photo sharing services, using human judgment to classify these landmarks would not be feasible.

In this project, we addressing this problem by building models to automatically predict the location of the image based on any landmarks depicted in the image. 

<img alt="Examples from the landmarks dataset - a road in Death Valley, the Brooklyn Bridge, and the Eiffel Tower" src="https://video.udacity-data.com/topher/2021/February/602dac82_landmarks-example/landmarks-example.png" class="chakra-image css-mvsohj"> *Examples from the landmarks dataset - a road in Death Valley, the Brooklyn Bridge, and the Eiffel Tower*

## Project Summary

### Step 1
A CNN is created to Classify Landmarks (from Scratch) - with a focus on visualizing the dataset, processing it for training, and then building a convolutional neural network from scratch to classify the landmarks. The best network will then be exported using Torch Script.

### Step 2
A CNN is created to Classify Landmarks (using Transfer Learning) - Different pre-trained models are explored and one is decided on to use for this classification task, along with training and testing this transfer-learned network. The best transfer learning solution is exported using Torch Script.

### Step 3
The algorithm is deployed in an app - Finally, using the best model to create a simple app for others to be able to use your model to find the most likely landmarks depicted in an image. 
