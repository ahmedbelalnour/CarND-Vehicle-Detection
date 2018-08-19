## Vehicle Detection Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./output_images/hog_image.png
[image3]: ./output_images/sliding_window.png
[image4]: ./output_images/heat_map.png
[video1]: ./output_videos/project_video.mp4


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the third code cell of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HLS` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters while training my classification model, and I found that the parameters mentioned above resulted in the best accuracy.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a non linear SVM using HOG, color bin and color histogram features in 10th code cell of the IPython notebook. My final model was a non Linear SVM trained on HLS images, the used parameters are as the following: 

* color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
* spatial_size = (32, 32) # Spatial binning dimensions
* hist_bins = 32    # Number of histogram bins
* orient = 9  # HOG orientations
* pix_per_cell = 8 # HOG pixels per cell
* cell_per_block = 2 # HOG cells per block
* hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
* spatial_feat = True # Spatial features on or off
* hist_feat = True # Histogram features on or off
* hog_feat = True # HOG features on or off

This resulted test Accuracy of SVC =  99.72% I trained on the GTI and extra data and tested on the other directories to avoid the risk of overfitting that came with having similar images in the same directory.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented a multi size sliding window search in multi_size_slide_window() in the 12th code cell of the IPython notebook with overlap 75%. I searched on windows of scales 95x95 to 140x140 with step 10px over the whole road area `y_start_stop = [350, 700]`, because these searches produce good results. Then I filtered false positives by implementing a threshold on the heatmap in the 14th code cell of the IPython notebook.
![alt text][image3]

Here is an example of a multi-size sliding window and heatmap of detections made on a test image:
![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?
I implemented the complete pipeline in process_image() in the 17th code cell of the IPython notebook.

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_videos/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of test images, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The multi size sliding window detection is currently take alot of time. In the future I would like to speed up the process and improve performance by using a CNN classifier on a GPU. I would also like to combine this project with the advanced lane finding project to create a full lane and car detection pipeline.

