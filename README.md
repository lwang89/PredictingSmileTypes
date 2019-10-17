# PredictingSmileTypes

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.<br>
### Prerequisites
1. Python 3.  We're using 3.6.2.
You can get it here: https://www.python.org/downloads/<br>
2. dlib.<br>
3. openCV(cv2).<br>
4. Pandas, NumPy and scikit-learn.
5. PyTorch. We're using 0.3.0.
You can get it here: https://pytorch.org/get-started/locally/<br>
## facial_landmarks_dection
We applied openCV(cv2) and dlib to fetch 68 landmarks( the predictor was trained on [iBUG 300-W dataset](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)).<br>
The whole methodology followed the idea from Adrian Rosebrock([Here is the link:)](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/).<br>

We generated the data for traditional ML models and LSTM models during processing frames.<br> Basicly, traditional Ml models needed me calculate mean and standard deviation values for every 30 frames while we feeded LSTM model raw data (a good boy, right?).<br>

If you don't understand the code, feel free to contact me:)

### human-level_performance_feedback_collection
We chose FLASK as our framework. It's simple and elegant. We used a Bootstrap template for front-end. You can google and use any template you like:)<br>
Here are screenshots how we collect human-level-performance feedback.
![login page](https://github.com/lwang89/PredictingSmileTypes/blob/master/human-level_performance_feedback_collection/images/1.png)
login page<br>
![submit result](https://github.com/lwang89/PredictingSmileTypes/blob/master/human-level_performance_feedback_collection/images/2.png)
submit result<br>
![rest page](https://github.com/lwang89/PredictingSmileTypes/blob/master/human-level_performance_feedback_collection/images/3.png)
rest page<br>
### traditional_ML
We apply different traditional ML models here.

### useful_facial_landmarks

### LSTM
LSTM model is deployed here.
## Built With
* [Flask](https://palletsprojects.com/p/flask/) - An web framework used
* [OpenCV](https://opencv.org) - An open source computer vision and machine learning software library.
* [Dlib](http://dlib.net/) - The C++ toolkit containing machine learning algorithms and tools used.
* [Scikit-learn](https://scikit-learn.org/stable/) - A free software machine learning library for the Python programming language used.
* [PyTorch] (https://pytorch.org) - An open source machine learning library based on the Torch library used.
## Authors
* **Leon Wang** - [lwang89](https://github.com/lwang89)
* **David Guy Brizan** - [dbrizan](https://github.com/dbrizan)
## Acknowledgments
* Beste Yuksel
* Adrian Rosebrock
