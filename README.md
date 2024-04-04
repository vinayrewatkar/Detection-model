# Licence-Plate-Detection-using-YOLO-V8

Welcome to the detection model repository

**Few things to note**

1. The images used as well as detected images in ``results`` directory are only for demo purpose here, new images should be used for detection purpose.

2. ``newpts.pt`` is an entire trained model for this project and should not be changed or removed anyhow...

3. This model is not yet integrated and hence you can use it saperately



## **Steps to run the model:** 

1. Clone the repo into local system: ``git clone https://github.com/vinayrewatkar/Detection-model.git``

2. install requirements: (use any IDE like pycharm or jupyter notebook and ensure your interpreter is properly configured for python 3, else download python3 and set its environment variable globally.. also if need, update pip command)

   ``pip install -r requirements.txt``

3. open python console or powershell in same IDE:

   For python console:

   ``!python ultralytics/yolo/v8/detect/predict.py model='newpts.pt' source='/replace with image path'``

   for powershell:

   ``python ultralytics/yolo/v8/detect/predict.py model='newpts.pt' source='/image path'``

4. If cuda and pytorch is properly configured in your model then you will be able to see detected image in ``results`` directory..

All the best...:)
