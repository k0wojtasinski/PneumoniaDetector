# Pneumonia Detector 

Aim of this project is to detect cases of pneumonia based on dataset Chest X-Ray Images (https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).  

Dataset consists of two categories (NORMAL, PNEUMONIA) and 5863 images.  

We prepared two Tensorflow models to make predictions - one trained from scratch, second based on Inception V3.  
We also created web application to upload image and make prediction.  
It consists of two services - backend (written using FastAPI framework) to make upload images, serve static files and make call to Tensorflow Serving service and Tensorflow Serving service to serve unified REST API for both models.  

To run this project as web application you need to have Docker engine installed (https://docs.docker.com/engine/install/) as well as Docker Compose.  

After instalation run these commands:   
``` docker-compose build ```   

``` docker-compose up ```

Application should be available at http://localhost:8000 (web application) and http://localhost:5801 - REST API for Tensorflow Serving.  

Due to warm-up of Tensorflow models first call to make prediction might take more time.  
In order to get correct preview image click 'Reset' button after each prediction.  

Additionaly we have notebook service to run Jupyter Notebook as a Docker container. Provided service has Tensorflow installed, but it should not be used to train models as it lacks GPU support. It is only for demonstrational purposes.

To run with this extra service use this command:  

``` docker-compose -f docker-compose-with-jupyter.yml up ```   

Notebook is available at http://localhost:8888, token will be available in output of this service  