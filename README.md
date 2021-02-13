# Pneumonia Detector 

To run this project as web application you need to have Docker engine installed (https://docs.docker.com/engine/install/) as well as Docker Compose.  

After instalation run these commands:   
``` docker-compose build ```   

``` docker-compose up ```

Application should be available at http://localhost:8000 (web application) and http://localhost:5801 - REST API for Tensorflow Serving.  

Due to warm-up of Tensorflow models first call to make prediction might take more time.  
In order to get correct preview image click 'Reset' button after each prediction.  
