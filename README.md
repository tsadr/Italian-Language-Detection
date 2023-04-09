# Italian-Language-Detection
Deep Learning Model for Italian Language Detection.

We have developed a deep learning model using Keras in Python to detect the Italian language based on the dataset available at https://www.kaggle.com/datasets/basilb2s/language-detection.

We have also provided a RESTful API for model inference. To run the code, first, you need to install the required packages for FastAPI, and then run the following command:

```
uvicorn app.main:app --port 5000
```

This will start the server on port 5000.
