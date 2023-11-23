import pickle
import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Create an instance of FastAPI
app = FastAPI(
    title="California Housing Price Prediction", 
    version="v1.0.0"
)

class data_california_house(BaseModel):
    longitude : float
    latitude : float
    housing_median_age : float
    total_rooms : float
    total_bedrooms : float
    population : float
    households : float
    median_income : float
    ocean_proximity : object
    total_rooms_per_household : float
    total_bedrooms_per_household : float

# Define a Python class to create a list to reformat the data
class Item(BaseModel):
    data: List[data_california_house]

# Loading the saved model
model = pickle.load(open('../model/final_model.sav', 'rb'))

# Create a POST endpoint to make prediction
@app.post('/prediction')
async def houseprice_prediction(parameters: Item):
    # Get inputs
    req = parameters.dict()['data']

    # Convert input into Pandas DataFrame
    data = pd.DataFrame(req)

    # Make the predictions
    res = model.predict(data).tolist()
    
    return {"Response": res}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)