from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel



model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')


student_app = FastAPI()

class Student(BaseModel):
    gender: str
    race_ethnicity: str
    parent: str
    lunch: str
    test: str
    math_score: float
    reading_score: float


@student_app.post('/predict')
async def check_score(student: Student):
   student_dict = student.dict()

   new_gender = student_dict.pop('gender')
   gender1_0 = [1 if new_gender == 'male' else 0]

   new_race_ethnicity = student_dict.pop('race_ethnicity')
   race_ethnicity1_0 = [
       1 if new_race_ethnicity == 'group B' else 0,
       1 if new_race_ethnicity == 'group C' else 0,
       1 if new_race_ethnicity == 'group D' else 0,
       1 if new_race_ethnicity == 'group E' else 0
  ]

   new_parent = student_dict.pop('parent')
   parent1_0 = [
       1 if new_parent == "bachelor's degree" else 0,
       1 if new_parent == "education_high school" else 0,
       1 if new_parent == "education_master's degree" else 0,
       1 if new_parent == "education_some college" else 0,
       1 if new_parent == "education_some high school" else 0
   ]

   new_lunch = student_dict.pop('lunch')
   lunch1_0 = [
       1 if new_lunch == 'standard' else 0
   ]

   new_test = student_dict.pop('test')
   test1_0 = [
       1 if new_test == 'none' else 0
   ]
   fetures = list(student_dict.values()) + gender1_0 + race_ethnicity1_0 + parent1_0 + lunch1_0 + test1_0
   scaled = scaler.transform([fetures])
   pred = model.predict(scaled)[0]
   return {'Примерный балл по writing score': round(pred, 2)}




if __name__ == '__main__':
    uvicorn.run(student_app, host='127.0.0.1', port=8000)