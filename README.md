# pythonImage2VecApi
pythonI mage to Vector Api for Elastic ImageSearch


## run
python3 -m venv venv //파이썬 가상환경 만들기   
source ./venv/bin/activate //파이썬 가상환경 활성화   
pip install fastapi //fastapi 설치   
pip install uvicorn //uvicorn 설치   
pip install img2vec_pytorch//image2vec_pytorch설치   
pip install tensorflow   
pip install keras    


uvicorn main:app --reload --host=0.0.0.0 --port=29880//29880포트에서 fastapi실행   
