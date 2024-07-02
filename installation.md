# Installation
``` bash
virtualenv env 
source env/bin/activate
pip install -r requirements.txt
```
# Usage
## To run your application on your local system : 
``` bash
export FLASK_APP=app.py
python -m flask run
```

## To run your application with camera enabled on your phone : (Provide Hotspot to the main server)
``` bash
export FLASK_APP=app.py
python -m flask run --host=0.0.0.0
```