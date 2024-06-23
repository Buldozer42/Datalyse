
# Data mining
This third year project aims to create a data mining application using the Streamlit library. The application will allow users to upload a dataset, visualize it, and apply different algorithms to it. The application will also allow users to visualize the results of the algorithms applied to the dataset.

# Table of contents  
1. [Features](#features)  
2. [Run Locally](#run-locally)  
3. [Authors](#authors)
   
## Features

- Exploration and cleaning
- Visualization
- Normalization
- Prediction models
- PCA (Principal Component Analysis)
- Clustering

## Run Online

The application is available online at the following address: [Datalyse](https://datalyse.streamlit.app/)

## Run Locally  

Clone the project  

~~~bash  
  git clone https://codefirst.iut.uca.fr/git/jeremy.besson/Datalyse.git
~~~

Go to the project directory  

~~~bash  
  cd Datalyse/
~~~


Create a virtual environment  

```bash
python -m venv .venv
```

Run the virtual environment 

```bash
# Windows
.venv\Scripts\activate.bat

# Powershell
.venv\Scripts\Activate.ps1

# macOS and Linux
source .venv/bin/activate
```

Add the requirements to the virtual environment  

```bash
pip install -r requirements.txt
```

If you want to see current installed packages  

```bash
pip freeze
```

Start the application  

```bash
python -m streamlit run .\datalyseApp\app.py
```

## Authors  
- [Noé GARNIER](https://codefirst.iut.uca.fr/git/noe.garnier)
- [Jérémy BESSON](https://codefirst.iut.uca.fr/git/jeremy.besson)
- [Matis MAZINGUE](https://codefirst.iut.uca.fr/git/matis.mazingue)
