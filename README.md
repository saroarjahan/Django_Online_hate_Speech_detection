# Django_Online_hate_Speech_detection
Download dataset from here, https://drive.google.com/uc?id=1iHw8GxVWLlavqQpXwY6zhwLTeYdnfdHU&export=download and add it to the project.

We have to create a new directory to install virtualenv . Browse that directory via command and run. separate virtual server for each project is best, it will not affect other project library.

pip install virtualenv
virtualenv .
Scripts\activate or scripts\activate
Python 3.7

Now we have to create another directory inside virtualenv, then we have to install django inside it. Browse that new directory via command line and install django inside it.

pipenv install django /pip install django
Copy this whole oproject inside django directory, browse the app and run python .\manage.py runserver

*This project need NLP library  which i will add a rquirement.txt file soon with version. But u can install yourself while running the project.
pandas
urllib
sklearn
inscriptis
urllib.request
urllib3
re
