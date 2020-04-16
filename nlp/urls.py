
from django.urls import path
from . import views
from django.contrib.auth.views import LoginView
from django.contrib.auth.models import User
from django.conf.urls import url
from django.contrib import admin

from .views import(
	aggression,
	attack,
	toxicity,
	allhate,

	)


urlpatterns = [
      path('', views.aggression, name='home'),
      path('attack/', views.attack, name='home'),
      path('toxicity/', views.toxicity, name='home'),
      path('allhatespeech/', views.allhate, name='allhatespeech'),
      
]