from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Home page with chatbot interface
    path('api/respond/', views.chat_response, name='chat_response'),  # API endpoint for chatbot
]
