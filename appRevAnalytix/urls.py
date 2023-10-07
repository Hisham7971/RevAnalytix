from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name="index"),
    path('chat.html/', views.chat_page, name='chat_page'),
    path('basic-table.html/', views.analyze_sentiment_and_priority, name='analyze_sentiment_and_priority'),
]