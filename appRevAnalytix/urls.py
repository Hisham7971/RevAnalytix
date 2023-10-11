from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('chat.html/', views.chat_page, name='chat_page'),
    path('basic-table.html/', views.basic_table, name='basic_table'),
    path('datatable.html/', views.datatable, name='datatable'),
    path('highchart.html/', views.highchart, name='highchart'),
    path('knob-chart.html/', views.knob_chart, name='knob_chart'),
    path('apexcharts.html/', views.apexcharts, name='apexcharts'),
    path('search_reviews/', views.search_and_display_reviews, name='search_reviews'),
    path('sentiment-analysis/', views.sentiment_analysis_view, name='sentiment-analysis')
]