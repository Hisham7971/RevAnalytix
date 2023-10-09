from django.shortcuts import render

def index(request):
    # Your logic here if needed
    return render(request, 'index.html')

def chat_page(request):
    # Your logic here if needed
    return render(request, 'chat.html')

def basic_table(request):
    # Your logic here if needed
    return render(request, 'basic-table.html')

def datatable(request):
    # Your logic here if needed
    return render(request, 'datatable.html')

def highchart(request):
    # Your logic here if needed
    return render(request, 'highchart.html')

def knob_chart(request):
    # Your logic here if needed
    return render(request, 'knob-chart.html')

def apexcharts(request):
    # Your logic here if needed
    return render(request, 'apexcharts.html')