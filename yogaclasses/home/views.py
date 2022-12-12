from django.shortcuts import render, HttpResponse, redirect
from django.template import loader
from datetime import datetime
from home.models import Form

# Create your views here.
def index(request):
    mydata = Form.objects.all().values()
    context = {
        'mymembers': mydata,
    }
    template = loader.get_template('index.html')
    return HttpResponse(template.render(context, request))

def form(request):
    if request.method == "POST":
        name = request.POST.get('name')
        age = request.POST.get('age')
        phone = request.POST.get('phone')
        address = request.POST.get('address')
        batch = request.POST.get('batch')
        # payment_status = request.POST.get('payment_status')
        
        form = Form(name = name, age = age, phone = phone, address = address, batch = batch, payment_status = True, date = datetime.today())
        form.save()
        return redirect('home')
    return render(request, 'form.html')