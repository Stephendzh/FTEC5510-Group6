from django.shortcuts import render

# Create your views here.

from django.shortcuts import render,redirect
from . import models
from django.views import View
import os
# Create your views here.

class page_signup(View):
    def get(self, request):
        return render(request, "page-signup(1).html")

    def post(self, request):
        user_email = request.POST.get("user_email")
        user_password = request.POST.get("user_password")
        # 注册成功跳转到到登录页面，注册加判断已经存在提示改用用户已存在
        users = models.userinfo.objects.all()
        for i in users:
            if user_email == i.Email:
                return redirect('http://127.0.0.1:8000/user/exist/')

        models.userinfo.objects.create(Email=user_email, Password=user_password)
        return redirect('http://127.0.0.1:8000/page/login/')

def user_exist(request):
    return render(request, "user-exist(1).html")


class page_login(View):
    def get(self, request):
        return render(request, "page-login(1).html")

    def post(self, request):
        login_email = request.POST.get("login_email")
        login_password = request.POST.get("login_password")
        if models.userinfo.objects.filter(Email=login_email).exists()==True:
            user=models.userinfo.objects.filter(Email=login_email)[0]
            if user.Email==login_email and user.Password==login_password:
                return redirect('http://127.0.0.1:8000/page/profile/')
        return redirect('http://127.0.0.1:8000/info/wrong')

def info_wrong(request):
    return render(request, "info-wrong(1).html")

def page_profile(request):
    return render(request,"index.html")


# 业务part

def show_table(request):
    if request.method == "GET":
        message1 = "This is the message, which will be changed later" #os.popen('python D:\\courses23fall\\computational_finance\\a2\\1.py').read()
        print("prediction:", message1)
    return render(request, 'tables.html', {"message1" : "33333"})

def show_log(request):
    message = os.popen('python D:\\courses23fall\\computational_finance\\a2\\1.py').read()
    return render(request, "logistic_reg.html", context={"log_reg" : message})

def show_knear(request):
    knear_reg = os.popen('python D:\\courses23fall\\computational_finance\\a2\\knear.py').read()
    return render(request, 'panels.html', context={"knear_reg" : knear_reg})

def show_tree(request):
    tree_class = os.popen('python D:\\courses23fall\\computational_finance\\a2\\tree.py').read()
    return render(request, 'tree_class.html', context={"tree_class" : tree_class})

def show_svc(request):
    message_svc = os.popen('python D:\\courses23fall\\computational_finance\\a2\\svc.py').read()
    return render(request, 'svc_class.html', context={"svc" : message_svc})

def GCN(request):
    message_gcn = os.popen('python D:\\courses23fall\\computational_finance\\a2\\gcn.py').read()
    print(message_gcn)
    return render(request, 'gcn.html', context={"message_gcn" : message_gcn})
