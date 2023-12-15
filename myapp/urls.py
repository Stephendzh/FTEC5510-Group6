from . import views
from django.urls import path


urlpatterns = [
    path('page/login/', views.page_login.as_view(), name='page_login'),
    path('page/profile/', views.page_profile, name='page_profile'),
    path('page/signup/', views.page_signup.as_view(), name='page_signup'),
    path('user/exist/', views.user_exist, name="user_exist"),
    path('info/wrong/',views.info_wrong, name="info_wrong"),
    path('show_table/', views.show_table),
    path('show_knear/', views.show_knear),
    path('show_log/', views.show_log),
    path('show_svc/', views.show_svc),
    path('show_tree/', views.show_tree),
    path('show_gcn/', views.GCN),
]