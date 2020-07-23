# Thesis

## "Personalized E-commerce Banner Recommendation using Machine Learning Algorithms"

Intelligent Systems & Software Engineering Labgroup (Electrical & Computer Engineering Dept AUTH)

### Abstract
The  rapid  technological  development  of  recent  years,  the  improvement  of  computer systems,  and  the  familiarization  of  a  large  percentage  of  the  world's  population  with  the digital  world  have  given  an  enormous  boost  to  e-commerce,  which  is  continually  evolving and  serving  more  needs.  Simultaneously,  the  significant  increase  of  users  and  products, coming  as  a  result  of  this  progress,  and  the  dynamic  entry  of  machine  learning  and  data science in the field of information technology has allowed e-commerce sites to improve the browsing   experience   significantly.   Nowadays,   e-commerce   sites   provide   users   with personalized product suggestions that meet their preferences, which means a simultaneous increase in sales for online stores. In  addition  to  personalized  directproduct  recommendations  to  consumers,  there  are  also advertising views (or banners). They are quite common on e-commerce websites, aiming to help and promote consumer product groups to the consumer according to his preferences or  by  categorizing  him  according  to  key  elements  of  his  electronic  imprint.  Personalized banner   recommendations   have   not   been   studied   to   the   same   degree   as   product personalization and are more applicable to large e-commerce platforms.This dissertation aims to design and build a real-time personalized banner recommendation system  for  a  medium-sized  online  e-shop  with  real-time  data  based  on  machine  learning methods  and  algorithms.  In  the  context  of  the  work,  we  propose  a  novel  framework  that takes into account the actions of the usersduring their navigation, known as "clickstream" data.  The  proposed  framework  effectively  recognizes  user  interests  and  suggests  banners that correspond to their preferences.

### Repository description
This repository contains the code for the implementation and evaluation of the real-time banner recommendation system we built.

### Contents
* **DeepMF_Keras**: A Deep Matrix Factorization approach for product recommendation using Keras library
* **GRU4REC_Tensorflow**: A Tensorflow implementation of *GRU4REC* algorithm, which was descibed in "Session-based Recommendations With Recurrent Neural Networks". See paper: http://arxiv.org/abs/1511.06939. 
* **banner_recsys_API**: A RESTful API developed in Django to communicate with e-commerce e-shop in order to receive, preprocess and train data with integrated ML models and sent back Top-N banner recommendations proposal  
* **testing_results**: Offline evaluation of the system through CTR prediction




