# -*- coding: utf-8 -*-
#! /usr/bin/env python
"""
Created on Fri June 10 14:01:52 2016

@author: Troy
"""

from numpy import *
from model_tools import *

def loadDataSet(filename):                       #定义读取数据函数,输出训练数据和测试数据
    fr=open(filename)
    num_series=len(open(filename).readlines())   #文件中时间序列数目
    length_series=float(len(open(filename).readline().strip().split(',')))
    input_data=[]                               #将数据存入数组中
    for line in fr.readlines():
        data=[]                                 #存取每个时间序列数据
        data=line.strip().split(',')
        data=[float(data[i]) for i in range(len(data))]                              
        input_data.append(data)
    Data=mat(input_data)                        #将输入数据转化成矩阵便于后续处理
    train_data=mat(Data[:int(num_series*3/float(4)),:])     #训练数据  占总样本10/12 
    #print length_series*10/12
    #print length_series
    #print num_series
    test_data=mat(Data[int(num_series*3/float(4)):,:])       #测试数据  占总样本2/12
    return train_data, test_data          #默认训练和输出数据为一个时间序列中样本       
 
#particle=0   

class FTSFM:                              #定义预测类，文本输入为一个时间序列处理
    def __init__(self,filename,The_order_of_series):
        self.train_data_initial,self.test_data_initial=loadDataSet(filename)
        self.train_data=self.train_data_initial.T[The_order_of_series,:]
        self.test_data=self.test_data_initial.T[The_order_of_series,:]   #此时为矩阵的某一行
        self.seasonality_factor=0
        
    
    def fts_training(self):
        train_data,test_data,self.seasonality_factor=seaonality_remove(self.train_data,self.test_data,15)
        train_data,test_data=mat(train_data),mat(test_data)   #将数据去除周期性
        
        
        #train_data=self.train_data
        #test_data=self.test_data
        #train_data=[train_data[0,i] for i in range(shape(train_data)[1])]
        #test_data=[test_data[0,i] for i in range(shape(test_data)[1])]
        num_particles=10               #particle=[smoothing_factor,waverate,forecast_w]   粒子构成
        particles=mat(random.rand(num_particles,3)*array([1.0, 1.5, 1.5])+array([0.8,0,0]))  #初始化粒子（模型参数）
        max_iteration=200                        #粒子群算法迭代上限
        pb_position=particles; pb_min_RMSE=[100000 for i in range(num_particles)]    #初始化局部最优位置
        gb_position=mat([1,1,1]); gb_min_RMSE=100000                                    #初始化全局最优位置
        vt=mat(random.rand(num_particles,3))       #初始化粒子速度
        
        for i in range(max_iteration):            #粒子群算法启动寻找最优参数设置（最优粒子）
            numble_particle=0          
            for particle in array(particles):
                interval=interval_divide(train_data,particle[0])   #划分论域
                fuzzy_data=fuzzier(train_data,interval)            #模糊历史数据
                FLRG_1=FLRG_Con(fuzzy_data,1)                         #构建一阶至三阶模糊逻辑关系组
                FLRG_2=FLRG_Con(fuzzy_data,2)
                FLRG_3=FLRG_Con(fuzzy_data,3)
                
                FLRG_r(FLRG_1,FLRG_total(FLRG_1))       #去除一阶至三阶FLRG中的噪声关系
                FLRG_r(FLRG_2,FLRG_total(FLRG_2))
                FLRG_r(FLRG_3,FLRG_total(FLRG_3))
                
                prediction_value=[0 for j in range(shape(train_data)[1]-3)]  #预测值为一个列表
                for numble in range(shape(train_data)[1]-3):
                    prediction_1=Prediction(fuzzy_data[numble:numble+3],array(train_data)[0][numble:numble+3],interval,FLRG_1,particle[1])
                    prediction_2=Prediction(fuzzy_data[numble:numble+3],array(train_data)[0][numble:numble+3],interval,FLRG_2,particle[1])
                    prediction_3=Prediction(fuzzy_data[numble:numble+3],array(train_data)[0][numble:numble+3],interval,FLRG_3,particle[1])
                    #分别计算一二三阶预测值，并合并成最终预测值
                    prediction_value[numble]=Prediction_merge(prediction_1,prediction_2,prediction_3,particle[2])
                current_RMSE=RMSE(prediction_value,array(train_data)[0][3:])    #计算本轮该粒子的RMSE  输入为一个列表 和一个一维数组
                if current_RMSE < pb_min_RMSE[numble_particle]:
                    pb_position[numble_particle]=particle
                    pb_min_RMSE[numble_particle]=current_RMSE
                numble_particle+=1
            if min(pb_min_RMSE)<gb_min_RMSE:
                for h in range(shape(pb_min_RMSE)[0]):
                    if pb_min_RMSE[h]==min(pb_min_RMSE):
                        min_index=h
                        break
                gb_position=pb_position[min_index]
                gb_min_RMSE=min(pb_min_RMSE)
            particles,vt=PSO(particles,vt,pb_position,gb_position)
            #print (i)
        return gb_position
        
    def fts_forecasting(self,particle):    #运用训练最优粒子进行预测
        train_data,test_data,self.seasonality_factor=seaonality_remove(self.train_data,self.test_data,15)
        train_data,test_data=mat(train_data),mat(test_data)   #将数据去除周期性
    
    
        #train_data=self.train_data;test_data=self.test_data
        particle=array(particle)
        interval=interval_divide(train_data,particle[0][0])   #划分论域
        fuzzy_data=fuzzier(train_data,interval)            #模糊历史数据
        fuzzy_forecast_data=fuzzier(test_data,interval)    #模糊预测数据
        FLRG_1=FLRG_Con(fuzzy_data,1)                         #构建一阶至三阶模糊逻辑关系组
        FLRG_2=FLRG_Con(fuzzy_data,2)
        FLRG_3=FLRG_Con(fuzzy_data,3)
                
        FLRG_r(FLRG_1,FLRG_total(FLRG_1))       #去除一阶至三阶FLRG中的噪声关系
        FLRG_r(FLRG_2,FLRG_total(FLRG_2))
        FLRG_r(FLRG_3,FLRG_total(FLRG_3))
                
        prediction_value=[0 for j in range(shape(test_data)[1]-3)]
        for numble in range(shape(test_data)[1]-3):     #预测未来值
            prediction_1=Prediction(fuzzy_forecast_data[numble:numble+3],array(test_data)[0][numble:numble+3],interval,FLRG_1,particle[0][1])
            prediction_2=Prediction(fuzzy_forecast_data[numble:numble+3],array(test_data)[0][numble:numble+3],interval,FLRG_2,particle[0][1])
            prediction_3=Prediction(fuzzy_forecast_data[numble:numble+3],array(test_data)[0][numble:numble+3],interval,FLRG_3,particle[0][1])
            #分别计算一二三阶预测值，并合并成最终预测值
            prediction_value[numble]=Prediction_merge(prediction_1,prediction_2,prediction_3,particle[0][2])
            #current_RMSE=RMSE(prediction_value,array(test_data)[0][3:])    #计算预测误差RMSE
        for i in range(3):    #为了季节性讲一个预测周期类的缺失前三点补全
            prediction_value.insert(0,0)
        prediction_value=seasonality_recover(prediction_value,self.seasonality_factor)    #将预测后的序列恢复季节性
        del prediction_value[0:3]   #补上周期性之后再去掉补全点
        return RMSE(prediction_value,array(self.test_data)[0][3:]) , RMSE_percent(prediction_value,array(self.test_data)[0][3:]),RMSE_percent_2(prediction_value,array(self.test_data)[0][3:])
                
            
  #The_order_of_series=1 选取第几个时间序列进行分析预测
     
if __name__=='__main__':     #主函数预测，也可以调用模块预测
    for The_order_of_series in range(10):
        fuzzy_time_series_model = FTSFM('TOP10_no_cellnumble',The_order_of_series)
        particle=fuzzy_time_series_model.fts_training()
        #print ('The model parameter are %r'%(particle))
        rmse,rmse_percent_1,rmse_percent_2=fuzzy_time_series_model.fts_forecasting(particle)  #后两个精度值分别代表两种不同方式计算的预测精度值
    
    
        #print ('The RMSE of forecasting is %f' %(RMSE))
        print ('%d th cell of Li and Ren are %f and %f' %((The_order_of_series+1),rmse_percent_1,rmse_percent_2))
        del fuzzy_time_series_model,particle,rmse,rmse_percent_1,rmse_percent_2