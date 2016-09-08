# -*- coding: utf-8 -*-
"""
Created on Fri June 10 19:49:30 2016

@author: Troy
"""

from numpy import *


def interval_divide(data_series,smoothing_factor):      #模型提出的论域区间划分函数 输入为历史数值数据,和模型论域划分参数值
    mean_value=mean(data_series)      #历史数据均值
    std_value=std(data_series)/2        #历史数据标准差的一半
    interval=[]
    interval.append(mean_value)
    while interval[-1]<max(array(data_series)[0]):                  #初始化论域划分 已均值为基点上下扩充等差论域区间长度为std_value
        interval.append((interval[-1]+std_value))
    while interval[0]>min(array(data_series)[0]):
        interval.insert(0,(interval[0]-std_value))   
    n=[0 for i in range(len(interval)-1)]        #n列表为统计各区间内历史数据的样本点数 
    data_series=[data_series[0,i] for i in range(shape(data_series)[1])]
    #print data_series
    for i in data_series:
        for j in range(len(interval)-1):
            if (i>=interval[j]) and (i<interval[j+1]):
                n[j]+=1
                break
            if j==len(interval)-2:
                n[-1]+=1
                break
    mean_n=mean(n)     #初始区间平均样本点数
    
    while 1:                                              #将所含历史样本点数大于初始均值的区间等分为两个子区间，直到不存在这样的区间
        count=0
        for i in range(len(n)):
            if n[i]>(int(smoothing_factor*mean_n)):
                interval.insert(i+1,(interval[i]+interval[i+1])/2)
                count+=1
        if count==0:
            break
        n=[0 for i in range(len(interval)-1)]
        for i in data_series:
            for j in range(len(interval)-1):
                if (i>=interval[j]) and (i<interval[j+1]):
                    n[j]+=1
                    break
                if j==len(interval)-2:
                    n[-1]+=1
                    break
    return interval                   #返回最终论域划分



def fuzzier(data_series,interval):    #模糊历史数据函数  输入为历史真实数据 和论域区间
    data_series=[data_series[0,i] for i in range(shape(data_series)[1])]
    fuzzy_data=[0 for i in range(len(data_series))]    #初始化模糊化后序列
    for i in range(len(data_series)):
        for j in range(len(interval)-1):
            if data_series[i]>=interval[j] and data_series[i]<interval[j+1]:
                fuzzy_data[i]=j+1
                break
            if j==(len(interval)-2):
                fuzzy_data[i]==j+1
                break
    return fuzzy_data




def FLRG_Con(fuzzy_data,n): #构建模糊逻辑关系组，输入：模糊数据，模糊逻辑关系阶数n
    FLRG=[[0 for i in range(n+2)]]    #初始化模糊逻辑关系组FLRG为一行全零值  
    length_data=len(fuzzy_data)   
    for i in range(length_data-n):
        for line in FLRG:
            if  line[:n+1]==fuzzy_data[i:i+n+1]:    #若模糊逻辑关系组中已存在该关系则计数加一
                line[-1]+=1
                break
            if line==FLRG[-1]:
                new=[]; new.append(fuzzy_data[i:i+n+1]); new.append(1)     
                FLRG.append(new)                             #若不存在则扩充如模糊逻辑关系组中
                break
    del(FLRG[0])                                   #删除FLRG中初始全零行
    return FLRG
   
   
def FLRG_total(FLRG):   #汇总相同左侧模糊逻辑关系的总数 将其储存在最右一列
    FLRG_sum=[[0 for i in range(shape(FLRG)[1]-1)]]
    num_order=shape(FLRG)[1]-2     #求出FLRG中所含阶数
    for line in FLRG:             #记录相同左侧模糊逻辑关系总数，组内不包含的则扩充入组内
        for j in FLRG_sum:
            if line[:num_order]==j[:num_order]:
                j[num_order]+=line[num_order+1]
                break
            if j==FLRG_sum[-1]:
                new=[]; new.append(line[:num_oder]); new.append(line[-1])
                FLRG_sum.append(new)
                break
    del(FLRG_sum[0])
    return FLRG_sum
        
                
   
   

def FLRG_r(FLRG,FLRG_sum):     #去除FLRG中噪声关系  标准为少于该类左侧模糊关系总数的0.05倍的则删除该模糊关系
    num_order=shape(FLRG)[1]-2   
    for line in range(shape(FLRG)[0]):
        for i in FLRG_sum:
            if FLRG[line][:num_order]==i[:num_order]:
                if line[-1]<0.05*i[-1]:
                    del(FLRG[line])
                break
    return
    

def Prediction(fuzzy_data,current_data,interval,FLRG,waverate):     #基于第i阶FLRG进行预测,输入为预测前模糊值，预测前若干时刻真实值，区间划分，FLRG，趋势参数
    num_order=shape(FLRG)[1]-2                       
    count=0; value=0                                             #记录预测值，预测为趋势性预测
    for line in FLRG:
        if fuzzy_data[-num_order:]==line[:num_order]:
            if line[-2]>line[-3]:
                value+=(current_data[-1]+float(waverate)*(interval[line[-3]]-interval[line[-3]-1])/2)*line[-1]
                count+=line[-1]
            elif line[-2]==line[-3]:
                value+=current_data[-1]*float(line[-1])
                count+=line[-1]
            else:
                value+=(current_data[-1]-float(waverate)*(interval[line[-3]]-interval[line[-3]-1])/2)*line[-1]
                count+=line[-1]
    if count>0:
        prediction_value=value/count
        return prediction_value, 1       #返回1代表匹配预测，返回0代表没有匹配预测
    else:
        prediction_value=(current_data[-1]*3+current_data[-2]*2+current_data[-3]*1)/float(6)
        return prediction_value, 0
        
def Prediction_merge(prediction_1,prediction_2,prediction_3,forecast_w):    #合并各阶预测值，输入为各阶预测值以及非匹配预测时合并权值
    weight_1=1;weight_2=2;weight_3=3
    if prediction_3[-1]==0:      #减少判断次数 根据数据特性而设置
        if prediction_1[-1]==0:
            weight_2,weight_3=0,0
        elif prediction_2[-1]==0:
            weight_2,weight_3=forecast_w,0
        elif prediction_3[-1]==0:
            weight_3==forecast_w
    prediction=(prediction_1[0]*weight_1+prediction_2[0]*weight_2+prediction_3[0]*weight_3)/float(weight_1+weight_2+weight_3)
    return prediction
    
    
def  RMSE(prediction_series,real_value):     #误差函数，计算军方误差RMSE，输入为预测序列和真实序列
    num=len(prediction_series)
    value=0      #储存累计误差
    for i in range(num):
        value+=(prediction_series[i]-real_value[i])**2
    rmse=(value/float(num))**0.5
    return rmse
    
def RMSE_percent(prediction_series,real_value):   #误差函数，计算平均误差百分比
    num=len(prediction_series)
    value=0
    for i in range(num):
        value+=abs(prediction_series[i]-real_value[i])/real_value[i]
    rmse=value/float(num)
    return 1-rmse
    
def RMSE_percent_2(prediction_series,real_value):
    num=len(prediction_series)
    value=0    #记录各个时刻预测误差总和
    total=0    #记录各个时刻真实值总和
    for i in range(num):
        value+=abs(prediction_series[i]-real_value[i])
        total+=real_value[i]
    return 1-float(value/total)
        
def seaonality_remove(real,test,seasonality_value):    #去除季节性函数  输入为真实值序列和 周期值
    real_value,test_value=array(real),array(test)    
    all_average=0      #记录所有点的平均值
    single_average=[0 for i in range(seasonality_value)]   #记录单个点的平均  最终做为季节因子返回
    num_period=int((shape(real_value)[1])/seasonality_value)   #周期数
    for i in range(len(real_value)):
        all_average+=real_value[0][i]
    all_average=all_average/float(len(real_value))
    for i in range(len(single_average)):
        for j in range(num_period):
            single_average[i]+=real_value[0][i+j*seasonality_value]
        single_average[i]/=float(num_period)
        single_average[i]/=all_average    #此时得到的为点的季节因子
    
    real_value_remove_seasonality_train=[(real_value[0][i])/float(single_average[int(i%seasonality_value)])  for i in range(shape(real_value)[1])]
    real_value_remove_seasonality_test=[(test_value[0][i])/float(single_average[int(i%seasonality_value)])  for i in range(shape(test_value)[1])]
    return  real_value_remove_seasonality_train,real_value_remove_seasonality_test,single_average
    
    
    
    
def PSO(particles,vt,pb_position,gb_position):   #粒子群算法迭代函数   （矩阵输入） 输入为当前例子，局部最优位置，全局最优位置  (矩阵)
    vt_1=0.729*(array(vt)+2.05*(array(pb_position)-array(particles))*array(random.rand(shape(particles)[0],3))+2.05*(array(gb_position)-array(particles))*array(random.rand(shape(particles)[0],3)))
    particles_1=array(particles+mat(vt_1))    #更新后的粒子
    for line in particles_1:           #调整出界粒子
        if line[0]<0.8 or line[0]>1.8:
            line[0]=0.8+random.rand()*1
        if line[1]<0 or line[1]>1.5:
            line[1]=random.rand()*1.5
        if line[2]<0 or line[2]>1.5:
            line[2]=random.rand()*1.5
    return mat(particles_1), mat(vt_1)




            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
