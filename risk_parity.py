#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/7/22 21:35
# @Author : Qian
# @Site : 
# @File : risk_parity.py
# @Software: PyCharm

from jqdatasdk import *
import datetime
import pandas as pd
import numpy as np
import scipy.stats as scs
import statsmodels as sm
from scipy.optimize import minimize

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题


id='17713586340'
password = 'Aa8293088'
auth(id,password)


def get_data(zl, index_list, start_date, end_date, code, count, fq):
    # fq:pre,None
    if zl == 'home':
        close = get_price(index_list, start_date, end_date, fields='close', frequency='daily', fq=fq)
        return close['close']
    elif zl == 'global':
        q = query(finance.GLOBAL_IDX_DAILY).filter(finance.GLOBAL_IDX_DAILY.code == code).order_by(
            finance.GLOBAL_IDX_DAILY.day.desc()).limit(count)
        df = finance.run_query(q)
        df = df[(df['day'] >= start_date) & (df['day'] <= end_date)].sort_values('day')
        df = df.set_index('day')['close']
        df.name = code
        return df
    elif zl == 'huangjin':
        q = query(finance.FUT_GLOBAL_DAILY).filter(finance.FUT_GLOBAL_DAILY.code == code).order_by(
            finance.FUT_GLOBAL_DAILY.day.desc()).limit(count)
        df = finance.run_query(q)

        df = df[(df['day'] >= start_date) & (df['day'] <= end_date)].sort_values('day')
        df = df.set_index('day')['close']
        df.name = code
        return df

class RiskParity:
    #待解决问题1：协方差的估计方法
    #待解决问题2：将其他资产配置模型纳入到类中，如最小方差、马科维茨、60/40组合、等权组合、风险贡献模型等
    #思考：加杠杆怎么做？
    #思考：如何根据市场情况及时更新，并反馈到我的微信中


    def __init__(self,**kwargs):
        self.ts_step=kwargs['ts_step'] if 'ts_step' in kwargs else None
        self.ob_step = kwargs['ob_step'] if 'ob_step' in kwargs else None
        self.start_time = kwargs['start_time'] if 'start_time' in kwargs else None
        self.end_time = kwargs['end_time'] if 'end_time' in kwargs else None
        self.price = kwargs['price'] if 'price' in kwargs else None
        self.trade_date = self.get_trade_day()[0]
        self.ob_date = self.get_trade_day()[1]
        self.n = kwargs['n'] if 'n' in kwargs else None
        self.initial_money = kwargs['initial_money'] if 'initial_money' in kwargs else None
        self.year = kwargs['year'] if 'year' in kwargs else None


    def get_trade_day(self):
        #比较粗糙的做法，还需要精细处理：如国内交易日不一定也是国外交易日
        price = self.price.copy().reset_index()
        date = price['index']
        step = self.ts_step
        ob_step = self.ob_step
        start = price[price['index']==self.start_time].index[0]
        end = price[price['index']==self.end_time].index[0]
        inx = [i for i in range(start,end,step)]
        trade_date = date[inx]
        ob_date = date[[i-ob_step for i in inx]]
        return trade_date,ob_date

    def calculate_port_var(self,w,cov):
        #计算组合的总风险
        w = np.asmatrix(w)
        risk = (w*cov*w.T)[0,0]
        return risk

    def calculate_risk_contribution(self,w,cov):
        w = np.asmatrix(w)
        #边际风险贡献
        sigma = np.sqrt(self.calculate_port_var(w,cov))
        MRC = cov*w.T
        RC = np.multiply(MRC,w.T)/sigma
        return RC

    def risk_objective(self,w,*args):
        w = np.asmatrix(w)
        cov = args[0]
        assest_RC = self.calculate_risk_contribution(w,cov).A1
        J = sum([sum((i-assest_RC)**2) for i in assest_RC])
        return J

    def sharp_objective(self,w,*args):
        pass

    def var_objective(self,w,*args):
        pass



    def calculate_w(self,cov):
        n = self.n
        x_t = np.array([1. / n] * n)
        bnds = tuple((0, 1) for i in range(len(x_t)))
        #模拟退火算法求初始解
        '''
        a = _dual_annealing.dual_annealing(risk_objective, bounds=bnds, args=(cov, x_t), maxiter=1000, local_search_options={},
                                       initial_temp=5230.0, restart_temp_ratio=2e-05, visit=2.62, accept=-5.0,
                                       maxfun=10000000.0, seed=None, no_local_search=False, callback=None, x0=None)
        
        w0 =a.x
        '''
        np.random.seed(10)
        w0 = x_t
        cons = ({'type':'eq','fun':lambda w:np.sum(w)-1.0})
        options = {'disp':False,'maxiter':1000,'ftol':1e-20}
        res = minimize(self.risk_objective,w0,args=(cov),method='SLSQP',bounds= bnds,constraints = cons,options=options)
        w_rb = np.asmatrix(res.x)
        return w_rb

    def get_weights(self):
        ret = self.price.pct_change(1)
        trade_date = self.trade_date.tolist()
        ob_date = self.ob_date.tolist()
        weight = pd.DataFrame()
        contribution = pd.DataFrame()
        for i in range(len(trade_date)):
            cov = np.asmatrix(ret.loc[ob_date[i]:trade_date[i]].cov())
            w = self.calculate_w(cov).A1
            weight[trade_date[i]] = w.tolist()
            contribution[trade_date[i]] = self.calculate_risk_contribution(w,cov).A1.tolist()
            print("=========={}计算完毕===============".format(trade_date[i]))
        weight.index =self.price.columns
        contribution.index = self.price.columns
        return weight.T,contribution.T

    def backtest_new(self,method,weight):
        money=[self.initial_money]
        trade_date = self.trade_date.tolist()
        price = self.price.copy()
        if method == 'risk_parity':
            count = money * weight.loc[trade_date[0]] / (price.loc[trade_date[0]])
            res = []
            for i in range(1, len(trade_date)):
                price_list = price.loc[trade_date[i - 1]:trade_date[i]].iloc[:-1]
                money = (price_list.multiply(count)).sum(axis=1).tolist()
                price_daily = price.loc[trade_date[i]]
                count = money[-1] * weight.loc[trade_date[i]] / price_daily
                res.extend(money)
            res.append((count * price_daily).sum())
        elif method =='equal':
            weight_i = pd.Series(1./self.n,index = weight.columns)
            count = money * weight_i / (price.loc[trade_date[0]])
            res = []
            for i in range(1, len(trade_date)):
                price_list = price.loc[trade_date[i - 1]:trade_date[i]].iloc[:-1]
                money = (price_list.multiply(count)).sum(axis=1).tolist()
                price_daily = price.loc[trade_date[i]]
                count = money[-1] * weight_i / price_daily
                res.extend(money)
            res.append((count * price_daily).sum())
        elif method =='60/40':
            weight_i = pd.Series([0.15,0.15,0.2,0.2,0.15,0.15,0.0],index = weight.columns)
            count = money * weight_i / (price.loc[trade_date[0]])
            res = []
            for i in range(1, len(trade_date)):
                price_list = price.loc[trade_date[i - 1]:trade_date[i]].iloc[:-1]
                money = (price_list.multiply(count)).sum(axis=1).tolist()
                price_daily = price.loc[trade_date[i]]
                count = money[-1] * weight_i / price_daily
                res.extend(money)
            res.append((count * price_daily).sum())

        return res



    def cal_drawback(self,ret_list):
        draw_back = 0
        high = ret_list[0]
        for i in range(1,len(ret_list)):
            low = np.min(ret_list[i:])
            if (low-high)/high < draw_back:
                draw_back=(low-high)/high
            else:
                continue
        return draw_back



    def indicator(self,res):
        res_indicator = pd.DataFrame()
        ret = res.pct_change(1)
        res_indicator['ann_ret'] = res.apply(lambda x:x[-1]**(1/self.year)-1,axis=0)
        res_indicator['ann_vol'] = ret.apply(lambda x:x.std()*np.sqrt(250),axis=0)
        res_indicator['sharp'] = (res_indicator['ann_ret']-0.03)/res_indicator['ann_vol']
        res_indicator['drawback'] = ret.apply(lambda x:self.cal_drawback(x.tolist()),axis=0)
        return res_indicator





if __name__=='__main__':

    start_date = datetime.date(2009, 1, 1)
    end_date = datetime.date(2020, 7, 1)
    index_list = ['000300.XSHG', '000905.XSHG', '000012.XSHG', '000013.XSHG']
    index_name = ['沪深300', '中证500', '国债指数', '企债指数']
    df1 = get_data('home', index_list, start_date, end_date, None, None, 'pre')
    codes=['800000.XHKG','INX']
    df2 = pd.DataFrame(index = df1.index)
    for code in codes:
        df = get_data('global',None,start_date,end_date,code,3200,None)
        df2 = pd.merge(pd.DataFrame(df2),pd.DataFrame(df),right_index=True,left_index=True,how='outer')
    df3 = get_data('huangjin',None,start_date,end_date,'XAU',3200,None)

    df = pd.merge(pd.merge(df1,df2,left_index=True,right_index=True,how='outer'),pd.DataFrame(df3),left_index=True,right_index=True,how='outer')
    df_new = df.fillna(method = 'ffill')
    rp = RiskParity(ts_step=20,ob_step=20,initial_money=1000000.0,n=7,price=df_new,year = 10,start_time=pd.datetime(2010,1,4),end_time=pd.datetime(2020,1,2))
    weight,contribution = rp.get_weights()
    pnl_rp = rp.backtest_new('risk_parity',weight)

    trade_date=rp.trade_date.tolist()
    res = df_new.loc[trade_date[0]:trade_date[-1],:]
    res['rp'] = pnl_rp
    res['60/40'] = rp.backtest_new('60/40',weight)
    res['equal'] = rp.backtest_new('equal', weight)
    res = res.div(res.iloc[0])
    rp.indicator(res)
    #==============================================================================
    res = pd.DataFrame()
    for i in [20,120,250]:
        rp = RiskParity(ts_step=20, ob_step=i, initial_money=1000000.0, n=7, price=df_new, year=10,
                         start_time=pd.datetime(2010, 1, 4), end_time=pd.datetime(2020, 1, 2))
        weight, contribution = rp.get_weights()
        pnl_rp = rp.backtest_new('risk_parity', weight)
        trade_date = rp.trade_date.tolist()
        res['rp_{}'.format(i)] = pnl_rp







