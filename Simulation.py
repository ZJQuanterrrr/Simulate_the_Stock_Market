# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 08:53:45 2023

@author: C.Z.J
"""
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

# set style
plt.style.use('seaborn')

class GSM:
    # Generate stock market
    
    def __init__(self, Per_num, per_In_cash, per_In_stock, market_stock_num, market_stock_In_price, time):
        
        self.In_person_Num = Per_num
        self.In_per_cash = float(per_In_cash)
        self.In_per_stock = float(per_In_stock)
        self.In_market_stock_Num = market_stock_num 
        self.In_market_stock_price = float(market_stock_In_price)
        self.trade_T = time 
                
    def Value(self):
            
            V = [] 
            v = self.In_market_stock_price
            V.append(v) 
            for d in range(self.trade_T + 100 - 1): 
                
                v = v * np.exp(random.normalvariate(mu=0, sigma=0.025)) 
                V.append(v) 
                
            return V
        
    # generate 100+n days of price/value (additional 100 days for investors applying trend strategy 
    def Price(self):
            
            # V = self.Value()
            # PTS = [] 
            # p = self.In_market_stock_price
            # PTS.append(p)
            # for d in range(self.trade_T + 100 - 1):
            #     if d == 0:
            #         p = p
            #     else:
            #         p = V[d] + random.normalvariate(mu=0, sigma=0.025)

            #     PTS.append(p)
            
            PTS = [] 
            p = self.In_market_stock_price
            PTS.append(p)
            for d in range(100 - 1): 
                
                p = p * np.exp(random.normalvariate(mu=0, sigma=0.025)) 
                PTS.append(p)
                
            return PTS
                
    def RF(self):
            
            Risk_flavor = [] 
            for per in range(self.In_person_Num):
                rf = np.random.uniform(0.25, 0.85) 
                Risk_flavor.append(rf)
                
            return Risk_flavor
            
                
    # organization investors (OI) flavor of price to value 
    def O_PTV(self):
            
            O_PtoV = [] 
            for per in range(int(float(self.In_person_Num)/3)): 
                d1 = np.random.uniform(0.2, 0.8)
                O_PtoV.append(d1)
                
            return O_PtoV
                
    # OI flavor of price based on trend
    def O_TBP(self):
            
            O_TbackP = [] 
            for per in range(int(float(self.In_person_Num)/3)): 
                d2 = np.random.uniform(0.2, 0.8)
                O_TbackP.append(d2)
                
            return O_TbackP
                
    # trend investor (TI) flavor of price based on trend
    def T_TBP(self):
            
            T_TbackP = [] 
            for per in range(int(float(self.In_person_Num)/3)): 
                d3 = np.random.uniform(-1.5, 1.5)
                T_TbackP.append(d3)
                
            return T_TbackP
                
    # OI sma flavor
    def O_sma_F(self):
            
            O_sma = []
            for per in range(int(float(self.In_person_Num)/3)):
                sma = np.random.randint(2, 30)
                O_sma.append(sma)
                
            return O_sma
        
    # TI sma flavor
    def T_sma_F(self):
            
            T_sma = []
            for per in range(int(float(self.In_person_Num)/3)):
                sma = np.random.randint(2, 100)
                T_sma.append(sma)
                
            return T_sma 
        
    # TI transform flavor
    def T_Trans_F(self):
            
            T_TF = []
            for per in range(int(float(self.In_person_Num)/3)):
                tf = np.random.uniform(1, 10)
                T_TF.append(tf)
                
            return T_TF
        
    # OI generate expected price
    def O_EP(self, P_list, V_list, d2_list, d1_list, L1_list, day, who, score):
            t = day+99
            t_1 = day+98
            pt_1 = P_list[t_1]
            f = math.log(abs(V_list[t])) - math.log(abs(pt_1)) 
            if score > abs(f):
                g = 1
            else:
                g = 0
             
            # print('pt_1', pt_1)
            # print('pt', pt)
            # print('vt', V_list[t])
            
            # print('g', g)
            # print('score_start', score)
            
            d2 = d2_list[who-1]
            d1 = d1_list[who-1]
            
            L1 = L1_list[who-1]
            MA = sum(P_list[t-L1:t])/L1
            
            sigma_t = 0
            for i in range(L1):
                ppp = P_list[t-i-1]
                sigma_t = sigma_t+(ppp-MA)**2/L1
                
            # print(sigma_t)
            
            p = pt_1 + (g*(d2*(pt_1-MA) + random.normalvariate(mu=0, sigma=sigma_t)) 
            + (1-g)*(d1*(V_list[t]-pt_1) + random.normalvariate(mu=0, sigma=0.025)))
            
            # print('part_1', g*(d2*(pt_1-MA) + random.normalvariate(mu=0, sigma=sigma_t)))
            # print('part_2', (1-g)*(d1*(V_list[t]-pt_1) + random.normalvariate(mu=0, sigma=0.025)))
            # print('p', p)
            
            signal = (pt_1 - MA)/abs(pt_1 - MA)
            
            return p, signal
            
    # TI generate expected price
    def T_EP(self, P_list, L2_list, gamma_list, day, who, d3, score_1, score_2):
            PTS = P_list
            t = day+99
            t_1 = day+98
            # print('pt', pt)
            
            pt_1 = PTS[t_1]
            
            L2 = L2_list[who-1]
            
            # print('L2', L2)
            
            MA = sum(PTS[t-L2:t])/L2
            
            # print(PTS[t-L2:t])
            # print('MA', MA)
            
            
            sigma_t = 0
            for i in range(L2):
                ppp = PTS[t-i-1]
                sigma_t = sigma_t+(ppp-MA)**2/L2
                
            # print('sigma_t', sigma_t)
                
            p = pt_1 + d3*(pt_1-MA) + random.normalvariate(mu=0, sigma=sigma_t)
            
            # print(p)
            
            signal = d3*(pt_1 - MA)/abs(d3*(pt_1 - MA))
            
            if signal == np.nan:
                signal = 1
                
            # gamma = gamma_list[who-1]
            # score_1 = signal*(math.log(abs(pt))-math.log(abs(pt_1))) + 0.9*score_1
            # score_2 = -signal*(math.log(abs(pt))-math.log(abs(pt_1))) + 0.9*score_2
            
            # ptr = np.exp(gamma*score_2)/(np.exp(gamma*score_1)+np.exp(gamma*score_2))
            # # print('ptr', ptr)
            
            # d3 = d3 + (-2*d3*np.random.binomial(1, ptr, 1))
            # # print('d3', d3)
            
            return p, signal
        
    # random investors generate expected price
    def R_EP(self, P_list, day, who):
            PTS = P_list
            t_1 = day+98
            pt_1 = PTS[t_1]
            
            p = pt_1 + random.normalvariate(mu=0, sigma=0.025)
            return p
        
    # generate all expected price
    def expec_propose_market(self, Day):
            VAL = self.Value()
            PTS = self.Price()
            Risk_flavor = self.RF()
            
            d2_list = self.O_TBP()
            d1_list = self.O_PTV()
            L1_list = self.O_sma_F()
            
            d3_list = self.T_TBP()
            L2_list = self.T_sma_F()
            gamma_list = self.T_Trans_F()
            
            
            OI = [0, 0] # estimate price
            score = list(np.zeros(int(float(self.In_person_Num)/3)))
            
            TI = [0, 0] #record score
            score_1 = list(np.zeros(int(float(self.In_person_Num)/3)))
            score_2 = list(np.zeros(int(float(self.In_person_Num)/3)))
            
            stock = list(100*np.ones(self.In_person_Num))
            cash = list(10000*np.ones(self.In_person_Num))
            wealth = list(100*100*np.ones(self.In_person_Num) + 10000*np.ones(self.In_person_Num))
            
            Wealth = pd.DataFrame()
            Wealth[0] = wealth
            
            wealth1 = wealth
            
            for day in range(1, Day):
                
                # compute the time
                start = time.time()
                
                p1_all = []
                p2_all = []
                p3_all = []
                
                p_signal1 = []
                p_signal2 = []
                
                for who in range(1, int(float(self.In_person_Num)/3)+1):
                    
                    OI = self.O_EP(PTS, VAL, d2_list, d1_list, L1_list, day, who, score[who-1])
                    p1_all.append(OI[0])
                    p_signal1.append(OI[1])
                    
                    TI = self.T_EP(PTS, L2_list, gamma_list, day, who, d3_list[who-1], score_1[who-1], score_2[who-1])
                    p2_all.append(TI[0])
                    p_signal2.append(TI[1])
                    # print(TI[1])
                    
                    RI = self.R_EP(PTS, day, who)
                    p3_all.append(RI)
                
                p_all = p1_all + p2_all + p3_all
            
                # propose price
                m = 3
                
                p_proposed = []
                p_state = []
                t = day + 99
                t_1 = day + 98
                pt_1 = PTS[t_1]
                
                for i in range(len(p_all)):
                    if p_all[i] <= pt_1*0.8:
                        p_all[i] = pt_1*0.8
                    if p_all[i] >= pt_1*1.2:
                        p_all[i] = pt_1*1.2
                        
                # print(p_all)
                
                for item in p_all:
                    
                    # print('item', item)
                    # print('pt_1', pt_1)
                    
                    if item > pt_1:
                        if sum(PTS[t-m:t])/m > item:
                            bid = np.random.uniform(pt_1, item)
                            
                        else:
                            bid = np.random.uniform(sum(PTS[t-m:t])/m, item)
                            
                        p_proposed.append(bid)
                        p_state.append(1)
                            
                    else:
                        if sum(PTS[t-m:t])/m >= item:
                            ask = np.random.uniform(item, sum(PTS[t-m:t])/m)
                            
                        else:
                            ask = np.random.uniform(item, pt_1)
                            
                        p_proposed.append(ask)
                        p_state.append(-1)
                        
                        
                p = 0
                shuffle = random.sample(list(range(self.In_person_Num)), self.In_person_Num)
                
                ask = {}
                bid = {}
                
                ask_p = {}
                bid_p = {}
                
                A_ = 0
                B_ = 0
                
                for i in range(len(p_proposed)):
                    
                    # print(shuffle[i], p_proposed[shuffle[i]])
                    
                    if p_state[shuffle[i]] == -1:
                        
                        if stock[shuffle[i]] > 0:
                            
                            ask.update({shuffle[i]:math.ceil(Risk_flavor[shuffle[i]]*stock[shuffle[i]])})
                            ask_p.update({shuffle[i]:p_proposed[shuffle[i]]})
                            
                            # print('卖量', ask[shuffle[i]])
                            
                            # print('初始卖方', ask)
                            # print('初始买方', bid)
                            # print('初始最大买价', B_)
                            
                            if bid:
                                if p_proposed[shuffle[i]] <= B_:
                                    
                                    B_key = max(bid_p, key=bid_p.get)
                                    # print('初始最大买价余量', bid[B_key])
                                    
                                    tr_stock = min(ask[shuffle[i]], bid[B_key])
                                    
                                    # ask
                                    stock[shuffle[i]] = stock[shuffle[i]] - tr_stock
                                    cash[shuffle[i]] = cash[shuffle[i]] + B_*tr_stock
                                    # wealth1[shuffle[i]] = cash[shuffle[i]] + p*stock[shuffle[i]]
                                    
                                    ask[shuffle[i]] = ask[shuffle[i]] - tr_stock
                                    
                                    # bid
                                    stock[B_key] = stock[B_key] + tr_stock
                                    cash[B_key] = cash[B_key] - B_*tr_stock
                                    # wealth1[B_key] = cash[B_key] + p*stock[B_key]
                                    
                                    bid[B_key] = bid[B_key] - tr_stock
                                    p = B_
                                    
                    else:
                        
                        bid.update({shuffle[i]:math.ceil(Risk_flavor[shuffle[i]]*cash[shuffle[i]]/p_proposed[shuffle[i]])})
                        bid_p.update({shuffle[i]:p_proposed[shuffle[i]]})
                        # print('买量', bid[shuffle[i]])
                        # print('初始卖方', ask)
                        # print('初始买方', bid)
                        # print('初始最小卖价', A_)
                        
                        if ask:
                            if p_proposed[shuffle[i]] >= A_:
                                
                                A_key = min(ask_p, key=ask_p.get)
                                # print('初始最小卖价余量', ask[A_key])
                                
                                tr_stock = min(bid[shuffle[i]], ask[A_key])
                                
                                # bid
                                stock[shuffle[i]] = stock[shuffle[i]] + tr_stock
                                cash[shuffle[i]] = cash[shuffle[i]] - A_*tr_stock
                                # wealth1[shuffle[i]] = cash[shuffle[i]] + p*stock[shuffle[i]]
                                
                                bid[shuffle[i]] = bid[shuffle[i]] - tr_stock
                                
                                # ask
                                stock[A_key] = stock[A_key] - tr_stock
                                cash[A_key] = cash[A_key] + A_*tr_stock
                                # wealth1[A_key] = cash[A_key] + p*stock[A_key]
                                
                                ask[A_key] = ask[A_key] - tr_stock
                                p = A_
                                
                    # delete 0 ask or bid
                    for key in list(ask.keys()):
                        if ask[key] == 0:
                            del ask[key]
                            del ask_p[key]
                            
                    for key in list(bid.keys()):
                        if bid[key] == 0:
                            del bid[key]
                            del bid_p[key]
                            
                    if ask:
                        A_ = min(ask_p.values())
                    else:
                        A_ = 0
                    if bid:
                        B_ = max(bid_p.values())
                    else:
                        B_ = 0
                        
                PTS.append(p)
                
                for i in range(len(p_signal1)):
                    score[i] = p_signal1[i]*(math.log(abs(p))-math.log(abs(pt_1))) + 0.9*score[i]
                
                for i in range(len(p_signal2)):
                    gamma = gamma_list[i]
                    score_1[i] = p_signal2[i]*(math.log(abs(p))-math.log(abs(pt_1)))/10000 + 0.9*score_1[i]
                    score_2[i] = -p_signal2[i]*(math.log(abs(p))-math.log(abs(pt_1)))/10000 + 0.9*score_2[i]
                    
                    ptr = np.exp(gamma*score_2[i])/(np.exp(gamma*score_1[i])+np.exp(gamma*score_2[i]))
                    
                    if ptr == np.nan:
                        ptr = 1
                           
                    # print('ptr', ptr)
                    # print('score1', score_1[i])
                    # print('score2', score_2[i])
                    # print('ptr', ptr)
                    
                    try:
                        d3_list[i] = d3_list[i] + (-2*d3_list[i]*np.random.binomial(1, ptr, 1))
                    except:
                        d3_list[i] = -d3_list[i]
                    
                    # print('d3', d3)
                
                for i in range(len(wealth1)):
                    wealth1[i] = cash[i] + p*stock[i]
                    
                    # print('最小卖价', A_)
                    # print('最大买价', B_)
                    # print('买方', bid)
                    # print('卖方', ask)
                        
                end = time.time()
                print("Day", day, "Running time: %s Seconds"%(end - start))
                    
                Wealth[day] = wealth1
                
            return shuffle, Wealth, ask, bid, VAL, PTS
        
        



if __name__ == '__main__':
    gsm = GSM(300, 10000, 100, 1, 100, 10000)
    shuffle, Wealth, ask, bid, value, price = gsm.expec_propose_market(10001)
    
    
    df_T = pd.DataFrame(Wealth.values.T)
    ### plot
    ### deviation between 1st eigenvalue and 2nd eigenvalue
    fig_1, ax_1 = plt.subplots()
    # ax_2 = ax_1.twinx() # two y axis

    ax_1.plot(price, label='stock price', color='darkcyan')
    ax_1.plot(value, label='stock value', color='darkred')
    ax_1.legend()

    # ax_2.set_xlabel('Date')
    # ax_2.plot(ret_date, indicator_2, label='1st eig substracts 2nd eig', color='grey')
    # ax_2.legend(loc='upper right')
    # ax_1.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    fig_1.show()

    df_1 = df_T.iloc[:, :100]
    df_2 = df_T.iloc[:, 100:200]
    df_3 = df_T.iloc[:, 200:300]
    
    total = df_T.apply ( lambda x: x. sum (), axis = 1 )
    total_organization = df_1.apply ( lambda x: x. sum (), axis = 1 )
    total_trend_trader = df_2.apply ( lambda x: x. sum (), axis = 1 )
    total_random_trader = df_3.apply ( lambda x: x. sum (), axis = 1 )
    
    organization = total_organization/total
    trend_trader = total_trend_trader/total
    random_trader = total_random_trader/total
    
    del total, total_organization, total_trend_trader, total_random_trader
    
    fig_1, ax_1 = plt.subplots()
    
    ax_1.plot(organization, label="Organization Trader", color='green')
    ax_1.plot(trend_trader, label="Trend Trader", color='maroon')
    ax_1.plot(random_trader, label="Random Trader", color='teal')
    ax_1.legend()
    
    fig_1.show()
    
    
    
    
    
    
        
                
    # # trade
    # def propose_price(self):
    #         PTS = self.Price()
            
    #         P_proposed = []
    #         P_state = []
    #         m = 3
            
    #         stock = 100*np.ones(self.Per_num)
    #         cash = 10000*np.ones(self.Per_num)
    #         wealth = stock + cash
            
    #         Wealth = []
    #         Wealth.append(wealth)
            
    #         for d in range(len(self.expec())):
    #             p_proposed = []
    #             p_state = []
    #             t = d+100
    #             t_1 = d+99
    #             pt = PTS[t]
    #             pt_1 = PTS[t_1]
                
    #             Item = self.expec()[d]
                
    #             for item in Item:
                    
    #                 if item > pt_1:
    #                     if sum(PTS[t-m:t])/m > item:
    #                         bid = np.random.uniform(pt_1, item)
                            
    #                     else:
    #                         bid = np.random.uniform(sum(PTS[t-m:t])/m, item)
                            
    #                     p_proposed.append(bid)
    #                     p_state.append(1)
                            
    #                 else:
    #                     if sum(PTS[t-m:t])/m >= item:
    #                         ask = np.random.uniform(item, sum(PTS[t-m:t])/m)
                            
    #                     else:
    #                         ask = np.random.uniform(item, pt_1)
                            
    #                     p_proposed.append(ask)
    #                     p_state.append(-1)
                        
    #             P_proposed.append(p_proposed)
    #             P_state.append(p_state)
            
    #         return P_proposed, P_state

    #         P_proposed = self.propose_price()[0]
    #         P_state = self.propose_price()[1]
    #         Risk_flavor = self.RF()
            

            
    #         for d in range(1):
    #         # for d in range(len(P_proposed)):
    #             p = PTS[d+100]
    #             shuffle = random.sample(list(range(self.In_person_Num)), self.In_person_Num)
    #             bid = {}
    #             ask = {}
                
    #             p_proposed = P_proposed[d]
    #             p_state = P_state[d]
                
    #             A_ = 0
    #             B_ = 0
                
    #             for i in range(len(p_proposed)):
    #                 if p_state[shuffle[i]] == -1:
                        
    #                     ask.update({shuffle[i]:p_proposed[shuffle[i]]})
                        
    #                     if A_ == 0:
    #                         A_ = p_proposed[shuffle[i]]
                            
    #                     else:
    #                         if p_proposed[shuffle[i]] <= A_:
    #                             A_ = p_proposed[shuffle[i]]
    #                         else:
    #                             A_ = A_
                        
    #                     if p_proposed[shuffle[i]] <= B_:
    #                         stock[shuffle[i]] = stock[shuffle[i]] - Risk_flavor[shuffle[i]]*stock[shuffle[i]]
    #                         cash[shuffle[i]] = cash[shuffle[i]] + B_*Risk_flavor[shuffle[i]]*stock[shuffle[i]]
    #                         wealth[shuffle[i]] = cash[shuffle[i]] + p*stock[shuffle[i]]
    #                         ask.pop(shuffle[i])
                            
    #                         B_key = max(bid, key=bid.get)
    #                         stock[B_key] = stock[B_key] + Risk_flavor[B_key]*stock[B_key]
    #                         cash[B_key] = cash[B_key] - B_*Risk_flavor[B_key]*stock[B_key]
    #                         wealth[B_key] = cash[B_key] + p*stock[B_key]
    #                         bid.pop(B_key)
                            
    #                 else:
                        
    #                     bid.update({shuffle[i]:p_proposed[shuffle[i]]})
                        
    #                     if p_proposed[shuffle[i]] >= B_:
    #                         B_ = p_proposed[shuffle[i]]
    #                     else:
    #                         B_ = B_
                            
    #                     if p_proposed[shuffle[i]] >= A_:
    #                         stock[shuffle[i]] = stock[shuffle[i]] + Risk_flavor[shuffle[i]]*stock[shuffle[i]]
    #                         cash[shuffle[i]] = cash[shuffle[i]] - A_*Risk_flavor[shuffle[i]]*stock[shuffle[i]]
    #                         wealth[shuffle[i]] = cash[shuffle[i]] + p*stock[shuffle[i]]
    #                         ask.pop(shuffle[i])
                            
    #                         A_key = min(ask, key=bid.get)
    #                         stock[A_key] = stock[A_key] + Risk_flavor[A_key]*stock[A_key]
    #                         cash[A_key] = cash[A_key] - A_*Risk_flavor[A_key]*stock[A_key]
    #                         wealth[A_key] = cash[A_key] + p*stock[A_key]
    #                         ask.pop(A_key)
                            
    #                 A_ = min(ask.values())
    #                 B_ = max(bid.values())
                    
    #             Wealth = Wealth.append(wealth)
       
        
            
    # # market
    # def market(self):
    #         PTS = self.Price()
    #         P_proposed = self.propose_price()[0]
    #         P_state = self.propose_price()[1]
    #         Risk_flavor = self.RF()
            
    #         stock = 100*np.ones(self.Per_num)
    #         cash = 10000*np.ones(self.Per_num)
    #         wealth = stock + cash
            
    #         Wealth = []
    #         Wealth.append(wealth)
            
    #         for d in range(1):
    #         # for d in range(len(P_proposed)):
    #             p = PTS[d+100]
    #             shuffle = random.sample(list(range(self.Per_num)), self.Per_num)
    #             bid = {}
    #             ask = {}
                
    #             p_proposed = P_proposed[d]
    #             p_state = P_state[d]
                
    #             A_ = 0
    #             B_ = 0
                
    #             for i in range(len(p_proposed)):
    #                 if p_state[shuffle[i]] == -1:
                        
    #                     ask.update({shuffle[i]:p_proposed[shuffle[i]]})
                        
    #                     if A_ == 0:
    #                         A_ = p_proposed[shuffle[i]]
                            
    #                     else:
    #                         if p_proposed[shuffle[i]] <= A_:
    #                             A_ = p_proposed[shuffle[i]]
    #                         else:
    #                             A_ = A_
                        
    #                     if p_proposed[shuffle[i]] <= B_:
    #                         stock[shuffle[i]] = stock[shuffle[i]] - Risk_flavor[shuffle[i]]*stock[shuffle[i]]
    #                         cash[shuffle[i]] = cash[shuffle[i]] + B_*Risk_flavor[shuffle[i]]*stock[shuffle[i]]
    #                         wealth[shuffle[i]] = cash[shuffle[i]] + p*stock[shuffle[i]]
    #                         ask.pop(shuffle[i])
                            
    #                         B_key = max(bid, key=bid.get)
    #                         stock[B_key] = stock[B_key] + Risk_flavor[B_key]*stock[B_key]
    #                         cash[B_key] = cash[B_key] - B_*Risk_flavor[B_key]*stock[B_key]
    #                         wealth[B_key] = cash[B_key] + p*stock[B_key]
    #                         bid.pop(B_key)
                            
    #                 else:
                        
    #                     bid.update({shuffle[i]:p_proposed[shuffle[i]]})
                        
    #                     if p_proposed[shuffle[i]] >= B_:
    #                         B_ = p_proposed[shuffle[i]]
    #                     else:
    #                         B_ = B_
                            
    #                     if p_proposed[shuffle[i]] >= A_:
    #                         stock[shuffle[i]] = stock[shuffle[i]] + Risk_flavor[shuffle[i]]*stock[shuffle[i]]
    #                         cash[shuffle[i]] = cash[shuffle[i]] - A_*Risk_flavor[shuffle[i]]*stock[shuffle[i]]
    #                         wealth[shuffle[i]] = cash[shuffle[i]] + p*stock[shuffle[i]]
    #                         ask.pop(shuffle[i])
                            
    #                         A_key = min(ask, key=bid.get)
    #                         stock[A_key] = stock[A_key] + Risk_flavor[A_key]*stock[A_key]
    #                         cash[A_key] = cash[A_key] - A_*Risk_flavor[A_key]*stock[A_key]
    #                         wealth[A_key] = cash[A_key] + p*stock[A_key]
    #                         ask.pop(A_key)
                            
    #                 A_ = min(ask.values())
    #                 B_ = max(bid.values())
                    
    #             Wealth = Wealth.append(wealth)
     