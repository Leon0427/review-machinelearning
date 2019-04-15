# -*- coding:UTF-8 -*-
'''

Filename: km_fake_nodes.py

km分配全天模拟代码

对全天m个机会，n个等级的咨询师进行全天分配模拟

查看模拟分配情况并可视化保存

Function:

Author: liangxiao@sunlands.com
Create: 2019-03-07 18:03:42


'''

import sys
import numpy as np
from munkres import Munkres
import matplotlib.pyplot as plt
import time
import math

# KM matrix
# 用于存储一次请求中的数据(m*n 矩阵)、计算分配结果(咨询师匹配)，以及维护计算中的一切中间结果(多轮分配的时候，记录已分配的机会和咨询师预分配数)。
class Matrix(object):
    """mat : m x n 的numpy数组，里面存了m个机会与n的咨询师的组合打分
    occupied : mat中所有已分配的机会存在此
    cost_mat : 因为Munkres实现的是最小化损失的km,因此将mat转为cost进行计算，使用mat进行得分统计
    singleLimit: 单批分配的单个咨询师可分配的机会上限
    totalLimit: 单个咨询师未跟进的机会数到达totalLimit便不可再分
    assign_counter_dict : 用于存储各个咨询师的未跟进数

    """
    def __init__(self,mat,assign_counter_dict,hat=1.0,singleLimit=3,totalLimit=16):
        self.mat = mat
        self.cost_mat = hat - self.mat
        self.m = mat.shape[0]
        self.n = mat.shape[1]
        self.singleLimit = singleLimit
        self.totalLimit = totalLimit
        self.assign_counter_dict = assign_counter_dict
        self.pre_distribute_dict = dict.fromkeys(range(mat.shape[1]),0)
        self.occupied = set()
        self.munkres = Munkres()

    """重置中间状态，目前似乎没什么用"""
    def _reset(self):
        self.occupied = set()
        self.pre_distribute_dict = dict.fromkeys(range(self.mat.shape[1]), 0)

    """获取尚未分配的机会的数量"""
    def get_usable_m_num(self):
        return self.m - len(self.occupied)

    """获取还可以分配机会的咨询师数量"""
    def get_usable_n_num(self):
        ns = range(self.mat.shape[1])
        for k,v in self.assign_counter_dict.items():
            if v + self.pre_distribute_dict[k] >= self.totalLimit:
                ns.remove(k)
        for k,v in self.pre_distribute_dict.items():
            if v == self.singleLimit:
                ns.remove(k)
        return len(ns)

    """根据当前的分配状态，获取下一个用于km分配的子矩阵，以及子矩阵的坐标与母矩阵坐标之间的映射关系，将机会数多于咨询师数量的子矩阵转置，然后返回"""
    def get_cost_mat_and_map(self,default_score=None):
        """get the output_mat and the index map between output_map and real_mat"""
        occupied_num = len(self.occupied)
        out_m = self.m - occupied_num
        # get useful n s
        ns = set(range(self.mat.shape[1]))
        for k,v in self.assign_counter_dict.items():
            if v + self.pre_distribute_dict[k] >= self.totalLimit and k in ns:
                ns.remove(k)
        for k,v in self.pre_distribute_dict.items():
            if v == self.singleLimit and k in ns:
                ns.remove(k)
        ns = list(ns)
        acc_map = {}
        for i,n in enumerate(ns):
            acc_map[i] = n
        out_map = dict()
        out_mat = np.full((out_m, len(ns)),-1.0,dtype="float32")
        idx = 0
        for i in range(self.m):
            if i in self.occupied:
                continue
            out_map[idx] = i
            out_mat[idx][:] = self.cost_mat[i][ns]
            idx += 1
        reversed = False
        if out_m>len(ns):
            reversed = True
            return out_mat.T, out_map,acc_map, reversed
        else:
            return out_mat, out_map, acc_map,reversed

    """根据km计算得到的分配列表，更新occupied列表"""
    def update_occupied(self,dist_list):
        for dist in dist_list:
            self.occupied.add(dist)

    """对外的KM分配接口，如果机会数量小于可用咨询师数量，使用单轮分配逻辑；反之，使用多轮分配逻辑"""
    def KM(self,default_score=None):
        n_usable = self.get_usable_n_num()
        m_usable = self.get_usable_m_num()
        if m_usable <= n_usable:
            matches, _sum = self._km_one_round()
        else:
            self._reset()
            matches, _sum = self._km_multiple_round(default_score)
        # print "Multiple round KM input as [ (m-i*n) x (m-i*n) ] when m > n:"
        # print "Matches are: %s" % sorted(matches, key=lambda x: x[1])
        # print "Matches are: %s" % sorted(matches, key=lambda x: x[0])
        print "Match Num is : %s" % len(matches)
        print "Sum score is : %s" % _sum
        return _sum, self.assign_counter_dict, len(matches)

    """单轮分配逻辑"""
    def _km_one_round(self):
        out_mat, out_map, acc_map, reversed = self.get_cost_mat_and_map()
        matches = self.munkres.compute(out_mat)
        _sum = 0.0
        for row, col in matches:
            value = self.mat[out_map[row]][acc_map[col]]
            self.assign_counter_dict[acc_map[col]] += 1
            _sum += value
        return matches, _sum

    """用于将转置矩阵的计算结果转置回去，没分出去的机会匹配的咨询师编号为-1"""
    def _reverse_matches(self, matches):
        max_j = max([k[1] for k in matches])
        reversed_matches = [[i,-1] for i in range(max_j + 1)]
        for i,j in matches:
            reversed_matches[j][1] = i
        return reversed_matches

    """多轮分配逻辑，如果根据当前分配状态，可分机会/可分咨询师有一项为0，停止分配；否则持续分配；每经过一轮分配，更新内部状态"""
    def _km_multiple_round(self, default_score=None):
        sum_ = 0.0
        all_matches = []
        while True:
            mat, map_,acc_map, reversed = self.get_cost_mat_and_map(default_score)
            if len(acc_map) ==0 or len(map_)==0:
                break
            matches_raw = self.munkres.compute(mat)
            if reversed:
                matches_raw = self._reverse_matches(matches_raw)
            matches = []
            # print "matches_raw %s " %matches_raw
            for row, col in matches_raw:
                if col>=0:
                    matches.append((map_[row], acc_map[col]))
                    val = self.mat[map_[row]][acc_map[col]]
                    self.assign_counter_dict[acc_map[col]] += 1
                    self.pre_distribute_dict[acc_map[col]] += 1
                    sum_ += val
            # print "matches: %s"%matches
            dist_list = [x[0] for x in matches]
            all_matches.extend(matches)
            self.update_occupied(dist_list)
        return all_matches, sum_

"""根据已分配未跟进机会数 和 已跟进机会数 字典，打印每个咨询师的分配与跟进情况"""
def plot_assignment(assign_counter_dict, follow_counter_dict,tag=""):
    x_data = []
    y_data = []
    assign_dict = merge_dict(assign_counter_dict, follow_counter_dict)
    # print assign_dict
    for k,v in assign_dict.items():
        x_data.append(k)
        y_data.append(v)
    plt.bar(x_data,y_data)
    # follow data
    x_follow_data =[]
    y_follow_data =[]
    for k,v in follow_counter_dict.items():
        x_follow_data.append(k)
        y_follow_data.append(v)
    plt.bar(x_follow_data,y_follow_data)
    plt.xticks(x_data)
    plt.ylim(bottom=0,top=30)
    plt.savefig("/home/liangxiao/project_data/personalized_alignment_classify/exp/plot_exp/%s_plot_assignment_%.2f.jpg"%(tag,time.time()))
    plt.cla()

"""按照每个咨询师全天跟进25%机会的期望，从25% * 0.5到 25% * 1.5 差异化咨询师的跟进能力，用概率的形式将跟进量分散到每一次请求之前"""
def generate_follow_num(m , n = 10, follow_ability_diff_num=3,time_span=5,average_follow_rate = 0.25):

    # 获得平均期望跟进数量
    averate_follow_each_time_span_evenly = m * average_follow_rate / (n * follow_ability_diff_num) / time_span
    weaker = averate_follow_each_time_span_evenly * 0.5
    stronger = averate_follow_each_time_span_evenly * 1.5
    averager = averate_follow_each_time_span_evenly
    def decimal_to_int_base_on_probability(f):
        parts = math.modf(f)
        decimal_part = parts[0]
        int_part = parts[1]
        if np.random.random() < decimal_part:
            return int_part + 1
        else:
            return int_part
    ans = []
    for i in range(n):
        ans.extend([decimal_to_int_base_on_probability(x) for x in [weaker, averager, stronger]])
    return ans

def merge_dict(dic1, dic2):
    assert dic1.keys() == dic2.keys()
    ans = {}
    for k in dic1.keys():
        ans[k] = dic1[k] + dic2[k]
    return ans

if __name__ == "__main__":
    m = int(sys.argv[1])
    n = int(sys.argv[2])

    singleLimit = int(sys.argv[3])
    totalLimit = int(sys.argv[4])

    disturb_strength = float(sys.argv[5])
    follow_ability_diff_num = int(sys.argv[6])
    "0. 构建(m, n)的打分矩阵"

    # 初始化機會分佈
    real_mat = np.random.normal(scale=0.1,size=(m,1))
    real_mat = np.abs(real_mat)


    # 扩充到n级咨询师，每级咨询师有follow_ability_diff_num个跟进能力不同的咨询师，初始化個性化擾動
    real_mat = np.hstack([real_mat] * n * follow_ability_diff_num)
    disturb = 1 + disturb_strength * 2 * (np.random.random(size=real_mat.shape) - 0.5) # 0.8~1.2之間的擾動
    real_mat = real_mat * disturb
    # print real_mat

    "exp. 差異化咨詢師能力"
    for i in range(n):
        ability = 0.1*(i+1)
        real_mat[:,i * follow_ability_diff_num: (i+1)*follow_ability_diff_num] *= ability

    # 截斷機會成單率到有效區間
    real_mat[real_mat>1.0] = 1.0
    real_mat[real_mat<0.0] = 0.0

    # print real_mat.shape
    #
    # sys.exit(0)
    # "1. 多轮分配 km 算法：共m个机会，n个咨询师，每轮分配n个机会，输入km算法的矩阵为(n,n)"
    # cnt = 0
    # km = Munkres()
    # if 1:
    # # if m>n:
    #     matches = []
    #     _sum = 0.0
    #     head = 0
    #     while head < m:
    #         sub_mat = real_mat[head:min(head+n, m)]
    #         cost_sub_mat = 1.0 - sub_mat
    #         indexes = km.compute(cost_sub_mat)
    #         for row, col in indexes:
    #             matches.append((head+row, col))
    #             _sum += sub_mat[row][col]
    #         head += n
    #         cnt += 1
    #         if cnt == singleLimit:
    #             break
    #     print "Multiple round KM input as [ n x n ] when m > n:"
    #     print "Matches: %s" % sorted(matches,key = lambda x: x[0])
    #     print "Score sum: %s" % _sum

    "exp. 切分打分矩陣"
    def cut_mat_reality(mat,opp_bottom_bound=1,opp_upper_bound = 3):
        assert mat.shape[0]/mat.shape[1] > totalLimit
        mats = []
        multi = singleLimit
        odd = 0
        start_idx = 0
        while start_idx<mat.shape[0]:
            scale = multi * mat.shape[1]  + odd
            end_idx = min(start_idx + scale,mat.shape[0])
            mats.append(mat[start_idx:end_idx])
            multi = max(0,multi-1)
            start_idx = end_idx
            odd = np.random.randint(opp_bottom_bound,opp_upper_bound)
        return mats

    def cut_mat_evenly(mat):
        assert mat.shape[0]/mat.shape[1] > totalLimit
        mats = []
        multi = singleLimit
        start_idx = 0
        while start_idx<mat.shape[0]:
            scale = multi * mat.shape[1]
            end_idx = min(start_idx + scale,mat.shape[0])
            mats.append(mat[start_idx:end_idx])
            start_idx = end_idx
        return mats

    mats_evenly = cut_mat_evenly(real_mat)
    mats_reality =  cut_mat_reality(real_mat)
    mats_wait =  cut_mat_reality(real_mat,opp_bottom_bound=0.3*n*follow_ability_diff_num,opp_upper_bound=0.6 * n * follow_ability_diff_num)

    # 获取时间间隔对比
    time_spans_evenly = len(mats_evenly)
    time_spans_reality = len(mats_reality)
    time_spans_wait = len(mats_wait)


    assign_counter_dict = dict.fromkeys(range(n*follow_ability_diff_num),0)
    follow_counter_dict = dict.fromkeys(range(n*follow_ability_diff_num),0)
    sum1 = 0.0
    dis_sum1 = 0
    for round_i,real_m in enumerate(mats_reality):
        if round_i != 0:
            follows = generate_follow_num(m, n, follow_ability_diff_num=follow_ability_diff_num, time_span=time_spans_reality)
            for ii,fl in enumerate(follows):
                fl = min(assign_counter_dict[ii], fl)
                follow_counter_dict[ii] +=  fl
                assign_counter_dict[ii] -= fl
        mat = Matrix(real_m, assign_counter_dict, singleLimit=singleLimit, totalLimit=totalLimit)
        s, assign_counter_dict, dis_num = mat.KM()
        plot_assignment(assign_counter_dict, follow_counter_dict,tag="real")
        dis_sum1 += dis_num
        sum1 +=s
    print "[ reality ] sum score 1: %s, dis num : %s" %(sum1,dis_sum1)
    print "================================================="

    assign_counter_dict = dict.fromkeys(range(n*follow_ability_diff_num),0)
    follow_counter_dict = dict.fromkeys(range(n*follow_ability_diff_num),0)
    sum2 = 0.0
    dis_sum2 = 0
    for round_i,real_m in enumerate(mats_wait):
        if round_i != 0:
            follows = generate_follow_num(m, n, follow_ability_diff_num=follow_ability_diff_num, time_span=time_spans_wait)
            for ii,fl in enumerate(follows):
                fl = min(assign_counter_dict[ii], fl)
                follow_counter_dict[ii] +=  fl
                assign_counter_dict[ii] -= fl
        mat = Matrix(real_m, assign_counter_dict, singleLimit=singleLimit, totalLimit=totalLimit)
        s, assign_counter_dict, dis_num = mat.KM()
        plot_assignment(assign_counter_dict, follow_counter_dict,tag="wait")
        dis_sum2 += dis_num
        sum2 +=s
    print "[ wait ], sum score 2: %s, dis num : %s" %(sum2,dis_sum2)
    print "================================================="


    assign_counter_dict = dict.fromkeys(range(n*follow_ability_diff_num),0)
    follow_counter_dict = dict.fromkeys(range(n * follow_ability_diff_num), 0)
    sum3 = 0.0
    dis_sum3 = 0
    for round_i,even_m in enumerate(mats_evenly):
        if round_i != 0:
            follows = generate_follow_num(m, n, follow_ability_diff_num=follow_ability_diff_num, time_span=time_spans_evenly)
            for ii,fl in enumerate(follows):
                fl = min(assign_counter_dict[ii], fl)
                follow_counter_dict[ii] +=  fl
                assign_counter_dict[ii] -= fl
        mat = Matrix(even_m, assign_counter_dict, singleLimit=singleLimit, totalLimit=totalLimit)
        s, assign_counter_dict, dis_num = mat.KM()
        plot_assignment(assign_counter_dict, follow_counter_dict, tag="even")
        dis_sum3 += dis_num
        sum3 +=s
    print "[ even ], sum score 3: %s, dis num : %s"%(sum3, dis_sum3)

    print "content\treality\twait\teven"
    print "sum_score\t%.2f\t%.2f\t%.2f"%(sum1,sum2,sum3)
    print "dis_num\t%s\t%s\t%s"%(dis_sum1,dis_sum2,dis_sum3)
    print "avg_score\t%.4f\t%.4f\t%.4f"%(sum1/dis_sum1, sum2/dis_sum2, sum3/dis_sum3)
    print "incre\t%.3f\t%.3f\t%.3f" % ( 0.0, (sum2/dis_sum2 - sum1/dis_sum1)/(sum1/dis_sum1),  (sum3/dis_sum3 - sum1/dis_sum1)/(sum1/dis_sum1) )




    # "2. 虚拟咨询师 多轮分配 km 算法：共m个机会，n个咨询师，每轮分配n个机会，输入km算法的矩阵为(m - i*n, m - i*n)"
    # mat = Matrix(real_mat,assign_counter_dict,singleLimit=singleLimit,totalLimit=totalLimit)
    # s, dic = mat.KM()
    # print s, dic
