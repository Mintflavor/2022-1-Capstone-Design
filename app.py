#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, url_for, redirect
from flask_bootstrap import Bootstrap
from flask_compress import Compress
from haversine import haversine

app = Flask(__name__)
app.jinja_env.globals.update(
    zip=zip,
    enumerate=enumerate,
)
Bootstrap(app)
Compress(app)

districtDF = pd.read_csv('assets/data/proc_sang.csv')
categoryDF = pd.read_csv('assets/data/proc_upjong.csv')
addr = pd.read_csv('assets/data/addr.csv')
parking = pd.read_csv('assets/data/parking.csv')

guList = list(addr['시군구명'].drop_duplicates())
guList.sort()
gu = ""
dongList = []
dong = ""

districts = []
district = ""
categories = []
category = ""

guRankList = np.load('assets/data/guRank.npy', allow_pickle=True).tolist()
seoulRankList = np.load('assets/data/seoulRank.npy', allow_pickle=True).tolist()

@app.route('/')
@app.route('/index.html')
def index():
    global categories
    global dongList
    global districts
    categories = []
    dongList = []
    districts = []

    return render_template('blank.html', title="Capstone", guList=guList, dongList=dongList, gu=gu, dong=dong, districts=districts, categories=categories, district=district, category=category)

@app.route('/404.html')
def not_found():
    global categories
    global dongList
    global districts
    categories = []
    dongList = []
    districts = []

    return render_template('404.html', title="Capstone", guList=guList, dongList=dongList, gu=gu, dong=dong, districts=districts, categories=categories, district=district, category=category)

@app.route('/getDongList')
def get_dong_list():
    gu = request.args.get('gu')
    global dongList
    dongList = list(addr[addr['시군구명'] == gu]['행정동명'].drop_duplicates())
    dongList.sort()

    return str(dongList)

@app.route('/getDistricts')
def get_districts():
    dong = request.args.get('dong')
    global districts
    districts = list(addr[addr['행정동명'] == dong]['상권_코드_명'].drop_duplicates())
    districts.sort()

    return str(districts)

@app.route('/getCategories')
def get_categories():
    district = request.args.get('district')
    global categories
    recent_category_df = categoryDF[(categoryDF['상권명']==district) & (categoryDF['연도'] == categoryDF['연도'].max())]
    recent_category_df = recent_category_df[recent_category_df['분기'] == recent_category_df['분기'].max()]
    categories = list(recent_category_df['업종명'].drop_duplicates())
    categories.sort()

    return str(categories)

@app.route('/getNearTimePopData')
def get_near_time_pop_data():
    near_district = request.args.get('nearDistrict')
    district_selected = districtDF[districtDF['상권명']==near_district]
    time_population = district_selected[['시간대_1_생활인구_수', '시간대_2_생활인구_수', '시간대_3_생활인구_수', \
                                       '시간대_4_생활인구_수', '시간대_5_생활인구_수', '시간대_6_생활인구_수']].values.tolist()[0]

    return str(time_population)

@app.route('/getNearTimeSalesVolumeData')
def get_near_time_sales_volume_data():
    category = request.args.get('category')
    near_district = request.args.get('nearDistrict')
    category_selected = categoryDF[(categoryDF['상권명']==near_district) & (categoryDF['업종명']==category)]
    try:
        time_sales_volume = list(category_selected[['시간대_00~06_매출_금액', '시간대_06~11_매출_금액', '시간대_11~14_매출_금액', \
                                                 '시간대_14~17_매출_금액', '시간대_17~21_매출_금액', '시간대_21~24_매출_금액']].iloc[0])
    except:
        time_sales_volume = []

    return str(time_sales_volume)

@app.route('/getCprData')
def get_cpr_data():
    dong = request.args.get('dong')
    cpr_districts = list(addr[addr['행정동명']==dong]['상권_코드_명'].drop_duplicates())
    cpr_districts.sort()
    districts_mean = districtDF[districtDF['상권명'].isin(cpr_districts)].groupby('상권명').mean().drop(['연도','분기'], axis=1)
    districts_mean_std = districts_mean.copy()
    districts_mean_std = (districts_mean_std / districts_mean_std.mean() -1) * 100
    districts_mean_std.loc['지역 평균'] = districts_mean.mean()
    districts_mean_std = districts_mean_std.dropna(axis=1)
    columns = list(districts_mean_std.columns)
    districts_mean_std_json = districts_mean_std.to_json(orient='index', force_ascii=False)

    return str([columns, districts_mean_std_json])

@app.route('/getCprSalesCat')
def get_cpr_sales_cat():
    dong = request.args.get('dong')
    cpr_districts = list(addr[addr['행정동명']==dong]['상권_코드_명'].drop_duplicates())
    cpr_districts.sort()
    categories_mean_cat = list(np.sort(categoryDF[categoryDF['상권명'].isin(cpr_districts)]['업종명'].unique()))

    return str(categories_mean_cat)

@app.route('/getCprSalesData')
def get_cpr_sales_data():
    dong = request.args.get('dong')
    category = request.args.get('category')
    cpr_districts = list(addr[addr['행정동명']==dong]['상권_코드_명'].drop_duplicates())
    cpr_districts.sort()

    categories_mean = categoryDF[categoryDF['상권명'].isin(cpr_districts)].groupby(['상권명', '업종명']).mean().drop(['연도','분기'], axis=1)
    categories_mean_std = categories_mean.copy()
    categories_mean_std = categories_mean_std.xs(category, level=1)
    categories_mean_std = (categories_mean_std / categories_mean_std.mean() -1) * 100
    categories_mean_std.loc['지역 평균'] = categories_mean.mean()
    categories_mean_std = categories_mean_std.dropna(axis=1)
    cat_columns = list(categories_mean_std.columns)
    categories_mean_std_json = categories_mean_std.to_json(orient='index', force_ascii=False)

    districts_mean = districtDF[districtDF['상권명'].isin(cpr_districts)].groupby('상권명').mean().drop(['연도','분기'], axis=1)
    districts_mean_std = districts_mean.copy()
    districts_mean_std = districts_mean_std.loc[categories_mean_std.index[:-1]]
    districts_mean_std = (districts_mean_std / districts_mean_std.mean() -1) * 100
    districts_mean_std.loc['지역 평균'] = districts_mean.mean()
    districts_mean_std = districts_mean_std.dropna(axis=1)
    columns = list(districts_mean_std.columns)
    districts_mean_std_json = districts_mean_std.to_json(orient='index', force_ascii=False)

    if len(categories_mean_std) <= 1:
        return "[0, 0, {}]"

    return str([columns, cat_columns, districts_mean_std_json, categories_mean_std_json])

@app.route('/map.html')
def map_page():
    global categories
    global dongList
    global districts
    categories = []
    dongList = []
    districts = []

    return render_template('map.html', title="Capstone", guList=guList, dongList=dongList, gu=gu, dong=dong, districts=districts, categories=categories, district=district, category=category)

@app.route('/compare.html')
def compare():
    global categories
    global dongList
    global districts
    categories = []
    dongList = []
    districts = []

    return render_template('compare.html', title="Capstone", guList=guList, dongList=dongList, gu=gu, dong=dong, districts=districts, categories=categories, district=district, category=category)

@app.route('/dashboard.html', methods=['POST', 'GET'])
def dashboard():
    if request.method == 'POST':
        pass

    elif request.method == 'GET':
        gu = request.args.get('gu')
        dong = request.args.get('dong')
        district = request.args.get('district')
        category = request.args.get('category')

        if gu == '구 선택' or gu == None or dong == '동 선택' or dong == None or district == '상권명 선택' or district == None or category == '업종명 선택' or category == None:
            return redirect(url_for('index'))

    district_selected = districtDF[districtDF['상권명'] == district]
    category_selected = categoryDF[(categoryDF['상권명'] == district) & (categoryDF['업종명'] == category)]
    recent_category_df = categoryDF[(categoryDF['상권명'] == district) & (categoryDF['연도'] == categoryDF['연도'].max())]
    recent_category_df = recent_category_df[recent_category_df['분기'] == recent_category_df['분기'].max()]

    #상권 연도별 총 유동인구
    floating_population = list(district_selected.groupby(['연도']).mean()['총_생활인구_수'])
    floating_pop_index = [f"{i}년" for i in district_selected.groupby(['연도']).mean().index]
    floating_pop_change_ratio = []
    floating_pop_change_ratio_index = []
    for i in range(len(floating_population)-1):
        diff = floating_population[i+1] - floating_population[i]
        floating_pop_change_ratio.append(round((diff/floating_population[i])*100, 1))
        floating_pop_change_ratio_index.append(f"{district_selected.groupby(['연도']).mean().index[i]}년->{district_selected.groupby(['연도']).mean().index[i]+1}년")

    #모든 기간 상권 총 유동인구
    whole_floating_population = list(district_selected['총_생활인구_수'])
    whole_floating_pop_index = list()
    for year, quarter in zip(district_selected['연도'], district_selected['분기']):
        whole_floating_pop_index.append(f"{year}-{quarter}")

    whole_floating_population = list(reversed(whole_floating_population))
    whole_floating_pop_index = list(reversed(whole_floating_pop_index))

    #가장 최근 연령대별 유동인구
    age_group_population = district_selected[['연령대_10_생활인구_수', '연령대_20_생활인구_수', '연령대_30_생활인구_수',
                                              '연령대_40_생활인구_수', '연령대_50_생활인구_수', '연령대_60_이상_생활인구_수']].values.tolist()[0]
    age_group_pop_index = ['10대', '20대', '30대', '40대', '50대', '60대 이상']

    #가장 최근 시간대별 유동인구
    time_population = district_selected[['시간대_1_생활인구_수', '시간대_2_생활인구_수', '시간대_3_생활인구_수',
                                         '시간대_4_생활인구_수', '시간대_5_생활인구_수', '시간대_6_생활인구_수']].values.tolist()[0]
    time_pop_index = ['0~6시', '6~11시', '11~14시', '14~17시', '17~21시', '21~24시']

    #가장 최근 시간대별 매출 규모
    try:
        time_sales_volume = list(category_selected[['시간대_00~06_매출_금액', '시간대_06~11_매출_금액', '시간대_11~14_매출_금액',
                                                  '시간대_14~17_매출_금액', '시간대_17~21_매출_금액', '시간대_21~24_매출_금액']].iloc[0])
        time_sales_volume_index = ['0~6시', '6~11시', '11~14시', '14~17시', '17~21시', '21~24시']
    except:
        time_sales_volume = []
        time_sales_volume_index = []

    ### 이거 돈 단위 *1000원 아닌가?

    #가장 최근 선택한 상권, 업종 분기별 총 매출 규모
    quarter_sales_volume_list = category_selected[category_selected['연도']==category_selected['연도'].max()].sort_values(by='분기', ascending=True)
    quarter_sales_volume = list((quarter_sales_volume_list['분기당_매출_금액']).values)
    quarter_sales_volume_index = [f"{i}분기" for i in quarter_sales_volume_list['분기']]
    quarter_sales_volume_change_ratio = []
    quarter_sales_volume_change_ratio_index = []
    for i in range(len(quarter_sales_volume)-1):
        diff = quarter_sales_volume[i+1] - quarter_sales_volume[i]
        quarter_sales_volume_change_ratio.append(round((diff/quarter_sales_volume[i])*100, 1))
        quarter_sales_volume_change_ratio_index.append(f"{quarter_sales_volume_index[i]}->{quarter_sales_volume_index[i+1]}")

    #모든 기간 선택한 상권, 업종 총 매출 규모
    whole_quarter_sales_volume = list(category_selected['분기당_매출_금액'])
    whole_quarter_sales_volume_index = list()
    for year, quarter in zip(category_selected['연도'], category_selected['분기']):
        whole_quarter_sales_volume_index.append(f"{year}-{quarter}")

    whole_quarter_sales_volume = list(reversed(whole_quarter_sales_volume))
    whole_quarter_sales_volume_index = list(reversed(whole_quarter_sales_volume_index))

    #가장 최근 선택한 상권, 업종 분기별 점포당 평균 매출 규모
    #점포수 : 프랜차이즈 포함한 모든 점포 수
    #점포_수 : 프랜차이즈 제외한 모든 점포 수
    #프랜차이즈_점포_수
    market_avg_sales_volume = list(quarter_sales_volume_list['분기당_매출_금액']/quarter_sales_volume_list['점포수'])
    market_avg_sales_volume_index = quarter_sales_volume_index
    market_avg_sales_volume_change_ratio = []
    market_avg_sales_volume_change_ratio_index = []
    for i in range(len(market_avg_sales_volume)-1):
        diff = market_avg_sales_volume[i+1] - market_avg_sales_volume[i]
        market_avg_sales_volume_change_ratio.append(round((diff/market_avg_sales_volume[i])*100, 1))
        market_avg_sales_volume_change_ratio_index.append(f"{market_avg_sales_volume_index[i]}->{market_avg_sales_volume_index[i+1]}")

    #성별 매출 비율
    try:
        sex_sales_volume_ratio = list(category_selected[['남성_매출_금액','여성_매출_금액']].iloc[0])
        sex_sales_volume_ratio_index = ['남자', '여자']
    except:
        sex_sales_volume_ratio = []
        sex_sales_volume_ratio_index = []

    #연령대별 매출 비율
    try:
        age_sales_volume_ratio = list(category_selected[['연령대_10_매출_금액', '연령대_20_매출_금액', '연령대_30_매출_금액', \
                                                     '연령대_40_매출_금액', '연령대_50_매출_금액', '연령대_60_이상_매출_금액']].iloc[0])
        age_sales_volume_ratio_index = ['10대', '20대', '30대', '40대', '50대', '60대 이상']
    except:
        age_sales_volume_ratio = []
        age_sales_volume_ratio_index = []

    #프랜차이즈 비율
    try:
        franchise_ratio = list(category_selected.iloc[0][['프랜차이즈_점포_수', '점포_수']].astype(int))
        franchise_ratio_index = ['프랜차이즈 수', '비 프랜차이즈 수']

    except:
        franchise_ratio = []
        franchise_ratio_index = []

    #가장 최근 업종 밀집도
    category_density = list(pd.Series(list(recent_category_df['점포수']), index=list(recent_category_df['업종명'])).sort_index())
    category_density_index = list(pd.Series(list(recent_category_df['점포수']), index=list(recent_category_df['업종명'])).sort_index().index)

    #주차장
    try:
        count_parking = 0
        district_coor = [addr[addr['상권_코드_명']==district]['위도'].iloc[0], addr[addr['상권_코드_명'] == district]['경도'].iloc[0]]
        area_name = addr[addr['상권_코드_명']==district]['시군구명'].iloc[0]
        area_parking = parking[parking['주소']==area_name]
        for i in area_parking.index:
            parking_coor = [area_parking.loc[i]['위도'], area_parking.loc[i]['경도']]
            if haversine(district_coor, parking_coor, unit='m') <= 500:
                count_parking += 1
    except:
        pass

    #주변상권 찾기(5개)
    near_districts = list()
    for i in addr.index:
        near_coor = [addr.loc[i]['위도'], addr.loc[i]['경도']]
        distance = haversine(district_coor, near_coor, unit='m')
        if (distance <= 1000) & (addr.loc[i]['상권_코드_명'] != district):
            near_districts.append((addr.loc[i]['상권_코드_명'], distance))
    near_districts = sorted(near_districts, key=lambda nearDistricts: nearDistricts[1])[:5]

    #가장 최근 집객시설 수
    count_school = int(district_selected.iloc[0][['초등학교_수', '중학교_수', '고등학교_수']].sum())
    count_univ = int(district_selected.iloc[0]['대학교_수'])
    count_transport = int(district_selected.iloc[0][['철도_역_수', '버스_터미널_수', '지하철_역_수', '버스_정거장_수']].sum())

    #서울 전체 상권 랭크
    global seoulRankList
    seoul_rank = list(seoulRankList[category].keys()).index(district)+1
    seoul_rank_m1 = list(seoulRankList[category].keys()).index(district)
    seoul_rank_p1 = list(seoulRankList[category].keys()).index(district)+2
    seoul_rank_last = len(seoulRankList[category])-2
    seoul_rank1_district = list(seoulRankList[category].keys())[0]
    seoul_rank_m1_district = list(seoulRankList[category].keys())[seoul_rank_m1-1]
    seoul_rank_p1_district = list(seoulRankList[category].keys())[seoul_rank_m1+1]
    seoul_rank_last_district = list(seoulRankList[category].keys())[-3]
    seoul_rank_whole_district = list(seoulRankList[category].keys())[:-2]
    try:
        seoul_rank_m1_score = round(list(seoulRankList[category].values())[seoul_rank_m1-1][-1], 2)
    except:
        seoul_rank_m1_score = -1
    seoul_rank_score = round(list(seoulRankList[category].values())[seoul_rank_m1][-1], 2)
    seoul_rank_p1_score = round(list(seoulRankList[category].values())[seoul_rank_m1+1][-1], 2)
    seoul_rank_whole_score = [round(seoulRankList[category][d][-1], 2) for d in list(seoulRankList[category].keys())[:-2]]

    #지역구 내 상권 랭크
    global guRankList

    gu_rank = list(guRankList[gu][category].keys()).index(district)+1
    gu_rank_m1 = list(guRankList[gu][category].keys()).index(district)
    gu_rank_p1 = list(guRankList[gu][category].keys()).index(district)+2
    gu_rank_last = len(guRankList[gu][category])-2
    gu_rank1_district = list(guRankList[gu][category].keys())[0]
    gu_rank_m1_district = list(guRankList[gu][category].keys())[gu_rank_m1-1]
    gu_rank_p1_district = list(guRankList[gu][category].keys())[gu_rank_m1+1]
    gu_rank_last_district = list(guRankList[gu][category].keys())[-3]
    gu_rank_whole_district = list(guRankList[gu][category].keys())[:-2]
    try:
        gu_rank_m1_score = round(list(guRankList[gu][category].values())[gu_rank_m1-1][-1], 2)
    except:
        gu_rank_m1_score = -1
    gu_rank_score = round(list(guRankList[gu][category].values())[gu_rank_m1][-1], 2)
    gu_rank_p1_score = round(list(guRankList[gu][category].values())[gu_rank_m1+1][-1], 2)
    gu_rank_whole_score = [round(guRankList[gu][category][d][-1], 2) for d in list(guRankList[gu][category].keys())[:-2]]

    #서울 전체 상권 랭크 산정 기준
    seoul_features_ratio = seoulRankList[category]['weights']
    seoul_features = seoulRankList[category]['columns']

    #지역구 내 상권 랭크 산정 기준
    gu_features_ratio = guRankList[gu][category]['weights']
    gu_features = guRankList[gu][category]['columns']


    return render_template('dashboard.html', title="Capstone", guList=guList, dongList=dongList, gu=gu, dong=dong,
                           districts=districts, categories=categories,
                           district=district, category=category,
                           districtCategory=district+" "+category,
                           floatingPopulation=floating_population, floatingPopIndex=floating_pop_index,
                           floatingPopChangeRatio=floating_pop_change_ratio, floatingPopChangeRatioIndex=floating_pop_change_ratio_index,
                           wholeFloatingPopulation=whole_floating_population, wholeFloatingPopIndex=whole_floating_pop_index,
                           ageGroupPopulation=age_group_population, ageGroupPopIndex=age_group_pop_index,
                           timePopulation=time_population, timePopIndex=time_pop_index,
                           timeSalesVolume=time_sales_volume, timeSalesVolumeIndex=time_sales_volume_index,
                           quarterSalesVolume=quarter_sales_volume, quarterSalesVolumeIndex=quarter_sales_volume_index,
                           quarterSalesVolumeChangeRatio=quarter_sales_volume_change_ratio, quarterSalesVolumeChangeRatioIndex=quarter_sales_volume_change_ratio_index,
                           wholeQuarterSalesVolume=whole_quarter_sales_volume, wholeQuarterSalesVolumeIndex=whole_quarter_sales_volume_index,
                           marketAvgSalesVolume=market_avg_sales_volume, marketAvgSalesVolumeIndex=market_avg_sales_volume_index,
                           marketAvgSalesVolumeChangeRatio=market_avg_sales_volume_change_ratio, marketAvgSalesVolumeChangeRatioIndex=market_avg_sales_volume_change_ratio_index,
                           sexSalesVolumeRatio=sex_sales_volume_ratio, sexSalesVolumeRatioIndex=sex_sales_volume_ratio_index,
                           ageSalesVolumeRatio=age_sales_volume_ratio, ageSalesVolumeRatioIndex=age_sales_volume_ratio_index,
                           franchiseRatio=franchise_ratio, franchiseRatioIndex=franchise_ratio_index,
                           categoryDensity=category_density, categoryDensityIndex=category_density_index,
                           nearDistricts=near_districts,
                           countParking=count_parking, countSchool=count_school, countUniv=count_univ, countTransport=count_transport,
                           guRank=gu_rank, guRankM1=gu_rank_m1, guRankP1=gu_rank_p1, guRankLast=gu_rank_last, guRank1District=gu_rank1_district, guRankM1District=gu_rank_m1_district,
                           guRankP1District=gu_rank_p1_district, guRankLastDistrict=gu_rank_last_district, guRankM1Score=gu_rank_m1_score, guRankScore=gu_rank_score, guRankP1Score=gu_rank_p1_score,
                           seoulRank=seoul_rank, seoulRankM1=seoul_rank_m1, seoulRankP1=seoul_rank_p1, seoulRankLast=seoul_rank_last, seoulRank1District=seoul_rank1_district, seoulRankM1District=seoul_rank_m1_district,
                           seoulRankP1District=seoul_rank_p1_district, seoulRankLastDistrict=seoul_rank_last_district, seoulRankM1Score=seoul_rank_m1_score, seoulRankScore=seoul_rank_score, seoulRankP1Score=seoul_rank_p1_score,
                           guFeaturesRatio=gu_features_ratio, guFeatures=gu_features, seoulFeaturesRatio=seoul_features_ratio, seoulFeatures=seoul_features,
                           seoulRankWholeDistrict=seoul_rank_whole_district, seoulRankWholeScore=seoul_rank_whole_score, guRankWholeDistrict=gu_rank_whole_district, guRankWholeScore=gu_rank_whole_score
                           )

@app.route('/ref.html')
def ref():
    return render_template('ref.html', title="Capstone", districts=districts, categories=categories, district=district, category=category)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False)
