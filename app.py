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

    return render_template('blank.html', title="Capstone", guList=guList, dongList=dongList, gu=gu, dong=dong,
                           districts=districts, categories=categories, district=district, category=category)

@app.route('/404.html')
def not_found():
    global categories
    global dongList
    global districts
    categories = []
    dongList = []
    districts = []

    return render_template('404.html', title="Capstone", guList=guList, dongList=dongList, gu=gu, dong=dong,
                           districts=districts, categories=categories, district=district, category=category)

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
    recentCategoryDF = categoryDF[(categoryDF['상권명']==district) & (categoryDF['연도'] == categoryDF['연도'].max())]
    recentCategoryDF = recentCategoryDF[recentCategoryDF['분기'] == recentCategoryDF['분기'].max()]
    categories = list(recentCategoryDF['업종명'].drop_duplicates())
    categories.sort()

    return str(categories)

@app.route('/getNearTimePopData')
def get_near_time_pop_data():
    nearDistrict = request.args.get('nearDistrict')
    districtSelected = districtDF[districtDF['상권명']==nearDistrict]
    timePopulation = districtSelected[['시간대_1_생활인구_수', '시간대_2_생활인구_수', '시간대_3_생활인구_수',
                                       '시간대_4_생활인구_수', '시간대_5_생활인구_수', '시간대_6_생활인구_수']].values.tolist()[0]

    return str(timePopulation)

@app.route('/getNearTimeSalesVolumeData')
def get_near_time_sales_volume_data():
    category = request.args.get('category')
    nearDistrict = request.args.get('nearDistrict')
    categorySelected = categoryDF[(categoryDF['상권명']==nearDistrict) & (categoryDF['업종명']==category)]
    try:
        timeSalesVolume = list(categorySelected[['시간대_00~06_매출_금액', '시간대_06~11_매출_금액', '시간대_11~14_매출_금액',
                                                 '시간대_14~17_매출_금액', '시간대_17~21_매출_금액', '시간대_21~24_매출_금액']].iloc[0])
    except:
        timeSalesVolume = []

    return str(timeSalesVolume)

@app.route('/getCprData')
def get_cpr_data():
    dong = request.args.get('dong')
    cprDistricts = list(addr[addr['행정동명']==dong]['상권_코드_명'].drop_duplicates())
    cprDistricts.sort()
    districtsMean = districtDF[districtDF['상권명'].isin(cprDistricts)].groupby('상권명').mean().drop(['연도','분기'], axis=1)
    districtsMeanStd = districtsMean.copy()
    districtsMeanStd = (districtsMeanStd / districtsMeanStd.mean() -1) * 100
    districtsMeanStd.loc['지역 평균'] = districtsMean.mean()
    districtsMeanStd = districtsMeanStd.dropna(axis=1)
    columns = list(districtsMeanStd.columns)
    districtsMeanStdJson = districtsMeanStd.to_json(orient='index', force_ascii=False)

    return str([columns, districtsMeanStdJson])

@app.route('/getCprSalesCat')
def get_cpr_sales_cat():
    dong = request.args.get('dong')
    cprDistricts = list(addr[addr['행정동명']==dong]['상권_코드_명'].drop_duplicates())
    cprDistricts.sort()
    categoriesMeanCat = list(np.sort(categoryDF[categoryDF['상권명'].isin(cprDistricts)]['업종명'].unique()))

    return str(categoriesMeanCat)

@app.route('/getCprSalesData')
def get_cpr_sales_data():
    dong = request.args.get('dong')
    category = request.args.get('category')
    cprDistricts = list(addr[addr['행정동명']==dong]['상권_코드_명'].drop_duplicates())
    cprDistricts.sort()

    categoriesMean = categoryDF[categoryDF['상권명'].isin(cprDistricts)].groupby(['상권명', '업종명']).mean().drop(['연도','분기'], axis=1)
    categoriesMeanStd = categoriesMean.copy()
    categoriesMeanStd = categoriesMeanStd.xs(category, level=1)
    categoriesMeanStd = (categoriesMeanStd / categoriesMeanStd.mean() -1) * 100
    categoriesMeanStd.loc['지역 평균'] = categoriesMean.mean()
    categoriesMeanStd = categoriesMeanStd.dropna(axis=1)
    catColumns = list(categoriesMeanStd.columns)
    categoriesMeanStdJson = categoriesMeanStd.to_json(orient='index', force_ascii=False)

    districtsMean = districtDF[districtDF['상권명'].isin(cprDistricts)].groupby('상권명').mean().drop(['연도','분기'], axis=1)
    districtsMeanStd = districtsMean.copy()
    districtsMeanStd = districtsMeanStd.loc[categoriesMeanStd.index[:-1]]
    districtsMeanStd = (districtsMeanStd / districtsMeanStd.mean() -1) * 100
    districtsMeanStd.loc['지역 평균'] = districtsMean.mean()
    districtsMeanStd = districtsMeanStd.dropna(axis=1)
    columns = list(districtsMeanStd.columns)
    districtsMeanStdJson = districtsMeanStd.to_json(orient='index', force_ascii=False)

    if len(categoriesMeanStd) <= 1:
        return "[0, 0, {}]"

    return str([columns, catColumns, districtsMeanStdJson, categoriesMeanStdJson])

@app.route('/map.html')
def map_page():
    global categories
    global dongList
    global districts
    categories = []
    dongList = []
    districts = []

    return render_template('map.html', title="Capstone", guList=guList, dongList=dongList, gu=gu, dong=dong,
                           districts=districts, categories=categories, district=district, category=category)

@app.route('/compare.html')
def compare():
    global categories
    global dongList
    global districts
    categories = []
    dongList = []
    districts = []

    return render_template('compare.html', title="Capstone", guList=guList, dongList=dongList, gu=gu, dong=dong,
                           districts=districts, categories=categories, district=district, category=category)

@app.route('/dashboard.html', methods=['POST', 'GET'])
def dashboard():
    if request.method == 'POST':
        pass

    elif request.method == 'GET':
        gu = request.args.get('gu')
        dong = request.args.get('dong')
        district = request.args.get('district')
        category = request.args.get('category')

        if gu == '구 선택' or gu is None or dong == '동 선택' or dong is None or district == '상권명 선택' or district is None or category == '업종명 선택' or category is None:
            return redirect(url_for('index'))

    districtSelected = districtDF[districtDF['상권명'] == district]
    categorySelected = categoryDF[(categoryDF['상권명'] == district) & (categoryDF['업종명'] == category)]
    recentCategoryDF = categoryDF[(categoryDF['상권명'] == district) & (categoryDF['연도'] == categoryDF['연도'].max())]
    recentCategoryDF = recentCategoryDF[recentCategoryDF['분기'] == recentCategoryDF['분기'].max()]

    #상권 연도별 총 유동인구
    floatingPopulation = list(districtSelected.groupby(['연도']).mean()['총_생활인구_수'])
    floatingPopIndex = [f"{i}년" for i in districtSelected.groupby(['연도']).mean().index]
    floatingPopChangeRatio = []
    floatingPopChangeRatioIndex = []
    for i in range(len(floatingPopulation)-1):
        diff = floatingPopulation[i+1] - floatingPopulation[i]
        floatingPopChangeRatio.append(round((diff/floatingPopulation[i])*100, 1))
        floatingPopChangeRatioIndex.append(f"{districtSelected.groupby(['연도']).mean().index[i]}년->{districtSelected.groupby(['연도']).mean().index[i]+1}년")

    #모든 기간 상권 총 유동인구
    wholeFloatingPopulation = list(districtSelected['총_생활인구_수'])
    wholeFloatingPopIndex = list()
    for year, quarter in zip(districtSelected['연도'], districtSelected['분기']):
        wholeFloatingPopIndex.append(f"{year}-{quarter}")

    wholeFloatingPopulation = list(reversed(wholeFloatingPopulation))
    wholeFloatingPopIndex = list(reversed(wholeFloatingPopIndex))

    #가장 최근 연령대별 유동인구
    ageGroupPopulation = districtSelected[['연령대_10_생활인구_수', '연령대_20_생활인구_수', '연령대_30_생활인구_수',
                                           '연령대_40_생활인구_수', '연령대_50_생활인구_수', '연령대_60_이상_생활인구_수']].values.tolist()[0]
    ageGroupPopIndex = ['10대', '20대', '30대', '40대', '50대', '60대 이상']

    #가장 최근 시간대별 유동인구
    timePopulation = districtSelected[['시간대_1_생활인구_수', '시간대_2_생활인구_수', '시간대_3_생활인구_수',
                                       '시간대_4_생활인구_수', '시간대_5_생활인구_수', '시간대_6_생활인구_수']].values.tolist()[0]
    timePopIndex = ['0~6시', '6~11시', '11~14시', '14~17시', '17~21시', '21~24시']

    #가장 최근 시간대별 매출 규모
    try:
        timeSalesVolume = list(categorySelected[['시간대_00~06_매출_금액', '시간대_06~11_매출_금액', '시간대_11~14_매출_금액',
                                                 '시간대_14~17_매출_금액', '시간대_17~21_매출_금액', '시간대_21~24_매출_금액']].iloc[0])
        timeSalesVolumeIndex = ['0~6시', '6~11시', '11~14시', '14~17시', '17~21시', '21~24시']
    except:
        timeSalesVolume = []
        timeSalesVolumeIndex = []

    ### 이거 돈 단위 *1000원 아닌가?

    #가장 최근 선택한 상권, 업종 분기별 총 매출 규모
    quarterSalesVolumeList = categorySelected[categorySelected['연도']==categorySelected['연도'].max()].sort_values(by='분기', ascending=True)
    quarterSalesVolume = list((quarterSalesVolumeList['분기당_매출_금액']).values)
    quarterSalesVolumeIndex = [f"{i}분기" for i in quarterSalesVolumeList['분기']]
    quarterSalesVolumeChangeRatio = []
    quarterSalesVolumeChangeRatioIndex = []
    for i in range(len(quarterSalesVolume)-1):
        diff = quarterSalesVolume[i+1] - quarterSalesVolume[i]
        quarterSalesVolumeChangeRatio.append(round((diff/quarterSalesVolume[i])*100, 1))
        quarterSalesVolumeChangeRatioIndex.append(f"{quarterSalesVolumeIndex[i]}->{quarterSalesVolumeIndex[i+1]}")

    #모든 기간 선택한 상권, 업종 총 매출 규모
    wholeQuarterSalesVolume = list(categorySelected['분기당_매출_금액'])
    wholeQuarterSalesVolumeIndex = list()
    for year, quarter in zip(categorySelected['연도'], categorySelected['분기']):
        wholeQuarterSalesVolumeIndex.append(f"{year}-{quarter}")

    wholeQuarterSalesVolume = list(reversed(wholeQuarterSalesVolume))
    wholeQuarterSalesVolumeIndex = list(reversed(wholeQuarterSalesVolumeIndex))

    #가장 최근 선택한 상권, 업종 분기별 점포당 평균 매출 규모
    #점포수 : 프랜차이즈 포함한 모든 점포 수
    #점포_수 : 프랜차이즈 제외한 모든 점포 수
    #프랜차이즈_점포_수
    marketAvgSalesVolume = list(quarterSalesVolumeList['분기당_매출_금액']/quarterSalesVolumeList['점포수'])
    marketAvgSalesVolumeIndex = quarterSalesVolumeIndex
    marketAvgSalesVolumeChangeRatio = []
    marketAvgSalesVolumeChangeRatioIndex = []
    for i in range(len(marketAvgSalesVolume)-1):
        diff = marketAvgSalesVolume[i+1] - marketAvgSalesVolume[i]
        marketAvgSalesVolumeChangeRatio.append(round((diff/marketAvgSalesVolume[i])*100, 1))
        marketAvgSalesVolumeChangeRatioIndex.append(f"{marketAvgSalesVolumeIndex[i]}->{marketAvgSalesVolumeIndex[i+1]}")

    #성별 매출 비율
    try:
        sexSalesVolumeRatio = list(categorySelected[['남성_매출_금액','여성_매출_금액']].iloc[0])
        sexSalesVolumeRatioIndex = ['남자', '여자']
    except:
        sexSalesVolumeRatio = []
        sexSalesVolumeRatioIndex = []

    #연령대별 매출 비율
    try:
        ageSalesVolumeRatio = list(categorySelected[['연령대_10_매출_금액', '연령대_20_매출_금액', '연령대_30_매출_금액', \
                                                     '연령대_40_매출_금액', '연령대_50_매출_금액', '연령대_60_이상_매출_금액']].iloc[0])
        ageSalesVolumeRatioIndex = ['10대', '20대', '30대', '40대', '50대', '60대 이상']
    except:
        ageSalesVolumeRatio = []
        ageSalesVolumeRatioIndex = []

    #프랜차이즈 비율
    try:
        franchiseRatio = list(categorySelected.iloc[0][['프랜차이즈_점포_수', '점포_수']].astype(int))
        franchiseRatioIndex = ['프랜차이즈 수', '비 프랜차이즈 수']

    except:
        franchiseRatio = []
        franchiseRatioIndex = []

    #가장 최근 업종 밀집도
    categoryDensity = list(pd.Series(list(recentCategoryDF['점포수']), index=list(recentCategoryDF['업종명'])).sort_index())
    categoryDensityIndex = list(pd.Series(list(recentCategoryDF['점포수']), index=list(recentCategoryDF['업종명'])).sort_index().index)

    #주차장
    try:
        countParking = 0
        districtCoor = [addr[addr['상권_코드_명']==district]['위도'].iloc[0], addr[addr['상권_코드_명'] == district]['경도'].iloc[0]]
        areaName = addr[addr['상권_코드_명']==district]['시군구명'].iloc[0]
        areaParking = parking[parking['주소']==areaName]
        for i in areaParking.index:
            parkingCoor = [areaParking.loc[i]['위도'], areaParking.loc[i]['경도']]
            if haversine(districtCoor, parkingCoor, unit='m') <= 500:
                countParking += 1
    except:
        pass

    #주변상권 찾기(5개)
    nearDistricts = list()
    for i in addr.index:
        nearCoor = [addr.loc[i]['위도'], addr.loc[i]['경도']]
        distance = haversine(districtCoor, nearCoor, unit='m')
        if (distance <= 1000) & (addr.loc[i]['상권_코드_명'] != district):
            nearDistricts.append((addr.loc[i]['상권_코드_명'], distance))
    nearDistricts = sorted(nearDistricts, key=lambda nearDistricts: nearDistricts[1])[:5]

    #가장 최근 집객시설 수
    countSchool = int(districtSelected.iloc[0][['초등학교_수', '중학교_수', '고등학교_수']].sum())
    countUniv = int(districtSelected.iloc[0]['대학교_수'])
    countTransport = int(districtSelected.iloc[0][['철도_역_수', '버스_터미널_수', '지하철_역_수', '버스_정거장_수']].sum())

    #서울 전체 상권 랭크
    global seoulRankList
    seoulRank = list(seoulRankList[category].keys()).index(district)+1
    seoulRankM1 = list(seoulRankList[category].keys()).index(district)
    seoulRankP1 = list(seoulRankList[category].keys()).index(district)+2
    seoulRankLast = len(seoulRankList[category])-2
    seoulRank1District = list(seoulRankList[category].keys())[0]
    seoulRankM1District = list(seoulRankList[category].keys())[seoulRankM1-1]
    seoulRankP1District = list(seoulRankList[category].keys())[seoulRankM1+1]
    seoulRankLastDistrict = list(seoulRankList[category].keys())[-3]
    seoulRankWholeDistrict = list(seoulRankList[category].keys())[:-2]
    try:
        seoulRankM1Score = round(list(seoulRankList[category].values())[seoulRankM1-1][-1], 2)
    except:
        seoulRankM1Score = -1
    seoulRankScore = round(list(seoulRankList[category].values())[seoulRankM1][-1], 2)
    seoulRankP1Score = round(list(seoulRankList[category].values())[seoulRankM1+1][-1], 2)
    seoulRankWholeScore = [round(seoulRankList[category][d][-1], 2) for d in list(seoulRankList[category].keys())[:-2]]

    #지역구 내 상권 랭크
    global guRankList

    guRank = list(guRankList[gu][category].keys()).index(district)+1
    guRankM1 = list(guRankList[gu][category].keys()).index(district)
    guRankP1 = list(guRankList[gu][category].keys()).index(district)+2
    guRankLast = len(guRankList[gu][category])-2
    guRank1District = list(guRankList[gu][category].keys())[0]
    guRankM1District = list(guRankList[gu][category].keys())[guRankM1-1]
    guRankP1District = list(guRankList[gu][category].keys())[guRankM1+1]
    guRankLastDistrict = list(guRankList[gu][category].keys())[-3]
    guRankWholeDistrict = list(guRankList[gu][category].keys())[:-2]
    try:
        guRankM1Score = round(list(guRankList[gu][category].values())[guRankM1-1][-1], 2)
    except:
        guRankM1Score = -1
    guRankScore = round(list(guRankList[gu][category].values())[guRankM1][-1], 2)
    guRankP1Score = round(list(guRankList[gu][category].values())[guRankM1+1][-1], 2)
    guRankWholeScore = [round(guRankList[gu][category][d][-1], 2) for d in list(guRankList[gu][category].keys())[:-2]]

    #서울 전체 상권 랭크 산정 기준
    seoulFeaturesRatio = seoulRankList[category]['weights']
    seoulFeatures = seoulRankList[category]['columns']

    #지역구 내 상권 랭크 산정 기준
    guFeaturesRatio = guRankList[gu][category]['weights']
    guFeatures = guRankList[gu][category]['columns']


    return render_template('dashboard.html', title="Capstone", guList=guList, dongList=dongList, gu=gu, dong=dong,
                           districts=districts, categories=categories,
                           district=district, category=category,
                           districtCategory=district+" "+category,
                           floatingPopulation=floatingPopulation, floatingPopIndex=floatingPopIndex,
                           floatingPopChangeRatio=floatingPopChangeRatio, floatingPopChangeRatioIndex=floatingPopChangeRatioIndex,
                           wholeFloatingPopulation=wholeFloatingPopulation, wholeFloatingPopIndex=wholeFloatingPopIndex,
                           ageGroupPopulation=ageGroupPopulation, ageGroupPopIndex=ageGroupPopIndex,
                           timePopulation=timePopulation, timePopIndex=timePopIndex,
                           timeSalesVolume=timeSalesVolume, timeSalesVolumeIndex=timeSalesVolumeIndex,
                           quarterSalesVolume=quarterSalesVolume, quarterSalesVolumeIndex=quarterSalesVolumeIndex,
                           quarterSalesVolumeChangeRatio=quarterSalesVolumeChangeRatio, quarterSalesVolumeChangeRatioIndex=quarterSalesVolumeChangeRatioIndex,
                           wholeQuarterSalesVolume=wholeQuarterSalesVolume, wholeQuarterSalesVolumeIndex=wholeQuarterSalesVolumeIndex,
                           marketAvgSalesVolume=marketAvgSalesVolume, marketAvgSalesVolumeIndex=marketAvgSalesVolumeIndex,
                           marketAvgSalesVolumeChangeRatio=marketAvgSalesVolumeChangeRatio, marketAvgSalesVolumeChangeRatioIndex=marketAvgSalesVolumeChangeRatioIndex,
                           sexSalesVolumeRatio=sexSalesVolumeRatio, sexSalesVolumeRatioIndex=sexSalesVolumeRatioIndex,
                           ageSalesVolumeRatio=ageSalesVolumeRatio, ageSalesVolumeRatioIndex=ageSalesVolumeRatioIndex,
                           franchiseRatio=franchiseRatio, franchiseRatioIndex=franchiseRatioIndex,
                           categoryDensity=categoryDensity, categoryDensityIndex=categoryDensityIndex,
                           nearDistricts=nearDistricts,
                           countParking=countParking, countSchool=countSchool, countUniv=countUniv, countTransport=countTransport,
                           guRank=guRank, guRankM1=guRankM1, guRankP1=guRankP1, guRankLast=guRankLast, guRank1District=guRank1District, guRankM1District=guRankM1District,
                           guRankP1District=guRankP1District, guRankLastDistrict=guRankLastDistrict, guRankM1Score=guRankM1Score, guRankScore=guRankScore, guRankP1Score=guRankP1Score,
                           seoulRank=seoulRank, seoulRankM1=seoulRankM1, seoulRankP1=seoulRankP1, seoulRankLast=seoulRankLast, seoulRank1District=seoulRank1District, seoulRankM1District=seoulRankM1District,
                           seoulRankP1District=seoulRankP1District, seoulRankLastDistrict=seoulRankLastDistrict, seoulRankM1Score=seoulRankM1Score, seoulRankScore=seoulRankScore, seoulRankP1Score=seoulRankP1Score,
                           guFeaturesRatio=guFeaturesRatio, guFeatures=guFeatures, seoulFeaturesRatio=seoulFeaturesRatio, seoulFeatures=seoulFeatures,
                           seoulRankWholeDistrict=seoulRankWholeDistrict, seoulRankWholeScore=seoulRankWholeScore, guRankWholeDistrict=guRankWholeDistrict, guRankWholeScore=guRankWholeScore
                           )

@app.route('/ref.html')
def ref():
    return render_template('ref.html', title="Capstone", districts=districts, categories=categories, district=district, category=category)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False)
