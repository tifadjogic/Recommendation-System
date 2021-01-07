import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
import numpy as np
import base64
import io
from matplotlib.pyplot import imread
import codecs
from scipy.spatial import distance
import operator

courses= pd.read_csv('Coursera_courses_catalog.csv')
print(courses.head())
print(courses.info())
print(courses.describe())

plt.subplots(figsize=(12,10))
list1 = []
for i in courses['university_name']:
    list1.append(i)
ax = pd.Series(list1).value_counts()[:20].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('hls',10))
for i, v in enumerate(pd.Series(list1).value_counts()[:20].sort_values(ascending=True).values): 
    ax.text(.8, i, v,fontsize=12,color='white',weight='bold')
plt.title('Top Universities')


uniList = []
for uni in courses['university_name']:
    if uni not in uniList:
        uniList.append(uni)

def binary(_list,List):
    binaryList = []
    
    for i in List:
        if i in _list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    
    return binaryList

courses['uni_bin'] = courses['university_name'].apply(lambda x: binary(x,uniList))
print (courses['uni_bin'].head())

plt.subplots(figsize=(12,10))
list2 = []
for i in courses['course_type']:
    list2.append(i)
ax = pd.Series(list2).value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('hls',10))
for i, v in enumerate(pd.Series(list2).value_counts()[:10].sort_values(ascending=True).values): 
    ax.text(.8, i, v,fontsize=12,color='white',weight='bold')
plt.title('Top Course Types')


typeList = []
for typ in courses['course_type']:
    if typ not in typeList:
        typeList.append(typ)

courses['type_bin'] = courses['course_type'].apply(lambda x: binary(x,typeList))
print (courses['type_bin'].head())

plt.subplots(figsize=(12,10))
list3 = []
for i in courses['course_language']:
    list3.append(i)
ax = pd.Series(list3).value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('hls',10))
for i, v in enumerate(pd.Series(list3).value_counts()[:10].sort_values(ascending=True).values): 
    ax.text(.8, i, v,fontsize=12,color='white',weight='bold')
plt.title('Course Languages')


langList = []
for lang in courses['course_language']:
    if lang not in langList:
        langList.append(lang)

courses['lang_bin'] = courses['course_language'].apply(lambda x: binary(x,langList))
print (courses['lang_bin'].head())

plt.subplots(figsize=(12,10))
list4 = []
for i in courses['category']:
    list4.append(i)
ax = pd.Series(list4).value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('hls',10))
for i, v in enumerate(pd.Series(list4).value_counts()[:10].sort_values(ascending=True).values): 
    ax.text(.8, i, v,fontsize=12,color='white',weight='bold')
plt.title('Course Categories')


catList = []
for cat in courses['category']:
    if cat not in catList:
        catList.append(cat)

courses['cat_bin'] = courses['category'].apply(lambda x: binary(x,catList))
print (courses['cat_bin'].head())

plt.subplots(figsize=(12,10))
list5 = []
for i in courses['sub_category']:
    list5.append(i)
ax = pd.Series(list5).value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('hls',10))
for i, v in enumerate(pd.Series(list3).value_counts()[:10].sort_values(ascending=True).values): 
    ax.text(.8, i, v,fontsize=12,color='white',weight='bold')
plt.title('Subcategories')


subList = []
for subc in courses['sub_category']:
    if subc not in subList:
        subList.append(subc)

courses['sub_bin'] = courses['sub_category'].apply(lambda x: binary(x,subList))
print (courses['sub_bin'].head())

plt.subplots(figsize=(12,10))
list6 = []
for i in courses['course_level']:
    list6.append(i)
ax = pd.Series(list6).value_counts()[:6].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('hls',10))
for i, v in enumerate(pd.Series(list6).value_counts()[:6].sort_values(ascending=True).values): 
    ax.text(.8, i, v,fontsize=12,color='white',weight='bold')
plt.title('Course Level')
plt.show()

levelList = []
for level in courses['course_level']:
    if level not in levelList:
        levelList.append(level)

courses['level_bin'] = courses['course_level'].apply(lambda x: binary(x,levelList))
print (courses['level_bin'].head())

def Similarity(cid1, cid2):
    a = courses.iloc[cid1]
    b = courses.iloc[cid2]
    
    uniA = a['uni_bin']
    uniB = b['uni_bin']
    
    uniDistance = distance.cosine(uniA, uniB)
    
    typeA = a['type_bin']
    typeB = b['type_bin']
    typeDistance = distance.cosine(typeA, typeB)
    
    langA = a['lang_bin']
    langB = b['lang_bin']
    langDistance = distance.cosine(langA, langB)
    
    catA = a['cat_bin']
    catB = b['cat_bin']
    catDistance = distance.cosine(catA, catB)

    subA = a['sub_bin']
    subB = b['sub_bin']
    subDistance = distance.cosine(subA, subB)

    levelA = a['level_bin']
    levelB = b['level_bin']
    levelDistance = distance.cosine(levelA, levelB)

    return uniDistance + typeDistance + langDistance + catDistance + subDistance + levelDistance

new_id = list(range(0,courses.shape[0]))
courses['new_id']=new_id

def recommend():
    name = input('Enter a course title: ')
    new_course = courses[courses['course_name'].str.contains(name)].iloc[0].to_frame().T
    print('Selected Course: ',new_course.course_name)
    def getNeighbors(baseCourse, K):
        distances = []
    
        for index, course in courses.iterrows():
            if course['new_id'] != baseCourse['new_id'].values[0]:
                dist = Similarity(baseCourse['new_id'].values[0], course['new_id'])
                distances.append((course['new_id'], dist))
    
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
    
        for x in range(K):
            neighbors.append(distances[x])
        return neighbors

    K = 10
    neighbors = getNeighbors(new_course, K)
    print('\nRecommended Courses: \n')
    for neighbor in neighbors:  
        print( courses.iloc[neighbor[0]][1])
    
    print('\n')
recommend()