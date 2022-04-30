############# python 3.6.9 #############
import json 
from urllib.request import urlopen
from bs4 import BeautifulSoup


########################################################################
#################경남 청년 일자리 사업 공지사항 크롤링##################
########################################################################


html = urlopen("https://gnjobs.kr/html/sub06_02.php")  
soup = BeautifulSoup(html, "html.parser") 
hotKeys = soup.select("a.list_tit")
TITLE=[]

#####게시글 제목#####
for key in hotKeys:
    TITLE.append(key.text.strip())

#####게시글 URL#####
title = soup.find_all(class_='list_tit')
URL=[]
for i in title:
    URL.append(i.attrs['href'])


#####긁어온 코드 도시별로 구분#####
products=[]
for i,j in zip(URL,TITLE):
    k=i.split('¤') #url을 긁었을때 저 이상한 문자가 같이 긁혀서 지우는 코드
    
    """창원시"""
    if j.rfind('창원')>0:
        if j.rfind('사업장')==-1 and j.rfind('업체')==-1 and j.rfind('기업')==-1   or j.rfind('청년 모집')>0 or j.rfind('청년구직')>0:
            print("창원시 :  "+k[0]+"   "+j)
            product = {"region": '경남',"city": '창원', "title": j, "link": k[0]}
            products.append(product)
    
    """진주시"""
    if j.rfind('진주')>0:
        if j.rfind('사업장')==-1 and j.rfind('업체')==-1 and j.rfind('기업')==-1   or j.rfind('청년 모집')>0 or j.rfind('청년구직')>0:
            print("진주시 :  "+k[0]+"   "+j)
            product = {"region": '경남',"city": '진주', "title": j, "link": k[0]}
            products.append(product)
    
    """통영시"""
    if j.rfind('통영')>0:
        if j.rfind('사업장')==-1 and j.rfind('업체')==-1 and j.rfind('기업')==-1   or j.rfind('청년 모집')>0 or j.rfind('청년구직')>0:
            print("통영시 :  "+k[0]+"   "+j)
            product = {"region": '경남',"city": '통영', "title": j, "link": k[0]}
            products.append(product)
    
    """사천시"""
    if j.rfind('사천')>0:
        if j.rfind('사업장')==-1 and j.rfind('업체')==-1 and j.rfind('기업')==-1   or j.rfind('청년 모집')>0 or j.rfind('청년구직')>0:
            print("사천시 :  "+k[0]+"   "+j)
            product = {"region": '경남',"city": '사천', "title": j, "link": k[0]}
            products.append(product)
    
    """김해시"""
    if j.rfind('김해')>0:
        if j.rfind('사업장')==-1 and j.rfind('업체')==-1 and j.rfind('기업')==-1   or j.rfind('청년 모집')>0 or j.rfind('청년구직')>0:
            print("김해시 :  "+k[0]+"   "+j)
            product = {"region": '경남',"city": '김해', "title": j, "link": k[0]}
            products.append(product)
    """밀양시"""
    if j.rfind('밀양')>0:
        if j.rfind('사업장')==-1 and j.rfind('업체')==-1 and j.rfind('기업')==-1   or j.rfind('청년 모집')>0 or j.rfind('청년구직')>0:
            print("밀양시 :  "+k[0]+"   "+j)
            product = {"region": '경남',"city": '밀양', "title": j, "link": k[0]}
            products.append(product)
    """거제시"""
    if j.rfind('거제')>0:
        if j.rfind('사업장')==-1 and j.rfind('업체')==-1 and j.rfind('기업')==-1   or j.rfind('청년 모집')>0 or j.rfind('청년구직')>0:
            print("거제시 :  "+k[0]+"   "+j)
            product = {"region": '경남',"city": '거제', "title": j, "link": k[0]}
            products.append(product)

    """양산시"""
    if j.rfind('양산')>0:
        if j.rfind('사업장')==-1 and j.rfind('업체')==-1 and j.rfind('기업')==-1   or j.rfind('청년 모집')>0 or j.rfind('청년구직')>0:
            print("양산시 :  "+k[0]+"   "+j)
            product = {"region": '경남',"city": '양산', "title": j, "link": k[0]}
            products.append(product)

    """의령군"""
    if j.rfind('의령')>0:
        if j.rfind('사업장')==-1 and j.rfind('업체')==-1 and j.rfind('기업')==-1   or j.rfind('청년 모집')>0 or j.rfind('청년구직')>0:
            print("의령군 :  "+k[0]+"   "+j)
            product = {"region": '경남',"city": '의령', "title": j, "link": k[0]}
            products.append(product)

    """함안군"""
    if j.rfind('함안')>0:
        if j.rfind('사업장')==-1 and j.rfind('업체')==-1 and j.rfind('기업')==-1   or j.rfind('청년 모집')>0 or j.rfind('청년구직')>0:
            print("함안군 :  "+k[0]+"   "+j)
            product = {"region": '경남',"city": '함안', "title": j, "link": k[0]}
            products.append(product)

    """창녕군"""
    if j.rfind('창녕')>0:
         if j.rfind('사업장')==-1 and j.rfind('업체')==-1 and j.rfind('기업')==-1   or j.rfind('청년 모집')>0 or j.rfind('청년구직')>0:
            print("창녕군 :  "+k[0]+"   "+j)
            product = {"region": '경남',"city": '창녕', "title": j, "link": k[0]}
            products.append(product)

    """고성군"""
    if j.rfind('고성')>0:
        if j.rfind('사업장')==-1 and j.rfind('업체')==-1 and j.rfind('기업')==-1   or j.rfind('청년 모집')>0 or j.rfind('청년구직')>0:
            print("고성군 :  "+k[0]+"   "+j)
            product = {"region": '경남',"city": '고성', "title": j, "link": k[0]}
            products.append(product)

    """남해군"""
    if j.rfind('남해')>0:
        if j.rfind('사업장')==-1 and j.rfind('업체')==-1 and j.rfind('기업')==-1   or j.rfind('청년 모집')>0 or j.rfind('청년구직')>0:
            print("남해군 :  "+k[0]+"   "+j)
            product = {"region": '경남',"city": '남해', "title": j, "link": k[0]}
            products.append(product)

    """하동군"""
    if j.rfind('하동')>0:
        if j.rfind('사업장')==-1 and j.rfind('업체')==-1 and j.rfind('기업')==-1   or j.rfind('청년 모집')>0 or j.rfind('청년구직')>0:
            print("하동군 :  "+k[0]+"   "+j)
            product = {"region": '경남',"city": '하동', "title": j, "link": k[0]}
            products.append(product)

    """산청군"""
    if j.rfind('산청')>0:
        if j.rfind('사업장')==-1 and j.rfind('업체')==-1 and j.rfind('기업')==-1   or j.rfind('청년 모집')>0 or j.rfind('청년구직')>0:
            print("산청군 :  "+k[0]+"   "+j)
            product = {"region": '경남',"city": '산청', "title": j, "link": k[0]}
            products.append(product)

    """함양군"""
    if j.rfind('함양')>0:
        if j.rfind('사업장')==-1 and j.rfind('업체')==-1 and j.rfind('기업')==-1   or j.rfind('청년 모집')>0 or j.rfind('청년구직')>0:
            print("함양군 :  "+k[0]+"   "+j)
            product = {"region": '경남',"city": '함양', "title": j, "link": k[0]}
            products.append(product)

    """거창군"""
    if j.rfind('거창')>0:
        if j.rfind('사업장')==-1 and j.rfind('업체')==-1 and j.rfind('기업')==-1   or j.rfind('청년 모집')>0 or j.rfind('청년구직')>0:
            print("거창군 :  "+k[0]+"   "+j)
            product = {"region": '경남',"city": '거창', "title": j, "link": k[0]}
            products.append(product)

    """합천군"""
    if j.rfind('합천')>0:
        if j.rfind('사업장')==-1 and j.rfind('업체')==-1 and j.rfind('기업')==-1   or j.rfind('청년 모집')>0 or j.rfind('청년구직')>0:
            print("합천군 :  "+k[0]+"   "+j)
            product = {"region": '경남',"city": '합천', "title": j, "link": k[0]}
            products.append(product)
    
    
    





####################################################################
####################부산 청년 일자리 지원 사이트####################
####################################################################

pshtml = urlopen("https://www.busan.go.kr/young/young007")  
soup = BeautifulSoup(pshtml, "html.parser") 
pshotKeys = soup.select("td > a")

'''제목 긁어오기'''
psTITLE=[]
for key in pshotKeys:
    print(key.text.strip())
    psTITLE.append(key.text.strip())


'''URL 긁어오기'''
pstitle = soup.select("td > a")
psURL=[]
for i in pstitle:
    url="https://www.busan.go.kr"+i.attrs['href']
    psURL.append(url)
    print(url)    # 링크 주소('href') 부분만 출력
    print()

'''필요한 데이터만 뽑아서 json형태로 저장'''
for i,j in zip(psURL,psTITLE):
  if j.rfind('사업장')==-1 and j.rfind('업체')==-1 and j.rfind('기업')==-1 or j.rfind('청년 모집')>0 or j.rfind('청년구직')>0 or j.rfind('참여청년')>0:
            print("부산 :  "+i+"   "+j)
            product = {"region": '부산', "title": j, "link": i}
            products.append(product)  


############################################################
##################울산 청년 일자리 지원 사이트##############
############################################################
#사이트1
''' 제목 크롤링'''
ushtml = urlopen("https://www.ulsan.go.kr/u/rep/bbs/list.ulsan?bbsId=BBS_0000000000000003&mId=001004001001000000")  
soup = BeautifulSoup(ushtml, "html.parser") 
ushotKeys = soup.select("td > a")
usTITLE=[]
for key in ushotKeys:
    print(key.text.strip())
    usTITLE.append(key.text.strip())


''' 링크 크롤링'''
ustitle = soup.select("td > a")
usURL=[]
for i in ustitle:
    url="https://www.ulsan.go.kr/u/rep/bbs"+i.attrs['href'][1:]
    usURL.append(url)
    print(url)    # 링크 주소('href') 부분만 출력
    print()


#사이트2
''' 제목2 크롤링'''
us2html = urlopen("http://yhf.kr/dongnam/training/trainList.do?category_cd=T1004")  
soup2 = BeautifulSoup(us2html, "html.parser") 
us2hotKeys = soup2.select("li > a > p")
us2TITLE=[]
for key in us2hotKeys:
    print(key.text.strip())
    us2TITLE.append(key.text.strip())


''' 링크2 크롤링'''
us2title = soup2.select("div > div > ul > li > a")
us2URL=[]
for i in us2title:
    url="http://yhf.kr"+i.attrs['href']
    us2URL.append(url)
    print(url)    # 링크 주소('href') 부분만 출력
    print()


''' jason 형태로 저장'''
#사이트1
print('사이트1')
for i,j in zip(usURL,usTITLE):
  if j.rfind('청년')>0 and j.rfind('사업장')==-1 and j.rfind('업체')==-1 and j.rfind('기업')==-1 or j.rfind('청년 모집')>0 or j.rfind('청년구직')>0 or j.rfind('참여청년')>0:
            print("울산 :  "+i+"   "+j)
            product = {"region": '울산', "title": j, "link": i}
            products.append(product)
print('사이트2')
#사이트2
for i,j in zip(us2URL,us2TITLE):
    print("울산 :  "+i+"   "+j)
    product = {"region": '울산', "title": j, "link": i}
    products.append(product)


############################################################
##################대구 청년 일자리 지원 사이트##############
############################################################
''' 제목 크롤링'''
dghtml = urlopen("http://www.dgyouth.kr/board/notice.asp")  
soup = BeautifulSoup(dghtml, "html.parser") 
dghotKeys = soup.select("dd > a > div > em")
dgTITLE=[]
for key in dghotKeys:
    print(key.text.strip())
    dgTITLE.append(key.text.strip())

''' 링크 크롤링'''
dgtitle = soup.select("dl > dd > a")
dgURL=[]
for i in dgtitle:
    url="http://www.dgyouth.kr/board/"+i.attrs['href']
    dgURL.append(url)
    print(url)    # 링크 주소('href') 부분만 출력
    print()


''' jason 형태로 저장'''
for i,j in zip(dgURL,dgTITLE):
  if j.rfind('청년')>0 and j.rfind('사업장')==-1 and j.rfind('업체')==-1 and j.rfind('기업')==-1 and j.rfind('합격자')==-1 and j.rfind('행정예고')==-1 and j.rfind('최종')==-1 or j.rfind('청년 모집')>0 or j.rfind('청년구직')>0 or j.rfind('참여청년')>0:
            print("대구 :  "+i+"   "+j)
            product = {"region": '대구', "title": j, "link": i}
            products.append(product)



############################################################
##################서울 청년 일자리 지원 사이트##############
############################################################
''' 제목 크롤링'''
suhtml = urlopen("https://youth.seoul.go.kr/site/main/board/notice/list?baCategory1=basic&baCommSelec=true")  
soup = BeautifulSoup(suhtml, "html.parser") 
suhotKeys = soup.select("tbody > tr > td > a")
suTITLE=[]
for key in suhotKeys:
    print(key.text.strip())
    suTITLE.append(key.text.strip())

''' 링크 크롤링'''
sutitle = soup.select("tbody > tr > td > a")
suURL=[]
for i in sutitle:
    url="https://youth.seoul.go.kr"+i.attrs['href']
    suURL.append(url)
    print(url)    # 링크 주소('href') 부분만 출력
    print()

''' jason 형태로 저장'''
for i,j in zip(suURL,suTITLE):
  k=j.split("\r\n\t\t\t\t\t\t\t\t\t\t\t\t")
  if j.rfind('청년')>0 and j.rfind('사업장')==-1 and j.rfind('업체')==-1 and j.rfind('기업')==-1 and j.rfind('합격자')==-1 and j.rfind('행정예고')==-1 and j.rfind('최종')==-1 or j.rfind('청년 모집')>0 or j.rfind('청년구직')>0 or j.rfind('참여청년')>0:
            print("서울 :  "+i+"   "+k[0])
            product = {"region": '서울', "title": k[0], "link": i}
            products.append(product)





print("########################################################################")
print("################################최종 json###############################")
print("########################################################################")
for i in products:
  print(i)



'''json 형태 확인'''
for i in products:
  print(i)

'''json 으로 저장'''
file = open("./data.json","w+")
file.write(json.dumps(products))


#'''json 열어보기'''
#with open('/content/data.json','r') as g:
#    json_data = json.load(g)
#print(json.dumps(json_data,indent="\t"))
