import sys
from time import sleep, strftime
import csv

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import ElementNotInteractableException
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By

###############
# based on the code from: https://hoho325.tistory.com/268
###############

sys.stdout.reconfigure(encoding='utf-8')

# 옵션
csv_out = False
csv_out = True
options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('lang=ko_KR')
driver = webdriver.Chrome(options=options)  # chromedriver 열기

p_data = {
    'name': '',
    'addr': '',
    'avgr': '',
}

def main():
    global driver, load_wb, review_num, wr

    if csv_out:
        f = open(f'km-reviews-{strftime("%Y%m%d_%H%M%S")}.csv', 'w', encoding='utf-8-sig', newline='')
        wr = csv.writer(f)
        wr.writerow(['name', 'location', 'avg_rate', 'review_text', 'rate'])

    driver.implicitly_wait(4)
    driver.get('https://map.kakao.com/')

    # 검색어 리스트
    place_infos = ['성균관대학교 자연과학캠퍼스 맛집']

    for i, place in enumerate(place_infos): 
        # delay
        if i % 4 == 0 and i != 0:
            sleep(5)
        print("#####", i)
        search(place)

    driver.quit()
    print("finish")


def search(place):
    global driver

    search_area = driver.find_element(By.XPATH, '//*[@id="search.keyword.query"]')  # 검색 창
    search_area.send_keys(place)  # 검색어 입력
    driver.find_element(By.XPATH, '//*[@id="search.keyword.submit"]').send_keys(Keys.ENTER)  # Enter로 검색
    sleep(1)

    # 검색된 정보가 있는 경우에만 탐색
    # 1번 페이지 place list 읽기
    html = driver.page_source

    soup = BeautifulSoup(html, 'html.parser')
    place_lists = soup.select('.placelist > .PlaceItem') # 검색된 장소 목록

    # 검색된 첫 페이지 장소 목록 크롤링하기
    crawling(place, place_lists)
    search_area.clear()

    # 우선 더보기 클릭해서 2페이지
    try:
        driver.find_element(By.XPATH, '//*[@id="info.search.place.more"]').send_keys(Keys.ENTER)
        sleep(1)

        # 2~ 5페이지 읽기
        for i in range(2, 6):
            # 페이지 넘기기
            xPath = '//*[@id="info.search.page.no' + str(i) + '"]'
            driver.find_element(By.XPATH, xPath).send_keys(Keys.ENTER)
            sleep(1)

            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            place_lists = soup.select('.placelist > .PlaceItem') # 장소 목록 list

            crawling(place, place_lists)

        # 그 이후 페이지
        while True:
            driver.find_element(By.XPATH, '//*[@id="info.search.page.next"]').send_keys(Keys.ENTER)
            sleep(1)

            for i in range(2, 6):
                # 페이지 넘기기
                xPath = '//*[@id="info.search.page.no' + str(i) + '"]'
                driver.find_element(By.XPATH, xPath).send_keys(Keys.ENTER)
                sleep(1)

                html = driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                place_lists = soup.select('.placelist > .PlaceItem') # 장소 목록 list

                crawling(place, place_lists)

    # except (NoSuchElementException, ElementNotInteractableException):
    except ElementNotInteractableException:
        print('not found')
    finally:
        search_area.clear()


def crawling(keyword, place_lists):

    for i, place in enumerate(place_lists):
        
        p_data['name'] = place.select('.head_item > .tit_name > .link_name')[0].text  # place name
        p_data['addr'] = place.select('.info_item > .addr > p')[0].text  # place address
        p_data['avgr'] = place.select('.score > .num')[0].text  # place avg rating
        
        print('\n####', p_data['name'])
        print(p_data['addr'], '별점: ', p_data['avgr'])

        # 후기 미제공 식당 건너뛰기
        if len(place.select('.score.HIDDEN'))!=0:
            print("후기 미제공")
            # wr.writerow([place_name, place_address, "후기 미제공", ''])
            continue

        detail_page_xpath = '//*[@id="info.search.place.list"]/li[' + str(i + 1) + ']/div[4]/span[1]/a'  # 상세정보 탭으로 변환
        driver.find_element(By.XPATH, detail_page_xpath).send_keys(Keys.ENTER)
        driver.switch_to.window(driver.window_handles[-1])  # 후기 탭으로 변환
        sleep(1)

        # 무한 스크롤
        prev_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            sleep(1)
            curr_height = driver.execute_script("return document.body.scrollHeight")
            if prev_height == curr_height:
                break
            prev_height = curr_height
        
        extract_review()

        driver.close()
        driver.switch_to.window(driver.window_handles[0])  # 검색 탭으로 전환


def extract_review():
    global driver

    ret = True

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    # 첫 페이지 리뷰 목록 찾기
    review_lists = soup.select('.list_review > li')

    print('리뷰 수:', len(review_lists))

    # 리뷰가 있는 경우
    if len(review_lists) != 0:
        for i, review in enumerate(review_lists):
            comment = review.select('.desc_review') # 리뷰
            rating = review.select('.starred_grade > span') # 별점
            val = ''
            if len(comment) != 0:
                review_text = comment[0].text
                rate = '0'
                if len(rating) != 0:
                    rate = rating[1].text.replace('점', '')
                    val = review_text + '/' + rate
                else:
                    val = review_text + '/0'
                print(val)
                if csv_out:
                    wr.writerow([p_data['name'], p_data['addr'], p_data['avgr'], review_text, rate])

    else:
        print('no review in extract')
        ret = False

    return ret


if __name__ == "__main__":
    main()