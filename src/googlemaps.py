# -*- coding: utf-8 -*-
import time
import traceback

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver import ChromeOptions as Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

GM_WEBPAGE = 'https://www.google.com/maps/'
MAX_WAIT = 10
MAX_RETRY = 5

class GoogleMapsScraper:

    def __init__(self, debug=False):
        self.debug = debug
        self.driver = self.__get_driver()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)

        self.driver.close()
        self.driver.quit()

        return True

    def sort_by(self, url, ind):
        self.driver.get(url)
        self.__click_on_cookie_agreement()

        wait = WebDriverWait(self.driver, MAX_WAIT)

        # open dropdown menu
        clicked = False
        tries = 0
        while not clicked and tries < MAX_RETRY:
            try:
                menu_bt = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[@data-value=\'Sort\']')))
                menu_bt.click()
                clicked = True
                time.sleep(3)
            except Exception as e:
                tries += 1
                print('Failed to click sorting button')

            # failed to open the dropdown
            if tries == MAX_RETRY:
                return -1

        sort_button = self.driver.find_elements(By.XPATH, '//div[@role=\'menuitemradio\']')[ind]
        sort_button.click()
        time.sleep(5)

        return 0

    def get_reviews(self, offset):
        """Scroll to load more reviews and then extract them"""
        self.__scroll()

        # wait for other reviews to load (ajax)
        time.sleep(4)
        self.__expand_reviews()

        # parse reviews
        response = BeautifulSoup(self.driver.page_source, 'html.parser')
        rblock = response.find_all('div', class_='jftiEf fontBodyMedium')
        parsed_reviews = []
        for index, review in enumerate(rblock):
            if index >= offset:
                r = self.__parse(review)
                parsed_reviews.append(r)

        return parsed_reviews

    def __parse(self, review):
        """Extract caption, rating, date, username, and user stats from review element"""
        item = {}

        try:
            review_text = review.find('span', class_='wiI7pd').text
            review_text = review_text.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
        except Exception as e:
            review_text = None

        try:
            rating = float(review.find('span', class_='kvMYJc')['aria-label'].split(' ')[0])
        except Exception as e:
            rating = None

        try:
            relative_date = review.find('span', class_='rsqaWe').text
        except Exception as e:
            relative_date = None

        try:
            username = review.find('div', class_='d4r55').text
        except Exception as e:
            username = 'Anonymous'

        try:
            review_count_text = review.find('div', class_='RfnDt').text
            n_review_user = int(review_count_text.split(' ')[0].replace(',', ''))
        except Exception as e:
            n_review_user = 1

        item['caption'] = review_text
        item['relative_date'] = relative_date
        item['rating'] = rating
        item['username'] = username
        item['n_review_user'] = n_review_user

        return item

    def __expand_reviews(self):
        """Expand 'More' buttons to show full review text"""
        buttons = self.driver.find_elements(By.CSS_SELECTOR,'button.w8nwRe.kyuRq')
        for button in buttons:
            self.driver.execute_script("arguments[0].click();", button)

    def __scroll(self):
        """Scroll to load more reviews"""
        scrollable_div = self.driver.find_element(By.CSS_SELECTOR,'div.m6QErb.DxyBCb.kA9KIf.dS8AEf')
        self.driver.execute_script('arguments[0].scrollTop = arguments[0].scrollHeight', scrollable_div)

    def __get_driver(self, debug=False):
        """Initialize Chrome WebDriver"""
        options = Options()

        if not self.debug:
            options.add_argument("--headless")
        else:
            options.add_argument("--window-size=1366,768")

        options.add_argument("--disable-notifications")
        options.add_argument("--accept-lang=en-GB")
        input_driver = webdriver.Chrome(service=Service(), options=options)

        input_driver.get(GM_WEBPAGE)

        return input_driver

    def __click_on_cookie_agreement(self):
        """Reject cookies if dialog appears"""
        try:
            agree = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//span[contains(text(), "Reject all")]')))
            agree.click()
        except:
            pass


def clean_reviews(df):
    """Clean review data and add calculated columns"""
    df['caption'] = df['caption'].fillna('')
    df['has_text'] = df['caption'].str.len() > 0
    df['text_length'] = df['caption'].str.len()
    
    print(f"Total reviews: {len(df)}")
    print(f"Reviews with text: {df['has_text'].sum()}, Rating-only: {(~df['has_text']).sum()}")
    return df
