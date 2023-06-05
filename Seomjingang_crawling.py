#-*-coding:utf-8-*-
import time
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd

driver = webdriver.Chrome(ChromeDriverManager().install())
driver.maximize_window()
driver.implicitly_wait(5)

# URL='http://datasvc.nmsc.kma.go.kr/datasvc/html/data/selectData.do?selectType=file&detailSq=3038&satUid=GK2A'
URL = 'http://datasvc.nmsc.kma.go.kr/datasvc/html/data/selectData.do?selectType=file&detailSq=3022&satUid=GK2A'
driver.get(URL)

#시간 타입 설정
driver.find_element_by_xpath("//select[@name='timeType']/option[text()='KST']").click()
driver.find_element_by_xpath("//*[@id='observationStartDt']").click()

#달력 초기 설정
cal = driver.find_element_by_name("observationStartDt")
cal.click()

for i in range(22):
    cal_pre_month_btn = driver.find_element_by_xpath("//*[@id='ui-datepicker-div']/div/a[1]")
    cal_pre_month_btn.click()

cal_year=driver.find_element_by_class_name("ui-datepicker-year")
cal_month = driver.find_element_by_class_name("ui-datepicker-month")
year_txt = cal_year.text
month_txt = cal_month.text
login = 0
temp = 0
week = [1,2,3,4,5,6]
day = [1,2,3,4,5,6,7]
while(year_txt !="2021" and month_txt != "8월"):
    for w in week:
        for d in day:
            try:
                print(w, d)
                cal.click()
                day_select = driver.find_element_by_xpath("//*[@id='ui-datepicker-div']/table/tbody/tr[{}]/td[{}]/a".format(w,d))
                day_select.click()
                search_btn = driver.find_element_by_id("dataSearchBtn")
                search_btn.click()
                time.sleep(10)


                load_num1 = driver.find_element_by_class_name("center")
                load_num2 = load_num1.find_element_by_tag_name('span')
                load_nums = load_num2.text.split('/')
                print(load_nums[0][1:], load_nums[1][:-1])
                while(load_nums[0][1:] != load_nums[1][:-1]):
                    load_more = load_num1.find_element_by_tag_name('input')
                    load_more.click()
                    time.sleep(1)
                    load_num2 = load_num1.find_element_by_tag_name('span')
                    load_nums = load_num2.text.split('/')
                    print(load_nums[0][1:], load_nums[1][:-1])

                select_all_btn = driver.find_element_by_class_name("nmsc_check")
                select_all_btn.click()

                request_data_btn = driver.find_element_by_class_name("nmsc_button.nmsc_button_style04.nmsc_button_radius.insertBtn")
                request_data_btn.click()

                if (login == 0):
                    id_txt = driver.find_element_by_id("userId")
                    id_txt.send_keys("")
                    pw_txt = driver.find_element_by_id("tempPassword")
                    pw_txt.send_keys("")
                    login_btn = driver.find_element_by_id("loginBtn")
                    login_btn.click()
                    login = 1
                    request_data_btn = driver.find_element_by_class_name(
                        "nmsc_button.nmsc_button_style04.nmsc_button_radius.insertBtn")
                    request_data_btn.click()
                time.sleep(1)
                confirm_btn = driver.find_element_by_class_name("lnv-btn-dialog.primary.confirm-btn")
                confirm_btn.click()
                time.sleep(1)
                ok_btn = driver.find_element_by_class_name("lnv-btn-dialog.primary.alert-btn")
                ok_btn.click()
                time.sleep(1)
                cancel_btn = driver.find_element_by_class_name("lnv-btn-dialog.default.cancel-btn")
                cancel_btn.click()
                time.sleep(1)
                cal.click()
            except NoSuchElementException as e:
                if(w == 1):
                    continue
                if(w == 4 or w == 5 or w == 6):
                    cal.click()
                    cal_next_month_btn = driver.find_element_by_xpath("//*[@id='ui-datepicker-div']/div/a[2]")
                    cal_next_month_btn.click()
                    cal_month = driver.find_element_by_class_name("ui-datepicker-month")
                    year_txt = cal_year.text
                    month_txt = cal_month.text
                    time.sleep(2)
                    temp = 1
                    break
        if(temp == 1):
            break