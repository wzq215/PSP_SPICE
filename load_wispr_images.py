import requests
import datetime as dt

'''
打开网站https://sppgway.jhuapl.edu/wispr_img，选择某一天的图片并加载，看一下这一天的图片的时间分辨率
将第一张图的时间修改为beg_time_str，并把两张图片的时间间隔填入下面的delta_time_minute和delta_time_sec变量
如果时间分辨率会变则还需修改该时间分辨率对应的end_time_str
根据所需日期修改对应的orbit，date，camera字符串
'''
#setting
orbit = 'orbit08'
# date = '20210111'
camera = 'inner'
beg_time_str = '20210426102421'
delta_time_minute = 15
delta_time_sec = 0
end_time_str = '20210426235721'
# download
time_beg = dt.datetime.strptime(beg_time_str,'%Y%m%d%H%M%S')
time_end = dt.datetime.strptime(end_time_str,'%Y%m%d%H%M%S')
url_base = 'https://sppgway.jhuapl.edu/rTools/WISPRN/pngs/L3/'
temp_time = time_beg
delta_time = dt.timedelta(minutes=delta_time_minute,seconds=delta_time_sec)
while temp_time<=time_end:
    print(temp_time)
    temp_time_str = dt.datetime.strftime(temp_time,'%Y%m%dT%H%M%S')
    temp_png_name = 'psp_L3_wispr_'+temp_time_str+'_V1_1221.png'
    temp_day = temp_time_str[0:8]
    url_for_dl = url_base+orbit+'/'+camera+'/'+temp_day+'/'+temp_png_name
    #url_for_dl = 'https://sppgway.jhuapl.edu/rTools/WISPRN/pngs/L3/orbit07/inner/20210115/psp_L3_wispr_20210115T172021_V1_1221.png'
    r =requests.get(url_for_dl)
    if str(r)!='<Response [404]>':
        with open('psp/wispr/images/'+camera+'/'+temp_png_name,'wb')as f:
            f.write(r.content)
            f.close
            print('done!')
    temp_time = temp_time+delta_time