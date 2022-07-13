from secrets import choice
import numpy as np
import pandas as pd
from underthesea import word_tokenize
#import regex
#import demoji
import emoji
#from pyvi import ViPosTagger, ViTokenizer
import string
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split  
# from sklearn.metrics import accuracy_score 
# from sklearn.metrics import confusion_matrix
# from sklearn. metrics import classification_report, roc_auc_score, roc_curve
import pickle
import streamlit as st
# import matplotlib.pyplot as plt
# from sklearn import metrics
# import seaborn as sns
from langdetect import detect
import re
#from emoticons import EMOTICONS_EMO

# # 1. Read data
# file = pd.read_csv('Files/bank_app_08072022.csv')
# data = pd.read_csv('Files/data.csv', index_col = 0)
# df_en = pd.read_csv('Files/df_en.csv', encoding='utf-8')
#--------------
# GUI
st.title("Data Science Project")
st.write("## Sentiment Analysis")

# 2. Data pre-processing
EMOTICONS_EMO = {
    u")))":"cười to", # lỗi chính tả
    u"))":"cười", # lỗi chính tả
    u":)))":"cười to",
    u":)":"cười",
    u":]]]":"cười to",
    u":]":"cười",
    u"=]]]":"cười to",
    u"=]":"cười",
    u"=)))":"cười to",
    u"=)":"cười",
    u":D":"cười",
    u"XD":"cười to",
    u":3":"cười",
    u":>":"cười",
    u":(":"buồn",
    u":((":"rất buồn",
    u":<":"buồn",
    u">_<":"nhăn mặt",
    u"><":"nhăn mặt",
    u"-_-":"chán",
    u"=_=":"chán",
    u"==":"chán",
    u"._.":"bối rối",
    u"-.-":"bối rối",
    u"@@":"bối rối",
    u"^_^":"vui",
    u"^_^":"vui",
    u"^O^":"vui",
    u"^o^":"vui",
    u"T_T":"khóc",
    u"ToT":"khóc",
    u";_;":"khóc",
    u";-;":"khóc",
    u"T.T":"khóc",
    u"TT":"khóc",
    u"^^":"cười",
    u"^—^":"cười",
    u":O":"bất ngờ",
    u"o.O":"bất ngờ",
    u"o.o":"bất ngờ",
    u"oO":"bất ngờ",
    u"<3":"yêu thích"
}

final_ko_list = ['không có', 'không nhanh', 'không được', 'không thích', 'không bao giờ', 'không hiểu', 'không chậm', 'không nên', 'không hợp', 'không nhiều', 'không vào được', 'không gửi',
                 'không cập nhật', 'không đúng', 'không thân thiện', 'không dễ', 'không phù hợp', 'không cho', 'không ra gì', 'không xảy ra', 'không mở', 'không nhập được',
                 'không rẻ', 'không thuận tiện', 'không kém', 'không hài lòng', 'không phải', 'không thể', 'không nhận được', 'không xong', 'không tìm được', 'không ra', 'không đắt']

def check_is_en(text):
    try:
        language = detect(text)
    except:
        language = "error"
    return (language == 'en')

def convert_emoticons(text, EMOTICONS_EMO):
    for emot in EMOTICONS_EMO:
        text = re.sub(u'('+re.escape(emot)+')', "_".join(EMOTICONS_EMO[emot].replace(",","").split()), text)
    return text

def convert_emoji(text, emojicon_dict):
    for emoj in emojicon_dict:
        text = text.replace(emoj, "_".join(emojicon_dict[emoj].split())+" ") 
        # thêm space do emoji hay để cạnh nhau
    return text
def convert_teencode(text, teencode_dict):
    for word in teencode_dict:
        text = re.sub(u'(^|[.,:;\s])'+re.escape(word)+'(?=[.,\s]|$)', " "+"_".join(teencode_dict[word].split()), text)
    return text
def remove_special_icons(text):
  for icon in ['\n', ',', '.', '...', '-', ';', '?', '%', '+', '/', '\\', '—', '_', '*', '#', '<', '>', '|', '&', '=', '+', '~', '!', '(', ')', ':', '@', '^']:
    text = text.replace(icon, ' ')
  return text
def convert_khong_words(text, final_ko_list):
  for word in final_ko_list:
    text = re.sub(u'('+re.escape(word)+')', "_".join(word.split()), text)
  return text
def clean_text(text):
  text = re.sub(r'\d', ' ', text) # bỏ số
  text = re.sub(r'(\s)\1+', ' ', text) # bỏ khoảng trắng liên tục
  return text.strip().lower() # bỏ khoảng trắng đầu cuối và in thường tất cả

# import emoji dictionary
emojicon_dict = {}
emojicon_file = open("Files/emojicon.txt", encoding='utf-8')
for line in emojicon_file:
    key, value = line.split('\t')
    emojicon_dict[key] = value.replace('\n', '')
# import file teencode
teencode_dict = {}
teencode_file = open("Files/teencode.txt", encoding='utf-8')
for line in teencode_file:
    key, value = line.split('\t')
    teencode_dict[key] = value.replace('\n', '')

def Load_Object(filename):
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
    return obj

my_model_vnm = Load_Object('Files/bank_app_LogisticRegression_vnm.pkl')
count_vnm = Load_Object('Files/Count_tfidf_vnm.pkl')
def xu_ly(text):
  text = convert_emoticons(text, EMOTICONS_EMO)
  text = convert_emoji(text, emojicon_dict)
  text = convert_teencode(text, teencode_dict)
  text = remove_special_icons(text)
  text = convert_khong_words(text, final_ko_list)
  text = clean_text(text)
  text = word_tokenize(text, format='text')
  return text

def predict(x_new):
  x_new = xu_ly(x_new)
  x_new = count_vnm.transform(np.array([x_new])).toarray()
  y_prob = my_model_vnm.predict_proba(x_new)
  if y_prob[0][0] >= 0.5:
    reply = 'Không thích/ Dislike'
  else:
    reply = 'Thích/ Like'
  return reply
my_model_eng = Load_Object('Files/bank_app_LogisticRegression_eng.pkl')
count_eng = Load_Object('Files/Count_tfidf_eng.pkl')
def process(text):
  text = emoji.demojize(text)
  text = remove_special_icons(text)
  text = clean_text(text)
  return text
def predict_eng(x_new):
  x_new = process(x_new)
  x_new = count_eng.transform(np.array([x_new])).toarray()
  y_prob = my_model_eng.predict_proba(x_new)
  if y_prob[0][0] >= 0.5:
    reply = 'Không thích/ Dislike'
  else:
    reply = 'Thích/ Like'
  return reply

def predict_reviews(text):
  if check_is_en(text):
    result = predict_eng(text)
  else:
    result = predict(text)
  return result

stop_words_file = 'Files/vietnamese-stopwords.txt'
with open(stop_words_file, 'r', encoding='utf-8') as file:
  stop_words = file.read()

stop_words = stop_words.split('\n')

#df_vi = pd.read_csv('Files/processed_data.csv')


menu = ['Business Objective', 'New Prediction']
# 'Build Project', 
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Business Objective':
    st.subheader('Business Objective')
    st.write("""
- Apple Appstore & Google Play Store là hai chợ nổi tiếng nhất dành cho các ứng dụng di động. Apple và Android cũng là hai hệ điều hành phổ biến nhất hiện nay.
- Trên chợ người dùng có thể đưa ra các đánh giá về các app dành cho các nhà phát triển
- Các nhà phát triển app ngân hàng mong muốn có thể phân loại được các đánh giá này để từ đó cải thiện chất lượng của app
- Mục tiêu của dự án trước mắt là phân loại được các đánh giá của các app ngân hàng tại Việt Nam

    """)  
    st.write("""###### => Problem/ Requirement: Xây dựng model để dự đoán những đánh giá của Khách hàng là Thích hay Không thích app ngân hàng.""")
#    st.image("hdbank.png")
# elif choice == 'Build Project':
#     st.subheader('Build Project')
#     # st.write('##### 1. Some data')
#     # st.dataframe(file.head(3))
#     # st.dataframe(file.tail(3))
#     st.code('data.shape')
# #    st.code(file.shape)
#     st.write('###### Xóa dữ liệu trùng/NaN/null')
#     st.code('data = data.drop_duplicates()')
#     st.code('data=data.dropna()')
#     st.code('data.shape')
# #    st.code(data.shape)
#     st.write('Dữ liệu gồm 17533 records (reviews) cho 29 bank apps')
#     st.write("###### Top 5 bank apps có nhiều reviews nhất")
# #    st.image("Files/many_reviews.jpg")
#     st.write('''###### Rating = 1 chiếm nhiều nhất với hơn 7645 reviews, số lượng lớn thứ 2 là rating = 5 với 6524 reviews, các điểm rating từ 2-4 chiếm số lượng xấp xỉ nhau khoảng 1000 reviews''')
# #    st.image("Files/review_score.jpg")
#     st.write('top 5 app có điểm trung bình cao nhất')
# #    st.image("Files/best_avg.jpg")
#     st.write('top 5 app có điểm trung bình thấp nhất')
# #    st.image("Files/worst_avg.jpg")
#     st.write('###### Tạo cột english với review_text tiếng Anh là 1, còn lại là 0')
#     st.code("data['english'] = data.review_text.apply(lambda x: 1 if check_is_en(x) else 0)")
#     st.code("df_en = data[data.english == 1]")
#     st.code("df_en.shape")
# #    st.code(df_en.shape)
#     st.write('## Xây dựng model với nhận xét Tiếng Việt:')
#     st.code("df_vi = data[data.english == 0]")
#     st.code("df_vi.shape")
#     st.write('#### (14837, 4)')
#     st.write('###### Gắn nhãn cho dữ liệu với review_score < 4 là label 0 (Dislike), review_score >= 4 là label 1 (Like)')
#     st.code("df_vi['target'] = df_vi.review_score.apply(lambda x: 1 if x >= 4 else 0)")
#     st.write("###### Chuyển đổi dữ liệu: chuyển emoji, emoticons, teencode, xóa bỏ ký tự đặc biệt, nhóm từ chứa 'không'...")
#     st.code("""df_vi.review_text = df_vi.review_text.apply(lambda x: convert_emoticons(x, EMOTICONS_EMO))
#             df_vi.review_text = df_vi.review_text.apply(lambda x: convert_emoji(x, emojicon_dict))
#             df_vi.review_text = df_vi.review_text.apply(lambda x: convert_teencode(x, teencode_dict))
#             df_vi.review_text = df_vi.review_text.apply(lambda x: remove_special_icons(x))
#             df_vi.review_text = df_vi.review_text.apply(lambda x: convert_khong_words(x, final_ko_list))
#             df_vi.review_text = df_vi.review_text.apply(lambda x: clean_text(x))""")
# #    st.image("Files/chuyen_doi_text.jpg")
#     st.write('###### Xóa bỏ các dòng có review_text trống:')
#     st.code("df_vi = df_vi[df_vi.review_text !=  '']")
#     st.code("df_vi = df_vi[df_vi.review_text !=  ' ']")
#     st.code('df.shape')
#     st.code('(14831, 5)')
#     st.write('###### Tokenize review_text để tạo cột review_text_wt kết nối các từ ghép:')
#     st.code("df_vi['review_text_wt'] = df_vi['review_text'].apply(lambda x: word_tokenize(x, format='text'))")
# #    st.image("word_tokenize.jpg")  
#     st.write('''####
#     - lấy ngẫu nhiên x_train, x_test, y_train, y_test với tỷ lệ 70:30
#     - Sử dụng TF-idf với 877 từ tác động tới biến target để tạo array cho x_train, transform cho x_test:''')
#     st.write('####Đánh giá với một vài thuật toán phân lớp, lựa chọn LR để xây dựng model')
# #    st.image("Files/danh_gia.jpg")
#     st.write('#### Sau khi huấn luận với Logistic Regression và tuning parameters, lựa chọn Logistic Regression mặc định để dự đoán vì cho kết quả khá tốt')
# #    st.image("Files/diem_lr_vnm.jpg")
# #    st.image("Files/vnm_lr.jpg")
# #    st.image("Files/auc_vnm.jpg")
    
#     st.write('###### Visualize Text:')
# #    st.image("Files/spacy_vnm.jpg")
#     st.write("""
#     - Class Dislike thường xuyên xuất hiện các từ tệ, quá tệ, chậm, chán, báo lỗi, bắt, không thể, ...;
#     - Class Like thường xuyên xuất hiện các từ tiện lợi, thích, dễ, nhanh chóng, tuyệt vời,...
#     """)
#     st.write('## Xây dựng model với nhận xét Tiếng Anh:')
#     st.write('#### Thực hiện tương tự như trên và lựa chọn LogisticRegression (C = 10) để dự đoán')
# #    st.image("Files/diem_lr_eng.jpg")
    
#     st.write('###### Visualize Text:')
# #    st.image("Files/spacy_eng.jpg")
#     st.write("""
#     - Class Dislike thường xuyên xuất hiện các từ bad, can not, terrible, slow, too, stupid, dificult, ...;
#     - Class Like thường xuyên xuất hiện các từ very usefull, very convenient, nice, best, love,...
#     """)
elif choice == 'New Prediction':
    st.subheader('Select data')
    flag = False
    lines = None
    type = st.radio('Upload data or Input data?', options = ('Upload', 'Input'))
    if type == 'Upload':
        # Upload file
        uploaded_file_1 = st.file_uploader('Choose a file', type = ['csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1, encoding='utf-8')
            st.dataframe(lines)
            st.write(lines.columns)
            lines = lines['review_text']
            content = lines.copy()
            lines = lines.tolist()
            flag = True
    if type == 'Input':
        review = st.text_area(label = 'Input your content:')
        if review != '':
            content = review[:]
            lines = [review]
            flag = True

    if flag:
        st.write('Content:')
        if (len(lines) >0):
            count = 1
            for line in lines:
                st.write(str(count) + '. ' + line)
                st.code(predict_reviews(line))
                count += 1
