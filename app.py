import streamlit as st
from PIL import Image  # Thư viện dùng để đọc file ảnh làm icon
import pandas as pd
import joblib
import time
import base64

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN & APP ICON (BẮT BUỘC PHẢI ĐỂ ĐẦU TIÊN)
# ==========================================
logo_path = "logo.png"

try:
    # Sử dụng chính tấm logo.png làm Page Icon hiển thị trên trình duyệt và thiết bị di động
    app_icon = Image.open(logo_path)
except Exception:
    app_icon = "☕"  # Backup nếu lỗi không tìm thấy file ảnh

# Thiết lập cấu hình trang (Lệnh Streamlit đầu tiên)
st.set_page_config(
    page_title="CafeinSight", 
    page_icon=app_icon, 
    layout="centered"
)

# ==========================================
# 2. HÀM FEATURE ENGINEERING & ĐƯỜNG DẪN FILE
# ==========================================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['gia_per_khach'] = df['gia_tb'] / (df['khach_ngay'] + 1)
    df['hieu_suat_nv'] = df['khach_ngay'] / (df['nhan_vien'] + 1)
    df['dien_tich_per_nv'] = df['dien_tich'] / (df['nhan_vien'] + 1)
    df['mat_do_doithu'] = df['doi_thu_500m'] / (df['dien_tich'] + 1)
    
    df['tiem_nang_giua_lai'] = (
        df['vi_tri'] * 2 +
        df['co_cho_ngoi'] +
        df['co_delivery'] +
        df['cho_ngoi_lau']
    )
    return df

FEATURES = [
    'gia_tb', 'khach_ngay', 'vi_tri', 'co_cho_ngoi', 'co_delivery',
    'nhan_vien', 'cho_ngoi_lau', 'dien_tich', 'doi_thu_500m',
    'gia_per_khach', 'hieu_suat_nv', 'dien_tich_per_nv', 'mat_do_doithu',
    'tiem_nang_giua_lai'
]

bg_path = "istockphoto-1822889082-612x612.png"

@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# ==========================================
# 3. CUSTOM CSS: HÌNH NỀN, ẨN MENU, ĐỔI MÀU
# ==========================================
try:
    img_base64 = get_base64_of_bin_file(bg_path)
    custom_css = f"""
    <style>
    /* Ẩn footer "Made with Streamlit" để giữ giao diện sạch đẹp */
    footer {{visibility: hidden;}}
    
    /* Hình nền */
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    [data-testid="stHeader"], [data-testid="stAppViewContainer"] > .main {{
        background: rgba(0,0,0,0);
    }}

    /* Khung nền mờ */
    [data-testid="stForm"], .st-emotion-cache-1wivap2, .st-emotion-cache-12w0qpk {{
        background-color: rgba(255, 255, 255, 0.9) !important;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }}

    /* Chữ to toàn bộ */
    html, body, [class*="css"], .stMarkdown p, label, div[data-testid="stMarkdownContainer"] p {{
        font-size: 18px !important; 
    }}
    
    /* Tiêu đề */
    h1 {{ font-size: 24px !important; text-align: center; }} 
    h3 {{ font-size: 1.8rem !important; }}

    /* Nút bấm vàng */
    .stButton > button {{
        font-size: 20px !important; 
        font-weight: bold !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        transition: 0.2s;
    }}
    .stButton > button:hover, .stButton > button:active, .stButton > button:focus {{
        border-color: #FFC107 !important;
        color: #FFC107 !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }}

    /* Ô nhập liệu & Radio vàng */
    div[data-baseweb="input"]:hover, div[data-baseweb="input"]:focus-within {{
        border-color: #FFC107 !important;
        box-shadow: none !important;
    }}
    div[data-baseweb="input"] {{
        font-size: 18px !important;
    }}
    .stRadio > div {{
        filter: hue-rotate(50deg) saturate(150%);
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
except Exception as e:
    st.error(f"Lỗi tải CSS hoặc hình nền: {e}")

# --- CHÈN LOGO CĂN GIỮA ---
try:
    logo_base64 = get_base64_of_bin_file(logo_path)
    st.markdown(
        f'<div style="text-align: center; margin-bottom: -10px;">'
        f'<img src="data:image/png;base64,{logo_base64}" width="280">'
        f'</div>',
        unsafe_allow_html=True
    )
except FileNotFoundError:
    st.warning("⚠️ Không tìm thấy file logo.png")

# --- TIÊU ĐỀ CHÍNH ---
st.title("Ước Tính Doanh Thu Quán Cà Phê")
st.markdown("---")

# ==========================================
# 4. NHẬP LIỆU GIAO DIỆN
# ==========================================
with st.container():
    st.markdown("### 📋 Nhập thông tin quán của bạn")
    col1, col2 = st.columns(2)

    with col1:
        gia_tb = st.number_input("💵 Giá trung bình 1 ly (VNĐ)", min_value=10000, max_value=500000, value=35000, step=5000)
        khach_ngay = st.number_input("👥 Số khách trung bình/ngày", min_value=0, max_value=2000, value=100, step=10)
        nhan_vien = st.number_input("🧑‍🍳 Số nhân viên", min_value=1, max_value=50, value=3)
        dien_tich = st.number_input("📐 Diện tích quán (m²)", min_value=5, max_value=1000, value=50, step=5)
        doi_thu_500m = st.number_input("⚔️ Số đối thủ trong bán kính 500m", min_value=0, max_value=30, value=2)

    with col2:
        vi_tri_str = st.radio("📍 Vị trí đặc biệt (mặt tiền, ngã tư)?", ["Không", "Có"])
        co_cho_ngoi_str = st.radio("🪑 Có không gian ngồi lại?", ["Không", "Có"], index=1)
        co_delivery_str = st.radio("🛵 Có giao hàng (App)?", ["Không", "Có"], index=1)
        cho_ngoi_lau_str = st.radio("💻 Cho khách ngồi lâu (làm việc)?", ["Không", "Có"], index=1)

# Chuyển đổi map_dict
map_dict = {"Có": 1, "Không": 0}
vi_tri = map_dict[vi_tri_str]
co_cho_ngoi = map_dict[co_cho_ngoi_str]
co_delivery = map_dict[co_delivery_str]
cho_ngoi_lau = map_dict[cho_ngoi_lau_str]

st.markdown("---")

# ==========================================
# 5. XỬ LÝ NÚT DỰ ĐOÁN & POP-UP KẾT QUẢ
# ==========================================
if st.button("🚀 PHÂN TÍCH DOANH THU", use_container_width=True):
    try:
        # Hiệu ứng pop-up toast loading
        st.toast('Đang khởi động AI...', icon='🤖')
        time.sleep(0.5)
        st.toast('Đang thu thập dữ liệu thị trường...', icon='📊')
        time.sleep(0.5)
        
        # Load mô hình
        model = joblib.load('best_revenue_model.pkl')
        
        raw_data = {
            'gia_tb': gia_tb, 'khach_ngay': khach_ngay, 'vi_tri': vi_tri,
            'co_cho_ngoi': co_cho_ngoi, 'co_delivery': co_delivery,
            'nhan_vien': nhan_vien, 'cho_ngoi_lau': cho_ngoi_lau,
            'dien_tich': dien_tich, 'doi_thu_500m': doi_thu_500m,
        }
        
        df_input = pd.DataFrame([raw_data])
        df_input = feature_engineering(df_input)
        
        daily_revenue = model.predict(df_input[FEATURES])[0]
        daily_revenue = max(daily_revenue, 0)
        
        # Hiệu ứng bóng bay
        st.balloons()
        st.toast('Tính toán thành công!', icon='🎉')
        
        # Hiển thị kết quả trong khung đẹp mắt
        st.success("💰 **KẾT QUẢ DỰ ĐOÁN**")
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Doanh thu / Ngày 🪙", f"{daily_revenue:,.0f} đ")
        col_res2.metric("Doanh thu / Tháng 💵", f"{daily_revenue * 30:,.0f} đ")
        col_res3.metric("Doanh thu / Năm 📈", f"{daily_revenue * 365:,.0f} đ")
        
        # Nhận xét
        st.markdown("### 🔍 Phân Tích Chuyên Gia")
        if daily_revenue >= 10_000_000:
            st.info("🌟 **Rất tiềm năng!** Mô hình kinh doanh đang mang lại dòng tiền xuất sắc. Hãy cân nhắc mở rộng quy mô.")
        elif daily_revenue >= 5_000_000:
            st.success("✅ **Khá tốt.** Quán hoạt động ổn định và có lãi. Giữ vững phong độ nhé!")
        elif daily_revenue >= 2_000_000:
            st.warning("⚠️ **Trung bình.** Bạn nên chạy thêm các chương trình khuyến mãi hoặc tối ưu chi phí nguyên liệu.")
        else:
            st.error("🆘 **Rủi ro cao.** Cần xem xét lại menu, giá bán hoặc cắt giảm nhân sự gấp.")
            
    except FileNotFoundError:
        st.error("⚠️ Lỗi: Không tìm thấy file `best_revenue_model.pkl`.")
    except Exception as e:
        st.error(f"⚠️ Đã xảy ra lỗi: {e}")
