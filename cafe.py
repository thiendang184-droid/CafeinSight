import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings('ignore')

# CẤU HÌNH CỘT DỮ LIỆU
NUMERIC_RAW = ['gia_tb', 'khach_ngay', 'nhan_vien', 'dien_tich', 'doi_thu_500m', 'doanh_thu_ngay']
CATEGORICAL_RAW = ['vi_tri', 'co_cho_ngoi', 'co_delivery', 'cho_ngoi_lau']

FEATURES = [
    'gia_tb', 'khach_ngay', 'vi_tri', 'co_cho_ngoi', 'co_delivery',
    'nhan_vien', 'cho_ngoi_lau', 'dien_tich', 'doi_thu_500m',
    'gia_per_khach', 'hieu_suat_nv', 'dien_tich_per_nv', 'mat_do_doithu',
    'tiem_nang_giua_lai'
]

NUMERIC_COLS = ['gia_tb', 'khach_ngay', 'nhan_vien', 'dien_tich', 'doi_thu_500m',
                'gia_per_khach', 'hieu_suat_nv', 'dien_tich_per_nv', 'mat_do_doithu',
                'tiem_nang_giua_lai']

CATEGORICAL_COLS = ['vi_tri', 'co_cho_ngoi', 'co_delivery', 'cho_ngoi_lau']

# 1. LOAD & TIỀN XỬ LÝ
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


def load_and_preprocess(file_path: str = 'cafe_data.csv') -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Ép kiểu số
    for col in NUMERIC_RAW + CATEGORICAL_RAW:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Điền giá trị thiếu
    df[NUMERIC_RAW] = df[NUMERIC_RAW].fillna(df[NUMERIC_RAW].median())
    
    # Xử lý cột categorical an toàn hơn
    for col in CATEGORICAL_RAW:
        mode_val = df[col].mode()
        fill_value = mode_val.iloc[0] if not mode_val.empty else 0
        df[col] = df[col].fillna(fill_value)
    
    # Đảm bảo cột nhị phân là 0 hoặc 1
    for col in CATEGORICAL_RAW:
        df[col] = df[col].clip(0, 1).round().astype(int)   # ← Đã sửa ở đây
    
    # Loại outliers (chỉ trên numeric raw)
    for col in NUMERIC_RAW:
        if col == 'doanh_thu_ngay':  # Không loại outlier trên target
            continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    
    # Feature engineering
    df = feature_engineering(df)
    
    print(f" Dữ liệu sau lọc: {df.shape[0]} dòng × {df.shape[1]} cột")
    return df


# 2. XÂY DỰNG PIPELINE & HUẤN LUYỆN
def build_pipeline(model) -> Pipeline:
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), NUMERIC_COLS),
        ('cat', 'passthrough', CATEGORICAL_COLS),
    ])
    return Pipeline([('preprocessor', preprocessor), ('regressor', model)])


def train(df: pd.DataFrame) -> Pipeline:
    X = df[FEATURES]
    y = df['doanh_thu_ngay']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    candidates = {
        "Hồi quy tuyến tính (Linear Regression)": LinearRegression(),
        "Rừng ngẫu nhiên (Random Forest)": RandomForestRegressor(
            n_estimators=300, max_depth=10,
            min_samples_split=4, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
    }
    
    best_pipeline, best_r2, best_name = None, -np.inf, ""
    
    print("\n Kết quả mô hình:")
    
    for name, model in candidates.items():
        pipe = build_pipeline(model)
        cv_r2 = cross_val_score(pipe, X, y, cv=5, scoring='r2')
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n  {name}")
        print(f" CV R² : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
        print(f" MAE   : {mae:>15,.0f} VNĐ")
        print(f" RMSE  : {rmse:>15,.0f} VNĐ")
        print(f" Test R²: {r2:.4f}")
        
        if r2 > best_r2:
            best_r2, best_pipeline, best_name = r2, pipe, name
    
    print(f"\n Mô hình tốt nhất: {best_name} (R² = {best_r2:.4f})")
    joblib.dump(best_pipeline, 'best_revenue_model.pkl')
    print(" Đã lưu mô hình → best_revenue_model.pkl")
    return best_pipeline


# 3. NHẬP THÔNG TIN & DỰ ĐOÁN
def _ask(prompt: str, min_val, max_val, dtype=float):
    while True:
        try:
            v = dtype(input(prompt).strip())
            if min_val <= v <= max_val:
                return v
            print(f" Vui lòng nhập trong khoảng [{min_val:,} – {max_val:,}]")
        except ValueError:
            print(" Vui lòng nhập số hợp lệ!")


def predict(model: Pipeline):
    print("\n─── Nhập thông tin quán cà phê ───")
    raw = {
        'gia_tb': _ask(" Giá trung bình 1 ly (VNĐ) : ", 10000, 500000),
        'khach_ngay': _ask(" Số khách trung bình/ngày : ", 0, 2000, int),
        'vi_tri': _ask(" Vị trí đặc biệt: (1=Có, 0=Không) : ", 0, 1, int),
        'co_cho_ngoi': _ask(" Phục vụ tại chỗ? (1=Có, 0=Không) : ", 0, 1, int),
        'co_delivery': _ask(" Dịch vụ giao hàng? (1=Có, 0=Không) : ", 0, 1, int),
        'nhan_vien': _ask(" Số nhân viên: ", 1, 50, int),
        'cho_ngoi_lau': _ask(" Cho ngồi lâu? (1=Có, 0=Không) : ", 0, 1, int),
        'dien_tich': _ask(" Diện tích quán (m²) : ", 5, 1_000),
        'doi_thu_500m': _ask(" Số đối thủ cạnh tranh trong 500m : ", 0, 30, int),
    }
    
    df = pd.DataFrame([raw])
    df = feature_engineering(df)
    daily = model.predict(df[FEATURES])[0]
    daily = max(daily, 0)
    
    print("\n" + "═" * 45)
    print(" DỰ ĐOÁN DOANH THU")
    print("═" * 45)
    print(f" Ngày : {daily:>18,.0f} VNĐ")
    print(f" Tháng: {daily * 30:>18,.0f} VNĐ")
    print(f" Năm  : {daily * 365:>18,.0f} VNĐ")
    print("═" * 45)
    
    if daily >= 10_000_000:
        nhan_xet = " Rất tiềm năng! Quán có doanh thu cao."
    elif daily >= 5_000_000:
        nhan_xet = " Khá tốt. Quán hoạt động ổn định."
    elif daily >= 2_000_000:
        nhan_xet = " Trung bình. Cần cải thiện thêm."
    else:
        nhan_xet = " Thấp. Xem xét lại mô hình kinh doanh."
    
    print(f" → {nhan_xet}")
    print("═" * 45)


# 4. MAIN
if __name__ == "__main__":
    print(" DỰ ĐOÁN DOANH THU QUÁN CÀ PHÊ")
    print("=" * 45)
    
    df = load_and_preprocess('cafe_data.csv')
    model = train(df)
    
    while True:
        predict(model)
        cont = input("\n Dự đoán quán khác? (y/n): ").strip().lower()
        if cont != 'y':
            print("\n Bye")
            break