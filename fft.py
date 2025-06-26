import pandas as pd
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
df = pd.read_excel("Acoustic_Extinguisher_Fire_Dataset.xlsx", sheet_name="A_E_Fire_Dataset")

# 2. FFT 적용 대상 데이터 (AIRFLOW 컬럼)
airflow_data = df['AIRFLOW'].values

# 3. FFT 수행
fft_values = fft(airflow_data)                   # 복소수 결과
fft_magnitudes = np.abs(fft_values)              # 진폭 (Magnitude)

# 4. 주파수 축 계산 (상대 주파수)
n = len(airflow_data)
freqs = np.fft.fftfreq(n)

# 5. 결과를 데이터프레임으로 정리
fft_df = pd.DataFrame({
    'Frequency': freqs,
    'Magnitude': fft_magnitudes
})

# 6. 결과 정렬 (내림차순)
fft_df_sorted = fft_df.sort_values(by='Magnitude', ascending=False).reset_index(drop=True)

# 7. 시각화 (선택)
plt.figure(figsize=(10, 5))
plt.plot(freqs[:n//2], fft_magnitudes[:n//2])  # 양의 주파수만 시각화
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.title('FFT of AIRFLOW')
plt.grid(True)
plt.show()
