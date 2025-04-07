from google_play_scraper import reviews_all, Sort
import pandas as pd
import time
import random
import traceback

# ID aplikasi di Play Store
app_id = 'com.deepseek.chat'

# Daftar kode bahasa dan kode negara yang ingin dicoba
lang_country_list = [
    ('en', 'us')
]
# Menyimpan semua ulasan yang berhasil di-scrape
all_reviews = []

# Pengaturan delay untuk menghindari pemblokiran
delay = 1.2
max_delay = 30

for lang, country in lang_country_list:
    print(f"Scraping untuk {lang}-{country}...")

    try:
        reviews = reviews_all(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,  # Urutkan berdasarkan ulasan terbaru
            sleep_milliseconds=0  # Tidak ada jeda tambahan antar permintaan internal
        )

        print(f"{len(reviews)} ulasan ditemukan untuk {lang}-{country}")

        if isinstance(reviews, list):
            all_reviews.extend(reviews)
        else:
            print(f"Respons format tidak sesuai: {type(reviews)}")

        # Pola delay yang tidak repetitif (meningkat secara eksponensial)
        time.sleep(delay + random.uniform(0.1, 0.5))
        delay = min(delay * 1.5, max_delay)

    except Exception as e:
        print(f"Error saat scraping {lang}-{country}: {e}")
        traceback.print_exc()
        # Meningkatkan delay sebelum mencoba ulang untuk menghindari pemblokiran
        time.sleep(min(delay, max_delay))
        delay = min(delay * 1.5, max_delay)

# Konversi ke DataFrame
df = pd.DataFrame(all_reviews)
df = df[['content', 'score']]
# Simpan ke file CSV
csv_file = 'data/dataset_reviews.csv'
df.to_csv(csv_file, index=False, encoding='utf-8')

print(f'\n✅ Total {len(df)} ulasan disimpan ke {csv_file}')
