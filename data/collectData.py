# pip install app_store_scraper

from app_store_scraper import AppStore
import pandas as pd
import numpy as np
import json

tiktok = AppStore(country="us", app_name="tiktok")
tiktok.review(how_many=1500)
print(tiktok.reviews)

df = pd.DataFrame(np.array(tiktok.reviews),columns=['review'])
df2 = df.join(pd.DataFrame(df.pop('review').tolist()))
df2.head()

df2.to_csv('tiktok.csv')