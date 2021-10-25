```python
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
#import necessary libraries for performing prediction
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
```


```python
Boston_list = pd.read_csv("D:/Download/DS_class/Code_To_Practice/Boston_Airbnb/listings.csv")
Boston_cld = pd.read_csv("D:/Download/DS_class/Code_To_Practice/Boston_Airbnb/calendar.csv")
Boston_review = pd.read_csv("D:/Download/DS_class/Code_To_Practice/Boston_Airbnb/reviews.csv")
```


```python
import os
for dirname, _, filenames in os.walk('D:/Download/DS_class/Code_To_Practice'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

    D:/Download/DS_class/Code_To_Practice\AllTogether.py
    D:/Download/DS_class/Code_To_Practice\AllTogetherSolns.py
    D:/Download/DS_class/Code_To_Practice\Clean_data_Model.py
    D:/Download/DS_class/Code_To_Practice\CMF_Model.py
    D:/Download/DS_class/Code_To_Practice\Code_To_Practice.iml
    D:/Download/DS_class/Code_To_Practice\DataVisualization.py
    D:/Download/DS_class/Code_To_Practice\survey_results_public.csv
    D:/Download/DS_class/Code_To_Practice\survey_results_schema.csv
    D:/Download/DS_class/Code_To_Practice\.idea\.gitignore
    D:/Download/DS_class/Code_To_Practice\.idea\misc.xml
    D:/Download/DS_class/Code_To_Practice\.idea\modules.xml
    D:/Download/DS_class/Code_To_Practice\.idea\runConfigurations.xml
    D:/Download/DS_class/Code_To_Practice\.idea\workspace.xml
    D:/Download/DS_class/Code_To_Practice\.idea\codeStyles\codeStyleConfig.xml
    D:/Download/DS_class/Code_To_Practice\.idea\codeStyles\Project.xml
    D:/Download/DS_class/Code_To_Practice\Boston_Airbnb\calendar.csv
    D:/Download/DS_class/Code_To_Practice\Boston_Airbnb\listings.csv
    D:/Download/DS_class/Code_To_Practice\Boston_Airbnb\reviews.csv
    D:/Download/DS_class/Code_To_Practice\Seattle_Airbnb\calendar.csv
    D:/Download/DS_class/Code_To_Practice\Seattle_Airbnb\listings.csv
    D:/Download/DS_class/Code_To_Practice\Seattle_Airbnb\reviews.csv
    D:/Download/DS_class/Code_To_Practice\__pycache__\AllTogether.cpython-37.pyc
    D:/Download/DS_class/Code_To_Practice\__pycache__\AllTogetherSolns.cpython-37.pyc
    


```python
for data in [Boston_list, Boston_cld, Boston_review]:
    print(data.shape)
    print(data.head(5))
```

    (3585, 95)
             id                            listing_url       scrape_id  \
    0  12147973  https://www.airbnb.com/rooms/12147973  20160906204935   
    1   3075044   https://www.airbnb.com/rooms/3075044  20160906204935   
    2      6976      https://www.airbnb.com/rooms/6976  20160906204935   
    3   1436513   https://www.airbnb.com/rooms/1436513  20160906204935   
    4   7651065   https://www.airbnb.com/rooms/7651065  20160906204935   
    
      last_scraped                                           name  \
    0   2016-09-07                     Sunny Bungalow in the City   
    1   2016-09-07              Charming room in pet friendly apt   
    2   2016-09-07               Mexican Folk Art Haven in Boston   
    3   2016-09-07  Spacious Sunny Bedroom Suite in Historic Home   
    4   2016-09-07                            Come Home to Boston   
    
                                                 summary  \
    0  Cozy, sunny, family home.  Master bedroom high...   
    1  Charming and quiet room in a second floor 1910...   
    2  Come stay with a friendly, middle-aged guy in ...   
    3  Come experience the comforts of home away from...   
    4  My comfy, clean and relaxing home is one block...   
    
                                                   space  \
    0  The house has an open and cozy feel at the sam...   
    1  Small but cozy and quite room with a full size...   
    2  Come stay with a friendly, middle-aged guy in ...   
    3  Most places you find in Boston are small howev...   
    4  Clean, attractive, private room, one block fro...   
    
                                             description experiences_offered  \
    0  Cozy, sunny, family home.  Master bedroom high...                none   
    1  Charming and quiet room in a second floor 1910...                none   
    2  Come stay with a friendly, middle-aged guy in ...                none   
    3  Come experience the comforts of home away from...                none   
    4  My comfy, clean and relaxing home is one block...                none   
    
                                   neighborhood_overview  ... review_scores_value  \
    0  Roslindale is quiet, convenient and friendly. ...  ...                 NaN   
    1  The room is in Roslindale, a diverse and prima...  ...                 9.0   
    2  The LOCATION: Roslindale is a safe and diverse...  ...                10.0   
    3  Roslindale is a lovely little neighborhood loc...  ...                10.0   
    4  I love the proximity to downtown, the neighbor...  ...                10.0   
    
      requires_license license jurisdiction_names instant_bookable  \
    0                f     NaN                NaN                f   
    1                f     NaN                NaN                t   
    2                f     NaN                NaN                f   
    3                f     NaN                NaN                f   
    4                f     NaN                NaN                f   
    
      cancellation_policy require_guest_profile_picture  \
    0            moderate                             f   
    1            moderate                             f   
    2            moderate                             t   
    3            moderate                             f   
    4            flexible                             f   
    
      require_guest_phone_verification calculated_host_listings_count  \
    0                                f                              1   
    1                                f                              1   
    2                                f                              1   
    3                                f                              1   
    4                                f                              1   
    
       reviews_per_month  
    0                NaN  
    1               1.30  
    2               0.47  
    3               1.00  
    4               2.25  
    
    [5 rows x 95 columns]
    (1308890, 4)
       listing_id        date available price
    0    12147973  2017-09-05         f   NaN
    1    12147973  2017-09-04         f   NaN
    2    12147973  2017-09-03         f   NaN
    3    12147973  2017-09-02         f   NaN
    4    12147973  2017-09-01         f   NaN
    (68275, 6)
       listing_id       id        date  reviewer_id reviewer_name  \
    0     1178162  4724140  2013-05-21      4298113       Olivier   
    1     1178162  4869189  2013-05-29      6452964     Charlotte   
    2     1178162  5003196  2013-06-06      6449554     Sebastian   
    3     1178162  5150351  2013-06-15      2215611        Marine   
    4     1178162  5171140  2013-06-16      6848427        Andrew   
    
                                                comments  
    0  My stay at islam's place was really cool! Good...  
    1  Great location for both airport and city - gre...  
    2  We really enjoyed our stay at Islams house. Fr...  
    3  The room was nice and clean and so were the co...  
    4  Great location. Just 5 mins walk from the Airp...  
    


```python
# Visualize the distribtuion of review data to undersand the skewness
fig, axs = plt.subplots(3,2, figsize=(10,15), dpi=80)

sns.distplot(Boston_list["review_scores_rating"], ax=axs[0,0])
sns.distplot(Boston_list["review_scores_accuracy"], ax=axs[0,1])
sns.distplot(Boston_list["review_scores_cleanliness"], ax=axs[1,0])
sns.distplot(Boston_list["review_scores_communication"], ax=axs[1,1])
sns.distplot(Boston_list["review_scores_checkin"], ax=axs[2,0])
sns.distplot(Boston_list["review_scores_value"], ax=axs[2,1])
```

    D:\Program Files\Anaconda3\lib\site-packages\seaborn\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    D:\Program Files\Anaconda3\lib\site-packages\seaborn\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    D:\Program Files\Anaconda3\lib\site-packages\seaborn\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    D:\Program Files\Anaconda3\lib\site-packages\seaborn\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    D:\Program Files\Anaconda3\lib\site-packages\seaborn\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    D:\Program Files\Anaconda3\lib\site-packages\seaborn\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    




    <AxesSubplot:xlabel='review_scores_value', ylabel='Density'>




    
![png](output_4_2.png)
    



```python
Boston_list.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3585 entries, 0 to 3584
    Data columns (total 95 columns):
     #   Column                            Non-Null Count  Dtype  
    ---  ------                            --------------  -----  
     0   id                                3585 non-null   int64  
     1   listing_url                       3585 non-null   object 
     2   scrape_id                         3585 non-null   int64  
     3   last_scraped                      3585 non-null   object 
     4   name                              3585 non-null   object 
     5   summary                           3442 non-null   object 
     6   space                             2528 non-null   object 
     7   description                       3585 non-null   object 
     8   experiences_offered               3585 non-null   object 
     9   neighborhood_overview             2170 non-null   object 
     10  notes                             1610 non-null   object 
     11  transit                           2295 non-null   object 
     12  access                            2096 non-null   object 
     13  interaction                       2031 non-null   object 
     14  house_rules                       2393 non-null   object 
     15  thumbnail_url                     2986 non-null   object 
     16  medium_url                        2986 non-null   object 
     17  picture_url                       3585 non-null   object 
     18  xl_picture_url                    2986 non-null   object 
     19  host_id                           3585 non-null   int64  
     20  host_url                          3585 non-null   object 
     21  host_name                         3585 non-null   object 
     22  host_since                        3585 non-null   object 
     23  host_location                     3574 non-null   object 
     24  host_about                        2276 non-null   object 
     25  host_response_time                3114 non-null   object 
     26  host_response_rate                3114 non-null   object 
     27  host_acceptance_rate              3114 non-null   object 
     28  host_is_superhost                 3585 non-null   object 
     29  host_thumbnail_url                3585 non-null   object 
     30  host_picture_url                  3585 non-null   object 
     31  host_neighbourhood                3246 non-null   object 
     32  host_listings_count               3585 non-null   int64  
     33  host_total_listings_count         3585 non-null   int64  
     34  host_verifications                3585 non-null   object 
     35  host_has_profile_pic              3585 non-null   object 
     36  host_identity_verified            3585 non-null   object 
     37  street                            3585 non-null   object 
     38  neighbourhood                     3042 non-null   object 
     39  neighbourhood_cleansed            3585 non-null   object 
     40  neighbourhood_group_cleansed      0 non-null      float64
     41  city                              3583 non-null   object 
     42  state                             3585 non-null   object 
     43  zipcode                           3547 non-null   object 
     44  market                            3571 non-null   object 
     45  smart_location                    3585 non-null   object 
     46  country_code                      3585 non-null   object 
     47  country                           3585 non-null   object 
     48  latitude                          3585 non-null   float64
     49  longitude                         3585 non-null   float64
     50  is_location_exact                 3585 non-null   object 
     51  property_type                     3582 non-null   object 
     52  room_type                         3585 non-null   object 
     53  accommodates                      3585 non-null   int64  
     54  bathrooms                         3571 non-null   float64
     55  bedrooms                          3575 non-null   float64
     56  beds                              3576 non-null   float64
     57  bed_type                          3585 non-null   object 
     58  amenities                         3585 non-null   object 
     59  square_feet                       56 non-null     float64
     60  price                             3585 non-null   object 
     61  weekly_price                      892 non-null    object 
     62  monthly_price                     888 non-null    object 
     63  security_deposit                  1342 non-null   object 
     64  cleaning_fee                      2478 non-null   object 
     65  guests_included                   3585 non-null   int64  
     66  extra_people                      3585 non-null   object 
     67  minimum_nights                    3585 non-null   int64  
     68  maximum_nights                    3585 non-null   int64  
     69  calendar_updated                  3585 non-null   object 
     70  has_availability                  0 non-null      float64
     71  availability_30                   3585 non-null   int64  
     72  availability_60                   3585 non-null   int64  
     73  availability_90                   3585 non-null   int64  
     74  availability_365                  3585 non-null   int64  
     75  calendar_last_scraped             3585 non-null   object 
     76  number_of_reviews                 3585 non-null   int64  
     77  first_review                      2829 non-null   object 
     78  last_review                       2829 non-null   object 
     79  review_scores_rating              2772 non-null   float64
     80  review_scores_accuracy            2762 non-null   float64
     81  review_scores_cleanliness         2767 non-null   float64
     82  review_scores_checkin             2765 non-null   float64
     83  review_scores_communication       2767 non-null   float64
     84  review_scores_location            2763 non-null   float64
     85  review_scores_value               2764 non-null   float64
     86  requires_license                  3585 non-null   object 
     87  license                           0 non-null      float64
     88  jurisdiction_names                0 non-null      float64
     89  instant_bookable                  3585 non-null   object 
     90  cancellation_policy               3585 non-null   object 
     91  require_guest_profile_picture     3585 non-null   object 
     92  require_guest_phone_verification  3585 non-null   object 
     93  calculated_host_listings_count    3585 non-null   int64  
     94  reviews_per_month                 2829 non-null   float64
    dtypes: float64(18), int64(15), object(62)
    memory usage: 2.6+ MB
    


```python
Boston_cld.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1308890 entries, 0 to 1308889
    Data columns (total 4 columns):
     #   Column      Non-Null Count    Dtype 
    ---  ------      --------------    ----- 
     0   listing_id  1308890 non-null  int64 
     1   date        1308890 non-null  object
     2   available   1308890 non-null  object
     3   price       643037 non-null   object
    dtypes: int64(1), object(3)
    memory usage: 39.9+ MB
    


```python
Boston_list.dropna(axis=1, how ='all', inplace =True)
Boston_list.shape
```




    (3585, 91)




```python
## ==== data preparation for modeling (Boston_list)
# fill na, skip the gap, jump to the next valid observation
Boston_list['host_response_rate'].fillna(method ='backfill', inplace =True)
Boston_list['host_acceptance_rate'].fillna(method='backfill', inplace=True)
Boston_list['host_response_time'].fillna(method='backfill',inplace=True)

# convert the string to int
Boston_list['host_response_rate'] =(Boston_list['host_response_rate'].str[:-1].astype(int))
Boston_list['host_acceptance_rate'] = (Boston_list['host_acceptance_rate'].str[:-1].astype(int))
```


```python
# the columns with numerous na values, drop na rows
Boston_list.dropna(axis=0, subset=["bathrooms","bedrooms","beds"], inplace =True)

# And subset the three columns to create the room dataframe
Boston_rooms = Boston_list[["bathrooms","bedrooms","beds"]]

#convert from string to integer
for room in Boston_rooms:
    Boston_rooms[room].astype(int)

# after this converting, we can put the three converted columns back to the original dataframe called list
Boston_list[["bathrooms","bedrooms","beds"]] = Boston_rooms[["bathrooms","bedrooms","beds"]]
```


```python
# replace the price "$," to normal numbers
# and replace the nas with medians
for prices in ["cleaning_fee","security_deposit","price"]:
    Boston_list[prices] = Boston_list[prices].map(lambda x:x.replace("$","").replace(",",''), na_action = 'ignore')
    Boston_list[prices] = Boston_list[prices].astype(float)
    Boston_list[prices].fillna(Boston_list[prices].median(), inplace = True)
```


```python
Boston_list.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 3557 entries, 0 to 3584
    Data columns (total 91 columns):
     #   Column                            Non-Null Count  Dtype  
    ---  ------                            --------------  -----  
     0   id                                3557 non-null   int64  
     1   listing_url                       3557 non-null   object 
     2   scrape_id                         3557 non-null   int64  
     3   last_scraped                      3557 non-null   object 
     4   name                              3557 non-null   object 
     5   summary                           3431 non-null   object 
     6   space                             2503 non-null   object 
     7   description                       3557 non-null   object 
     8   experiences_offered               3557 non-null   object 
     9   neighborhood_overview             2162 non-null   object 
     10  notes                             1604 non-null   object 
     11  transit                           2287 non-null   object 
     12  access                            2089 non-null   object 
     13  interaction                       2024 non-null   object 
     14  house_rules                       2377 non-null   object 
     15  thumbnail_url                     2965 non-null   object 
     16  medium_url                        2965 non-null   object 
     17  picture_url                       3557 non-null   object 
     18  xl_picture_url                    2965 non-null   object 
     19  host_id                           3557 non-null   int64  
     20  host_url                          3557 non-null   object 
     21  host_name                         3557 non-null   object 
     22  host_since                        3557 non-null   object 
     23  host_location                     3546 non-null   object 
     24  host_about                        2255 non-null   object 
     25  host_response_time                3557 non-null   object 
     26  host_response_rate                3557 non-null   int32  
     27  host_acceptance_rate              3557 non-null   int32  
     28  host_is_superhost                 3557 non-null   object 
     29  host_thumbnail_url                3557 non-null   object 
     30  host_picture_url                  3557 non-null   object 
     31  host_neighbourhood                3223 non-null   object 
     32  host_listings_count               3557 non-null   int64  
     33  host_total_listings_count         3557 non-null   int64  
     34  host_verifications                3557 non-null   object 
     35  host_has_profile_pic              3557 non-null   object 
     36  host_identity_verified            3557 non-null   object 
     37  street                            3557 non-null   object 
     38  neighbourhood                     3020 non-null   object 
     39  neighbourhood_cleansed            3557 non-null   object 
     40  city                              3555 non-null   object 
     41  state                             3557 non-null   object 
     42  zipcode                           3521 non-null   object 
     43  market                            3543 non-null   object 
     44  smart_location                    3557 non-null   object 
     45  country_code                      3557 non-null   object 
     46  country                           3557 non-null   object 
     47  latitude                          3557 non-null   float64
     48  longitude                         3557 non-null   float64
     49  is_location_exact                 3557 non-null   object 
     50  property_type                     3554 non-null   object 
     51  room_type                         3557 non-null   object 
     52  accommodates                      3557 non-null   int64  
     53  bathrooms                         3557 non-null   float64
     54  bedrooms                          3557 non-null   float64
     55  beds                              3557 non-null   float64
     56  bed_type                          3557 non-null   object 
     57  amenities                         3557 non-null   object 
     58  square_feet                       56 non-null     float64
     59  price                             3557 non-null   float64
     60  weekly_price                      888 non-null    object 
     61  monthly_price                     874 non-null    object 
     62  security_deposit                  3557 non-null   float64
     63  cleaning_fee                      3557 non-null   float64
     64  guests_included                   3557 non-null   int64  
     65  extra_people                      3557 non-null   object 
     66  minimum_nights                    3557 non-null   int64  
     67  maximum_nights                    3557 non-null   int64  
     68  calendar_updated                  3557 non-null   object 
     69  availability_30                   3557 non-null   int64  
     70  availability_60                   3557 non-null   int64  
     71  availability_90                   3557 non-null   int64  
     72  availability_365                  3557 non-null   int64  
     73  calendar_last_scraped             3557 non-null   object 
     74  number_of_reviews                 3557 non-null   int64  
     75  first_review                      2807 non-null   object 
     76  last_review                       2807 non-null   object 
     77  review_scores_rating              2751 non-null   float64
     78  review_scores_accuracy            2740 non-null   float64
     79  review_scores_cleanliness         2745 non-null   float64
     80  review_scores_checkin             2743 non-null   float64
     81  review_scores_communication       2745 non-null   float64
     82  review_scores_location            2741 non-null   float64
     83  review_scores_value               2742 non-null   float64
     84  requires_license                  3557 non-null   object 
     85  instant_bookable                  3557 non-null   object 
     86  cancellation_policy               3557 non-null   object 
     87  require_guest_profile_picture     3557 non-null   object 
     88  require_guest_phone_verification  3557 non-null   object 
     89  calculated_host_listings_count    3557 non-null   int64  
     90  reviews_per_month                 2807 non-null   float64
    dtypes: float64(17), int32(2), int64(15), object(57)
    memory usage: 2.5+ MB
    


```python
# We can tell from the diagrams, the means and medians in the six columns, they are not normally distributed, but left skewed. 
# medians to replace could not work; we can try mean
for feature in ["review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", 
                "review_scores_checkin", "review_scores_communication", "review_scores_location", 
                "review_scores_value", "reviews_per_month"]:
    Boston_list[feature].fillna(Boston_list[feature].mean(), inplace = True)

```


```python
## create dataframe for prediction
Boston_inf = Boston_list.select_dtypes(include=['int64','float64'])

# add categorical columns to Boston_inf
Boston_inf[['superhost', 'room_type', 'neighbourhood_cleansed', 'cancellation_policy', 'property_type', 'host_response_time']] = Boston_list[['host_is_superhost', 
                                                           'room_type', 'neighbourhood_cleansed', 
                                                           'cancellation_policy', 'property_type', 'host_response_time']]
# drop square_feet
Boston_inf.drop(labels=['square_feet'],axis=1, inplace=True)
Boston_inf.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 3557 entries, 0 to 3584
    Data columns (total 37 columns):
     #   Column                          Non-Null Count  Dtype  
    ---  ------                          --------------  -----  
     0   id                              3557 non-null   int64  
     1   scrape_id                       3557 non-null   int64  
     2   host_id                         3557 non-null   int64  
     3   host_listings_count             3557 non-null   int64  
     4   host_total_listings_count       3557 non-null   int64  
     5   latitude                        3557 non-null   float64
     6   longitude                       3557 non-null   float64
     7   accommodates                    3557 non-null   int64  
     8   bathrooms                       3557 non-null   float64
     9   bedrooms                        3557 non-null   float64
     10  beds                            3557 non-null   float64
     11  price                           3557 non-null   float64
     12  security_deposit                3557 non-null   float64
     13  cleaning_fee                    3557 non-null   float64
     14  guests_included                 3557 non-null   int64  
     15  minimum_nights                  3557 non-null   int64  
     16  maximum_nights                  3557 non-null   int64  
     17  availability_30                 3557 non-null   int64  
     18  availability_60                 3557 non-null   int64  
     19  availability_90                 3557 non-null   int64  
     20  availability_365                3557 non-null   int64  
     21  number_of_reviews               3557 non-null   int64  
     22  review_scores_rating            3557 non-null   float64
     23  review_scores_accuracy          3557 non-null   float64
     24  review_scores_cleanliness       3557 non-null   float64
     25  review_scores_checkin           3557 non-null   float64
     26  review_scores_communication     3557 non-null   float64
     27  review_scores_location          3557 non-null   float64
     28  review_scores_value             3557 non-null   float64
     29  calculated_host_listings_count  3557 non-null   int64  
     30  reviews_per_month               3557 non-null   float64
     31  superhost                       3557 non-null   object 
     32  room_type                       3557 non-null   object 
     33  neighbourhood_cleansed          3557 non-null   object 
     34  cancellation_policy             3557 non-null   object 
     35  property_type                   3554 non-null   object 
     36  host_response_time              3557 non-null   object 
    dtypes: float64(16), int64(15), object(6)
    memory usage: 1.0+ MB
    


```python
# the categorical variables, replace with dummies
cleanup_cols = {"superhost":{"t":1,"f":2},
               "room_type":{"Entire home/apt":1, "Private room":2, "Shared room":3},
               "cancellation_policy":{"moderate": 1, "flexible": 2, "strict": 3, "super_strict_30": 4}
               }
Boston_int_copy = Boston_inf
Boston_int_copy = Boston_int_copy.replace(cleanup_cols)

df_int_dummies = pd.get_dummies(Boston_int_copy)
```


```python
##========================================== Data preparation - Calendar data - Boston_cld
# extract month from dates column
Boston_cld['month'] = Boston_cld.date.apply(lambda value: value.split('-')[1])
Boston_cld['month'] = Boston_cld['month'].replace({
    '01':'Jan',
    '02':'Feb',
    '03':'Mar',
    '04':'Apr',
    '05': 'May',
    '06': 'Jun',
    '07': 'Jul',
    '08': 'Aug',
    '09': 'Sep',
    '10': 'Oct',
    '11': 'Nov',
    '12': 'Dec'
})

# also, the cld dataframe, replace '$,' with ''
Boston_cld.dropna(axis=0, subset =["price"], inplace = True)
Boston_cld["price"] = Boston_cld['price'].map(lambda x:x.replace("$",'').replace(",",""), na_action ='ignore').astype(float)
Boston_cld.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>listing_id</th>
      <th>date</th>
      <th>available</th>
      <th>price</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>365</th>
      <td>3075044</td>
      <td>2017-08-22</td>
      <td>t</td>
      <td>65.0</td>
      <td>Aug</td>
    </tr>
    <tr>
      <th>366</th>
      <td>3075044</td>
      <td>2017-08-21</td>
      <td>t</td>
      <td>65.0</td>
      <td>Aug</td>
    </tr>
    <tr>
      <th>367</th>
      <td>3075044</td>
      <td>2017-08-20</td>
      <td>t</td>
      <td>65.0</td>
      <td>Aug</td>
    </tr>
    <tr>
      <th>368</th>
      <td>3075044</td>
      <td>2017-08-19</td>
      <td>t</td>
      <td>75.0</td>
      <td>Aug</td>
    </tr>
    <tr>
      <th>369</th>
      <td>3075044</td>
      <td>2017-08-18</td>
      <td>t</td>
      <td>75.0</td>
      <td>Aug</td>
    </tr>
  </tbody>
</table>
</div>




```python
## ================ develop a model to predict property pricing
#Assiang target variables and explanatory variables
y = df_int_dummies.price
X = df_int_dummies.drop(labels = ["price"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=42)

```


```python

# a function to precess the data as learned in the previous chapter
def model_evaluate(model, x_train,y_train, x_test, y_test):
    '''input:
    model: the prediction algorithm we want to apply
    x_train: training dataset
    y_test: 
    
    1_ fit the model to training dataset
    2_ predict the target variable fot the test dataset
    3_ compare the evaluate the predicted target variable and actual values for the test dataset
    
    output:
    1) mean absolute error for models applied
    2_scatter plot for the difference between predicted value and acture values
    '''
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, pred)
    print("MAE from {0}:{1}".format(model, mae))
    fig = plt.figure(figsize=(12,8), dpi=80)
    ax1 = fig.add_subplot(111)
    lines = plt.plot(y_test - pred, marker ='o',linestyle='')
    ax1.set_xlabel('True Values_{}'.format(model))
    ax1.set_ylabel('Predictions_{}'.format(model))

    
# random forest model
model_rf = RandomForestRegressor(n_estimators=100, random_state=0)
model_evaluate(model_rf, x_train, y_train, x_test, y_test)

model_lr = LinearRegression()
model_evaluate(model_lr, x_train, y_train, x_test, y_test)
```

    MAE from RandomForestRegressor(random_state=0):47.16267790262172
    MAE from LinearRegression():171.42041198501872
    


    
![png](output_17_1.png)
    



    
![png](output_17_2.png)
    



```python
##=========== data visualization and answer questions
# number of rooms and room_type
Boston_inf.groupby(['room_type']).count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>scrape_id</th>
      <th>host_id</th>
      <th>host_listings_count</th>
      <th>host_total_listings_count</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>...</th>
      <th>review_scores_communication</th>
      <th>review_scores_location</th>
      <th>review_scores_value</th>
      <th>calculated_host_listings_count</th>
      <th>reviews_per_month</th>
      <th>superhost</th>
      <th>neighbourhood_cleansed</th>
      <th>cancellation_policy</th>
      <th>property_type</th>
      <th>host_response_time</th>
    </tr>
    <tr>
      <th>room_type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Entire home/apt</th>
      <td>2113</td>
      <td>2113</td>
      <td>2113</td>
      <td>2113</td>
      <td>2113</td>
      <td>2113</td>
      <td>2113</td>
      <td>2113</td>
      <td>2113</td>
      <td>2113</td>
      <td>...</td>
      <td>2113</td>
      <td>2113</td>
      <td>2113</td>
      <td>2113</td>
      <td>2113</td>
      <td>2113</td>
      <td>2113</td>
      <td>2113</td>
      <td>2111</td>
      <td>2113</td>
    </tr>
    <tr>
      <th>Private room</th>
      <td>1365</td>
      <td>1365</td>
      <td>1365</td>
      <td>1365</td>
      <td>1365</td>
      <td>1365</td>
      <td>1365</td>
      <td>1365</td>
      <td>1365</td>
      <td>1365</td>
      <td>...</td>
      <td>1365</td>
      <td>1365</td>
      <td>1365</td>
      <td>1365</td>
      <td>1365</td>
      <td>1365</td>
      <td>1365</td>
      <td>1365</td>
      <td>1364</td>
      <td>1365</td>
    </tr>
    <tr>
      <th>Shared room</th>
      <td>79</td>
      <td>79</td>
      <td>79</td>
      <td>79</td>
      <td>79</td>
      <td>79</td>
      <td>79</td>
      <td>79</td>
      <td>79</td>
      <td>79</td>
      <td>...</td>
      <td>79</td>
      <td>79</td>
      <td>79</td>
      <td>79</td>
      <td>79</td>
      <td>79</td>
      <td>79</td>
      <td>79</td>
      <td>79</td>
      <td>79</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 36 columns</p>
</div>




```python
# visualizing the data to room_type and their mean price for each type
fig, axs = plt.subplots(1,2, figsize=(13,5), dpi=80)
sns.countplot(x='room_type', data=Boston_inf, 
              palette='cool', ax = axs[0]).set_title("Number of Properties based on Room Type")
axs[0].set_xlabel('Room Type')

# visualize and understand the distribution of pricing among the each room type
sns.distplot(Boston_inf[Boston_inf.room_type =='Private room']['price'],
            kde=False, ax=axs[1], label = 'Private room')
sns.distplot(Boston_inf[Boston_inf.room_type =='Shared room']['price'],
            kde=False, ax=axs[1], label = 'Shared room')
sns.distplot(Boston_inf[Boston_inf.room_type =='Entire home/apt']['price'],
            kde=False, ax=axs[1], label = 'Entire home/apt')

axs[1].set_xlim(0,600)
axs[1].set_title('Room Type')
axs[1].set_xlabel('Price ($)')
axs[1].legend()
```

    D:\Program Files\Anaconda3\lib\site-packages\seaborn\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    




    <matplotlib.legend.Legend at 0x2b800dd2ca0>




    
![png](output_19_2.png)
    



```python
## visualize the hotels in Boston map
BBox = (Boston_inf.longitude.min(), Boston_inf.longitude.max(),Boston_inf.latitude.min(), Boston_inf.latitude.max())
BBox
```




    (-71.17178882136899, -71.00009991969033, 42.23594180770681, 42.38998167884297)




```python
ruh_m = plt.imread('D:/Download/DS_class/Code_To_Practice/Boston_Airbnb/map.png')
#plt.gcf().set_size_inches(12,12)
fig, ax = plt.subplots(figsize =(24,20))
ax.scatter(Boston_inf.longitude, Boston_inf.latitude, zorder=1, alpha= 0.2, c='b', s=10)
ax.set_title('Plotting Spatial Data on Boston Map')
ax.set_xlim(BBox[0], BBox[1])
ax.set_ylim(BBox[2], BBox[3])
ax.imshow(ruh_m, zorder=0, extent = BBox, aspect='equal')
```




    <matplotlib.image.AxesImage at 0x2b801b43a00>




    
![png](output_21_1.png)
    



```python
# visualize the data between the selected features against price column

fig, axs = plt.subplots(3,2, figsize=(25,30), dpi=80)

df_roomtype = Boston_inf.groupby(['room_type','bedrooms'], as_index = False).mean()

## Neighborhoods
neighbourhoods = Boston_inf.groupby('neighbourhood_cleansed').agg(
    {
        'price':np.mean
    }
).reset_index()
sns.barplot(
    x='price', y ='neighbourhood_cleansed',
    data=neighbourhoods.sort_values('price'),
    orient='h',
    palette='muted',
    ax=axs[0,0]
)
axs[0,0].set_title('Neighbourhood and Pricing')
axs[0,0].set_xlabel('Mean Price ($)')
axs[0,0].set_ylabel('')

sns.barplot(x="room_type", y='price', hue='bedrooms', data= df_roomtype, palette= 'muted', ax=axs[0,1])

axs[0,1].set_title('Price difference in each room type based on bedrooms')
axs[0,1].set_xlabel('Room Type')
axs[0,1].legend()

# Cancellation policy
sns.boxplot(x='price', y='cancellation_policy', fliersize=1, linewidth=0.75,
            data=Boston_inf, palette='muted', ax=axs[1, 0],
            order=['flexible', 'moderate', 'strict', 'super_strict_30'])
axs[1, 0].set_xlim(0, 600)
axs[1, 0].set_title('Cancellation Policy and Pricing')
axs[1, 0].set_xlabel('Price ($)')
axs[1, 0].set_ylabel('')

# Property type
sns.boxplot(x='price', y='property_type', fliersize=1, linewidth=0.75,
            data=Boston_inf, palette='muted', ax=axs[1, 1])
axs[1, 1].set_xlim(0, 600)
axs[1, 1].set_title('Property Type and Pricing')
axs[1, 1].set_xlabel('Price ($)')
axs[1, 1].set_ylabel('')


#calendar
cats = ['Jan', 'Feb', 'Mar', 'Apr','May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
Boston_cld['month'] = pd.Categorical(Boston_cld['month'], ordered=True, categories=cats)

plt.figure(figsize=(10,5))

sns.lineplot(data = Boston_cld, x = "month", y = "price", sort=True, ax = axs[2,0])

axs[2, 0].set_title("Month and Price")
axs[2, 0].set_xlabel('Month')
axs[2, 0].set_ylabel('Price ($)')


#Superhost
sns.barplot(x="superhost", y="price", hue= 'room_type', data= Boston_inf, 
            palette= 'muted', ax=axs[2, 1])

axs[2, 1].set_title('Price difference in each room type based on superhost')
axs[2, 1].set_xlabel('superhost t/f')
axs[2, 1].set_ylabel('Price ($)')
axs[2, 1].legend()


plt.tight_layout()
plt.show();
```


    
![png](output_22_0.png)
    



    <Figure size 720x360 with 0 Axes>



```python
# count the number of superhost in the listing
ax1 = sns.countplot(x="superhost", data=Boston_inf, palette='cool').set_title("Number of superhost availabe in the listing")

```


    
![png](output_23_0.png)
    



```python
# analyze the response time between superhost and other host
# group the dataframe with respect to superhost and mean of the each response time

grp_res = Boston_inf.groupby(['superhost','host_response_time']).size()
grp_host = Boston_inf.groupby(['superhost']).size()
grp_res_mean =grp_res/grp_host

#reset index
grp_res_mean = grp_res_mean.reset_index(name='counts')
grp_res_mean
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>superhost</th>
      <th>host_response_time</th>
      <th>counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>f</td>
      <td>a few days or more</td>
      <td>0.015556</td>
    </tr>
    <tr>
      <th>1</th>
      <td>f</td>
      <td>within a day</td>
      <td>0.153333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>f</td>
      <td>within a few hours</td>
      <td>0.394603</td>
    </tr>
    <tr>
      <th>3</th>
      <td>f</td>
      <td>within an hour</td>
      <td>0.436508</td>
    </tr>
    <tr>
      <th>4</th>
      <td>t</td>
      <td>within a day</td>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>5</th>
      <td>t</td>
      <td>within a few hours</td>
      <td>0.351351</td>
    </tr>
    <tr>
      <th>6</th>
      <td>t</td>
      <td>within an hour</td>
      <td>0.557740</td>
    </tr>
  </tbody>
</table>
</div>




```python
# visualize and compare the host respose time between superhost and other host
import plotly.express as px
px.bar(grp_res_mean, x='superhost', y='counts', color='host_response_time', hover_data=['host_response_time'], barmode='stack', width=650, height= 450, title='average response time per host type')
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-114-c32ab608b36e> in <module>
          1 # visualize and compare the host respose time between superhost and other host
    ----> 2 import plotly.express as px
          3 px.bar(grp_res_mean, x='superhost', y='counts', color='host_response_time', hover_data=['host_response_time'], barmode='stack', width=650, height= 450, title='average response time per host type')
    

    ModuleNotFoundError: No module named 'plotly'



```python

```
