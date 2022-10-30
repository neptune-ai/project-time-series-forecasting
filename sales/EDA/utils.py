
def pre_process_data(df):
    
    # process dates and create year, month and week features
    df["Date"] = pd.to_datetime(df.Date)
    df["Year"] = pd.DatetimeIndex(df.Date).year
    df["Month"] = pd.DatetimeIndex(df.Date).month
    df["Week"] = df.Date.dt.isocalendar().week
    
    # convert from F to C
    df["Temperature"] = pd.DataFrame((df.Temperature.values - 32) * 5/9)    
    
    # fill missing values 
    df["MarkDown1"].fillna(df["MarkDown1"].mean(),inplace=True)
    df["MarkDown2"].fillna(df["MarkDown2"].mean(),inplace=True)
    df["MarkDown3"].fillna(df["MarkDown3"].mean(),inplace=True)
    df["MarkDown4"].fillna(df["MarkDown4"].mean(),inplace=True)
    df["MarkDown5"].fillna(df["MarkDown5"].mean(),inplace=True)
    
    # change position of weekly sales column to last
    weekly_sales = df.pop("Weekly_Sales")
    df.insert(len(df.columns), "Weekly_Sales", weekly_sales)
    return df

def load_data(path, cache=False, all_df=False):
    if os.path.exists("complete_set.csv") and cache == True:
        return pd.read_csv("complete_set.csv", index_col=0)
    
    df_train = pd.read_csv(f"{path}/train.csv")
    df_test = pd.read_csv(f"{path}/test.csv")
    df_fts = pd.read_csv(f"{path}/features.csv")
    df_stores = pd.read_csv(f"{path}/stores.csv")
    df = pd.merge(df_train, df_stores)
    df = pd.merge(df, df_fts)
    df = pre_process_data(df)
    df.to_csv("complete_set.csv")
    
    if all_df:
        return df, df_train, df_test, df_fts, df_stores
    else:
        return df

def remove_outliers(df,column,n_std):
    
    print('Working on column: {}'.format(column))

    mean = df[column].mean()
    sd = df[column].std()

    df = df[(df[column] <= mean+(n_std*sd))]
    df.loc[df['Weekly_Sales']<0, 'Weekly_Sales'] = 0 # remove negative sales 

    return df
