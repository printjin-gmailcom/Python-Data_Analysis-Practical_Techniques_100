import warnings
warnings.filterwarnings('ignore')



import pandas as pd
uselog = pd.read_csv('use_log.csv')
print(len(uselog))
uselog.head()

customer = pd.read_csv('customer_master.csv')
print(len(customer))
customer.head()

class_master = pd.read_csv('class_master.csv')
print(len(class_master))
class_master.head()

campaign_master = pd.read_csv('campaign_master.csv')
print(len(campaign_master))
campaign_master.head()



customer_join = pd.merge(customer, class_master, on="class", how="left")
customer_join = pd.merge(customer_join, campaign_master, on="campaign_id", how="left")
customer_join.head()

print(len(customer))
print(len(customer_join))

customer_join.isnull().sum()



customer_join.groupby("class_name").count()["customer_id"]

customer_join.groupby("campaign_name").count()["customer_id"]

customer_join.groupby("gender").count()["customer_id"]

customer_join.groupby("is_deleted").count()["customer_id"]

customer_join["start_date"] = pd.to_datetime(customer_join["start_date"])
customer_start = customer_join.loc[customer_join["start_date"]>pd.to_datetime("20180401")]
print(len(customer_start))



customer_join["end_date"] = pd.to_datetime(customer_join["end_date"])
customer_newer = customer_join.loc[(customer_join["end_date"]>=pd.to_datetime("20190331"))|(customer_join["end_date"].isna())]
print(len(customer_newer))
customer_newer["end_date"].unique()

customer_newer.groupby("class_name").count()["customer_id"]

customer_newer.groupby("campaign_name").count()["customer_id"]

customer_newer.groupby("gender").count()["customer_id"]



uselog["usedate"] = pd.to_datetime(uselog["usedate"])
uselog["연월"] = uselog["usedate"].dt.strftime("%Y%m")
uselog_months = uselog.groupby(["연월","customer_id"],as_index=False).count()
uselog_months.rename(columns={"log_id":"count"}, inplace=True)
del uselog_months["usedate"]
uselog_months.head()

uselog_customer = uselog_months.groupby("customer_id").agg(["mean", "median", "max", "min" ])["count"]
uselog_customer = uselog_customer.reset_index(drop=False)
uselog_customer.head()



uselog["weekday"] = uselog["usedate"].dt.weekday
uselog_weekday = uselog.groupby(["customer_id","연월","weekday"],
                                as_index=False).count()[["customer_id","연월", "weekday","log_id"]]
uselog_weekday.rename(columns={"log_id":"count"}, inplace=True)
uselog_weekday.head()

uselog_weekday = uselog_weekday.groupby("customer_id",as_index=False).max()[["customer_id", "count"]]
uselog_weekday["routine_flg"] = 0
uselog_weekday["routine_flg"] = uselog_weekday["routine_flg"].where(uselog_weekday["count"]<4, 1)
uselog_weekday.head()



customer_join = pd.merge(customer_join, uselog_customer, on="customer_id", how="left")
customer_join = pd.merge(customer_join, uselog_weekday[["customer_id", "routine_flg"]], on="customer_id", how="left")
customer_join.head()

customer_join.isnull().sum()



from dateutil.relativedelta import relativedelta
customer_join["calc_date"] = customer_join["end_date"]
customer_join["calc_date"] = customer_join["calc_date"].fillna(pd.to_datetime("20190430"))
customer_join["membership_period"] = 0
for i in range(len(customer_join)):
    delta = relativedelta(customer_join["calc_date"].iloc[i], customer_join["start_date"].iloc[i])
    customer_join["membership_period"].iloc[i] = delta.years*12 + delta.months
customer_join.head()



customer_join[["mean", "median", "max", "min"]].describe()

customer_join.groupby("routine_flg").count()["customer_id"]

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline
plt.hist(customer_join["membership_period"])



customer_end = customer_join.loc[customer_join["is_deleted"]==1]
customer_end.describe()

customer_stay = customer_join.loc[customer_join["is_deleted"]==0]
customer_stay.describe()

customer_join.to_csv("customer_join.csv", index=False)
