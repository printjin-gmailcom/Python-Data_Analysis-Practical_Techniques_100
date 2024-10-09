import pandas as pd
customer_master = pd.read_csv('customer_master.csv')
customer_master.head()



import warnings
warnings.filterwarnings('ignore')

item_master = pd.read_csv('item_master.csv')
item_master.head()

transaction_1 = pd.read_csv('transaction_1.csv')
transaction_1.head()

transaction_detail_1 = pd.read_csv('transaction_detail_1.csv')
transaction_detail_1.head()



transaction_2 = pd.read_csv('transaction_2.csv')
transaction = pd.concat([transaction_1, transaction_2], ignore_index=True)
transaction.head()

print(len(transaction_1))
print(len(transaction_2))
print(len(transaction))

transaction_detail_2 = pd.read_csv('transaction_detail_2.csv')
transaction_detail=pd.concat([transaction_detail_1,transaction_detail_2], ignore_index=True)
transaction_detail.head()



join_data = pd.merge(transaction_detail, transaction[["transaction_id", "payment_date", "customer_id"]],
                     on="transaction_id", how="left")
join_data.head()

print(len(transaction_detail))
print(len(transaction))
print(len(join_data))



join_data = pd.merge(join_data, customer_master, on="customer_id", how="left")
join_data = pd.merge(join_data, item_master, on="item_id", how="left")
join_data.head()



join_data["price"] = join_data["quantity"] * join_data["item_price"]
join_data[["quantity", "item_price","price"]].head()



print(join_data["price"].sum())
print(transaction["price"].sum())

join_data["price"].sum() == transaction["price"].sum()



join_data.isnull().sum()

join_data.describe()

print(join_data["payment_date"].min())
print(join_data["payment_date"].max())



join_data.dtypes

join_data["payment_date"] = pd.to_datetime(join_data["payment_date"])
join_data["payment_month"] = join_data["payment_date"].dt.strftime("%Y%m")
join_data[["payment_date", "payment_month"]].head()

join_data.groupby("payment_month").sum()["price"]



join_data.groupby(["payment_month","item_name"]).sum()[["price", "quantity"]]

pd.pivot_table(join_data, index='item_name', columns='payment_month', values=['price', 'quantity'], aggfunc='sum')



graph_data = pd.pivot_table(join_data, index='payment_month', columns='item_name', values='price', aggfunc='sum')
graph_data.head()

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline
plt.plot(list(graph_data.index), graph_data["PC-A"], label='PC-A')
plt.plot(list(graph_data.index), graph_data["PC-B"], label='PC-B')
plt.plot(list(graph_data.index), graph_data["PC-C"], label='PC-C')
plt.plot(list(graph_data.index), graph_data["PC-D"], label='PC-D')
plt.plot(list(graph_data.index), graph_data["PC-E"], label='PC-E')
plt.legend()
