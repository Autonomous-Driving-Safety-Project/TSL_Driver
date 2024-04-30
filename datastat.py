import glob
import pandas as pd
all_files = glob.glob('./log/2024-04-30_16-01-25/*.csv')
print(all_files)
data = pd.read_csv(all_files[0], index_col=0)
data.iloc[-1:]["done"]
df=pd.DataFrame({"frame_ave",
            "speed_ave",
            "TTC_front_ave",
            "TTC_rear_ave",
            "done_ave",
            "truncated_ave"})
df=pd.DataFrame(columns=["frame_max", "speed_ave", "TTC_front_min", "TTC_rear_min", "done", "truncated"])
for file in all_files:
    data = pd.read_csv(file, index_col=0)
    #df=df.append(pd.DataFrame([[data["frame"].max(),data["speed"].mean(),data["TTC_front"].min(),data["TTC_rear"].min(),data.iloc[-1]["done"].min(),data.iloc[-1]["truncated"].min()]],columns=df.columns))
    df=pd.concat([df,
        pd.DataFrame({
            "frame_max": data["frame"].max(),
            "speed_ave": data["speed"].mean(),
            "TTC_front_min": data["TTC_front"].min(),
            "TTC_rear_min": data["TTC_rear"].min(),
            "done": 1 if data.iloc[-1]["done"] else 0,
                #data.iloc[-1:]["done"],
            "truncated": 1 if data.iloc[-1]["truncated"] else 0,
                #data.iloc[-1:]["truncated"]
            },index=[0])]
    )
df
frame_ave = df["frame_max"].mean()
speed_ave = df["speed_ave"].mean()
TTC_front_min_ave = df["TTC_front_min"].mean()
TTC_rear_min_ave = df["TTC_rear_min"].mean()
done = df["done"].sum()
truncated = df["truncated"].sum()
rate = done/(done+truncated)
speed_std = df["speed_ave"].std()
TTC_front_min_std = df["TTC_front_min"].std()

print(rate, speed_ave, TTC_front_min_ave, done+truncated,speed_std,TTC_front_min_std)