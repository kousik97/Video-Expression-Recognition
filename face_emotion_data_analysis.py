import os
import pandas as pd
import matplotlib.pyplot as plt
#Extracting the CSV file as pandas dataframe:
df=pd.read_csv("./Extracted_frames/result.csv",index_col=0)

#Adding time information:
time_temp=df["frame_name"].str.split('-').tolist()
time_temp=[x[1] for x in time_temp]
time_temp=[int(x.split('.')[0])*0.5 for x in time_temp]
df["Time"]=time_temp

#Selecting the cluster/unique face and refining the results:
os.chdir("output")
list_all_clusters=next(os.walk('.'))[1]
print("List of All clusters:")
print(list_all_clusters)
flag=True
while(flag==True):
    input=int(input("For which cluster do you want the emotion?(input the number alone)"))
    if(input<=len(list_all_clusters)):
        flag=False
os.chdir(("pic"+str(input)))
frames=next(os.walk('.'))[-1]
df2=df[df['face_name'].isin(frames)]

#Extracting only the necessary columns:
df2=df2.sort(columns="frame_name")
df2.reset_index()
df3=df2[["Anger","Disgust","Fear","Happy","Sad","Surprise","Neutral"]].copy()
df3=df3.reset_index()
del df3["index"]
dominant_emotions=list(df3.idxmax(axis=1))
print(dominant_emotions)

#Zeroing the other emotions:
df4=df3.copy()
emotion_list=["Anger","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
for i in range(0,len(dominant_emotions)):
   df4.xs(i)[emotion_list]=0
   df4.xs(i)[dominant_emotions[i]]=df3.xs(i)[dominant_emotions[i]]
df4.plot(kind="area",x=df2["Time"],subplots=True, figsize=(6, 6),ylim=[0,1])
plt.show()

