#Code for faster modeliing of large numbers of independent categories
#Author: Samiran Kundu
#Date: 5-05-2019 


#library imports
import numpy as np
import pandas as pd
import time
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from multiprocessing import Pool
import multiprocessing
import pickle
import glob

#class for doing a split model
class SplitModel:

	def __init__(self,splitted_path,result_path,model_path):
		"""function that will initialize the class SplitModel with splitted path(the path where 
		splitted data is stored, result path(the path where all the final result will be stored,
		model path( path where all the 
		models files will be saved"""
		self.splitted_path=splitted_path
		self.result_path=result_path
		self.model_path=model_path
		self.spcl_char="___"

	def splitwise_model(self,uuid):
		"""function for modelling for indivisual indivisual component(i have here shown the example with
		linear regression only and one can edit it in the way one want and use it)"""
		input_path=self.splitted_path+uuid+".csv"
		train_data=pd .read_csv(input_path,index_col=0).drop("user_id",axis=1) 
		train_data["Target_scaled"]=(train_data["Target"]-min(train_data["Target"]))/(max(train_data["Target"])-min(train_data["Target"]))
		train_data=train_data.drop("Target",axis=1)
		model = LinearRegression()
		X =train_data.loc[:, train_data.columns != 'Target_scaled']
		y =train_data['Target_scaled']
		x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
		model.fit(x_train,y_train)
		r2_test = r2_score(y_test,model.predict(x_test))
		r2_train = r2_score(y_train,model.predict(x_train))		
		#For dumping the modelling file
		self.model_dump(uuid,model)		
		print("Completed for {} with test accuracy {} and train accuracy {}".format(uuid,r2_test,r2_train))
		return(uuid+self.spcl_char+str(r2_test)+self.spcl_char+str(r2_train))

	def model_dump(self,uuid,model):
		"""function for dumping the models"""
		filename=self.model_path+uuid+".sav"
		pickle.dump(model, open(filename, 'wb'))	
		
	def model_train(self):
		"""the function that will be called by the user during the training porpose"""
		#Data read
		start_time=time.time()
		#List all the files in the splitted directory
		all_files=glob.glob(self.splitted_path+"*")
		splitted_list=[x.split("\\")[1].split(".")[0] for x in all_files] #Change the value and regex expression according to need
		simulteneous_process=10
		p=Pool(simulteneous_process)
		results=p.map(self.splitwise_model,splitted_list[:100])
		final_results=pd.DataFrame()
		final_results["concat"]=results
		#Define the column names
		cols = ['uuid', 'r2_test', 'r2_train']
		final_results=pd.DataFrame(final_results.concat.str.split(spcl_char).tolist(), columns=cols)
		final_results.to_csv(self.result_path+"Results.csv")
		print("Time taken for modelling {} sec".format(time.time()-start_time))
		return(final_results)
	
if __name__=='__main__':
	splitted_path="C:/Users/Mainak Kundu/Desktop/Samiran/Codes/Multiprocessing/Data/Splitted_User/" #Splitted path
	result_path="C:/Users/Mainak Kundu/Desktop/Samiran/Codes/Multiprocessing/Result/" #Path for final model and result output
	model_path="C:/Users/Mainak Kundu/Desktop/Samiran/Codes/Multiprocessing/Model_path/"
	spcl_char="___" 
	a=SplitModel(splitted_path,result_path,model_path) #creating a object of class SplitModel with splitted ,model and result path
	final_results=a.model_train() #calling model train for training on the indisual files of the splitted path and coleecting the final results as dataframe 
	print(final_results) 
	
	